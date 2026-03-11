#include "src/subgraph/rewrites/fp16_to_fp32.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <vector>

#include "include/xnnpack.h"
#include "src/xnnpack/allocator.h"
#include "src/xnnpack/cache.h"
#include "src/xnnpack/internal.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/node-type.h"
#include "src/xnnpack/subgraph.h"

namespace xnnpack {

namespace {

bool ReplaceInSet(uint32_t* set, uint32_t size, uint32_t old_value,
                  uint32_t new_value) {
  bool replaced = false;
  for (uint32_t i = 0; i < size; i++) {
    if (set[i] == old_value) {
      set[i] = new_value;
      replaced |= true;
    }
  }
  return replaced;
}

enum class OpAction {
  kNone,             // Don't do anything, skip the node.
  kRewrite,          // Force outputs to fp32, insert converts from fp16 input
                     // to fp32
  kNeedsFP16Inputs,  // The inputs must be converted back to fp16 if they werre
                     // rewritten.
  kTransparent,  // If the inputs have been rewritten, the outputs must be also.
  kElide,        // This op should be removed (eg. convert(fp32, fp32)).
};

// Is this op supported when fp16 hardware is missing (allow-list).
// TODO: b/487077315 - Add allow list for supported ops.
OpAction GetOpAction(const xnn_subgraph_t subgraph, const xnn_node& node) {
  switch (node.type) {
    case xnn_node_type_unary_elementwise: {
      switch (node.unary_operator) {
        case xnn_unary_convert: {
          const xnn_value& output = subgraph->values[node.outputs[0]];
          const xnn_value& input = subgraph->values[node.inputs[0]];
          // Elide converts from T to T. These are no-ops that may be introduced
          // by the rewrite.
          if (output.datatype == input.datatype) {
            return OpAction::kElide;
          }
          if (output.datatype == xnn_datatype_fp16 ||
              input.datatype == xnn_datatype_fp16) {
            return OpAction::kNone;
          }
        } break;
        default:
          break;
      }
    } break;
    case xnn_node_type_static_reshape:
      return OpAction::kTransparent;
    default:
      break;
  }
  return OpAction::kRewrite;
}

// Checks if an op has an fp16 input or output and whether we currently
// support that.
bool HasFp16Values(const xnn_subgraph_t subgraph, const xnn_node& node) {
  auto IsFp16Value = [subgraph](uint32_t id) {
    return subgraph->values[id].datatype == xnn_datatype_fp16 ||
           subgraph->values[id].fp16_to_fp32_fallback.was_overwritten;
  };
  return std::any_of(node.inputs, node.inputs + node.num_inputs, IsFp16Value) ||
         std::any_of(node.outputs, node.outputs + node.num_outputs,
                     IsFp16Value);
}

void RemoveFlag(uint32_t& bitfield, uint32_t flag) {
  const uint32_t mask = 0xFFFFFFFF ^ flag;
  bitfield &= mask;
}

}  // namespace

}  // namespace xnnpack

enum xnn_status xnn_subgraph_fallback_from_fp16_to_fp32(
    xnn_subgraph_t subgraph, int optimization_flags) {
  // Maps fp16 value ids to the corresponding fp32 value id if a conversion
  // has been inserted.
  std::vector<uint32_t> fp16_id_to_fp32_id(subgraph->num_values,
                                           XNN_INVALID_VALUE_ID);
  // Maps fp32 value ids to the corresponding fp16 value id if a conversion
  // has been inserted.
  std::vector<uint32_t> fp32_id_to_fp16_id(subgraph->num_values,
                                           XNN_INVALID_VALUE_ID);

  xnn_log_debug("Running fp16 analysis and falling back to fp32.");

  // Go through the graph. Count nodes that will need to be converted.
  const uint32_t original_num_nodes = subgraph->num_nodes;
  for (uint32_t n = 0; n < original_num_nodes; ++n) {
    // Editing the subgraph may reallocate nodes, we need to access the
    // current node through the array each time.
    auto CurrentNode = [=]() -> xnn_node& { return subgraph->nodes[n]; };
    if (CurrentNode().type == xnn_node_type_invalid) {
      continue;
    }

    if (!xnnpack::HasFp16Values(subgraph, CurrentNode())) {
      xnn_log_debug("node %d doesn't have fp16 values", n);
      continue;
    }

    const xnnpack::OpAction op_action =
        xnnpack::GetOpAction(subgraph, CurrentNode());
    if (op_action == xnnpack::OpAction::kNone) {
      continue;
    }

    if (op_action == xnnpack::OpAction::kNeedsFP16Inputs) {
      // Check for overwritten inputs that need to be converted back to fp16.
      for (uint32_t i = 0; i < CurrentNode().num_inputs; i++) {
        // The value is copied because adding new values may invalidate
        // references.
        const xnn_value value = subgraph->values[CurrentNode().inputs[i]];
        if (value.datatype == xnn_datatype_fp32 &&
            value.fp16_to_fp32_fallback.was_overwritten) {
          if (fp32_id_to_fp16_id[value.id] == XNN_INVALID_VALUE_ID) {
            XNN_RETURN_IF_ERROR(xnn_subgraph_add_internal_values(subgraph, 1));
            xnn_value& fp16_value = subgraph->values[subgraph->num_values - 1];
            xnn_value_copy(&fp16_value, &value);
            fp16_value.datatype = xnn_datatype_fp16;
            fp16_value.size = xnn_tensor_get_size(&fp16_value);
            xnn_log_debug("Adding a convert[fp32, fp16](%d, %d) node.",
                          value.id, fp16_value.id);
            xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr,
                             value.id, fp16_value.id,
                             /*flags=*/0);
            fp32_id_to_fp16_id[value.id] = fp16_value.id;
          } else {
            xnn_log_debug("Reusing convert[fp32, fp16](%d, %d) node.", value.id,
                          fp32_id_to_fp16_id[value.id]);
          }
          CurrentNode().inputs[i] = fp32_id_to_fp16_id[value.id];
        }
      }
      continue;
    }

    if (op_action == xnnpack::OpAction::kElide) {
      if (CurrentNode().num_inputs != CurrentNode().num_outputs) {
        xnn_log_error("Node %" PRIu32
                      " should be elided but it doesn't have that same number "
                      "of inputs and outputs.",
                      n);
      } else {
        xnn_node& node = CurrentNode();
        bool cancel_eliding = false;
        for (uint32_t i = 0; i < node.num_inputs; ++i) {
          xnn_value& input = subgraph->values[node.inputs[i]];
          if (xnn_value_is_external_output(input.flags)) {
            cancel_eliding = true;
            break;
          }
        }

        if (cancel_eliding) {
          xnn_log_debug(
              "Node %" PRIu32
              " should be elided but one of its inputs is also a graph output.",
              n);
        } else {
          xnn_log_debug("Eliding node %" PRI_U32, n);
          for (uint32_t i = 0; i < node.num_inputs; ++i) {
            xnn_value& output = subgraph->values[node.outputs[i]];
            xnn_value& input = subgraph->values[node.inputs[i]];
            xnn_node& producer = subgraph->nodes[input.producer];
            // Overwrite the input producer to write to this node's output.
            xnnpack::ReplaceInSet(producer.outputs, producer.num_outputs,
                                  input.id, output.id);
            // Update the input's consumers' input set.
            uint32_t k = std::min(n, input.first_consumer);
            for (int j = 0; j < input.num_consumers && k < subgraph->num_nodes;
                 ++k) {
              xnn_node& node_k = subgraph->nodes[k];
              j += xnnpack::ReplaceInSet(node_k.inputs, node_k.num_inputs,
                                         input.id, output.id);
            }
            output.producer = input.producer;
            output.num_consumers += input.num_consumers - 1;
            output.first_consumer =
                std::min(input.first_consumer, output.first_consumer);
            node.type = xnn_node_type_invalid;
          }
          continue;
        }
      }
    }

    if (op_action == xnnpack::OpAction::kTransparent) {
      // If an input has been rewritten from fp16 to fp32, the outputs should
      // also be rewritten.
      bool needs_output_rewrite = false;
      for (uint32_t i = 0; i < CurrentNode().num_inputs; i++) {
        xnn_value& value = subgraph->values[CurrentNode().inputs[i]];
        if (value.datatype == xnn_datatype_fp32 &&
            value.fp16_to_fp32_fallback.was_overwritten) {
          needs_output_rewrite = true;
          break;
        }
      }
      if (!needs_output_rewrite) {
        continue;
      }
      xnn_log_debug("Node %" PRIu32
                    " is transparent and it's inputs have been rewritten.",
                    n);
    }

    // Force outputs to be fp32.
    for (uint32_t i = 0; i < CurrentNode().num_outputs; i++) {
      xnn_value& value = subgraph->values[CurrentNode().outputs[i]];
      if (value.datatype != xnn_datatype_fp16) {
        continue;
      }

      if (CurrentNode().outputs[i] < subgraph->external_value_ids) {
        // External values can't be overwritten, so we insert a value to get
        // the fp32 output and a conversion to the original external tensor.
        XNN_RETURN_IF_ERROR(xnn_subgraph_add_internal_values(subgraph, 1));
        xnn_value& fp32_value = subgraph->values[subgraph->num_values - 1];
        xnn_value_copy(&fp32_value, &value);
        fp32_value.datatype = xnn_datatype_fp32;
        fp32_value.size = xnn_tensor_get_size(&fp32_value);
        xnnpack::RemoveFlag(
            fp32_value.flags,
            XNN_VALUE_FLAG_EXTERNAL_INPUT | XNN_VALUE_FLAG_EXTERNAL_OUTPUT);
        CurrentNode().outputs[i] = fp32_value.id;
        xnn_log_debug("Adding a convert[fp32, fp16](%d, %d) node.",
                      fp32_value.id, value.id);
        xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr,
                         fp32_value.id, value.id,
                         /*flags=*/0);
      } else {
        xnn_log_debug("Overriding value %d from fp16 to fp32.", value.id);
        value.datatype = xnn_datatype_fp32;
        value.size = xnn_tensor_get_size(&value);
        value.fp16_to_fp32_fallback.was_overwritten = true;
      }
    }

    // Insert conversions to fp32 for fp16 inputs.
    for (uint32_t i = 0; i < CurrentNode().num_inputs; i++) {
      // The value is copied because adding new values may invalidate
      // references.
      const xnn_value value = subgraph->values[CurrentNode().inputs[i]];
      if (value.datatype == xnn_datatype_fp16) {
        if (fp16_id_to_fp32_id[value.id] == XNN_INVALID_VALUE_ID) {
          XNN_RETURN_IF_ERROR(xnn_subgraph_add_internal_values(subgraph, 1));
          xnn_value& fp32_value = subgraph->values[subgraph->num_values - 1];
          xnn_value_copy(&fp32_value, &value);
          fp32_value.datatype = xnn_datatype_fp32;
          fp32_value.size = xnn_tensor_get_size(&fp32_value);
          xnnpack::RemoveFlag(
              fp32_value.flags,
              XNN_VALUE_FLAG_EXTERNAL_INPUT | XNN_VALUE_FLAG_EXTERNAL_OUTPUT);
          if (xnn_value_is_static(value.allocation_type)) {
            xnn_log_debug("Converting static value %d to new fp32 value %d.",
                          value.id, fp32_value.id);
            // We convert static values directly to the new value without
            // inserting a convert node.
            fp32_value.data = xnn_allocate_zero_memory(
                xnn_tensor_get_size(&value) / 2 + XNN_EXTRA_BYTES);
            fp32_value.flags |= XNN_VALUE_FLAG_NEEDS_CLEANUP;
            fp32_value.fp16_to_fp32_fallback.original_data = value.data;
            xnn_run_unary_elementwise_nc(
                xnn_unary_convert, xnn_datatype_fp16, xnn_datatype_fp32,
                /*params=*/nullptr, /*input_quantization=*/nullptr,
                /*output_quantization=*/nullptr, /*flags=*/0,
                /*batch_size=*/xnn_shape_multiply_all_dims(&value.shape),
                /*channels=*/1,
                /*input_stride=*/1, /*output_stride=*/1, /*threadpool=*/nullptr,
                /*input=*/value.data, /*output=*/fp32_value.data);
          } else {
            xnn_log_debug("Adding a convert[fp16, fp32](%d, %d) node.",
                          value.id, fp32_value.id);
            xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr,
                             value.id, fp32_value.id,
                             /*flags=*/0);
          }
          fp16_id_to_fp32_id[value.id] = fp32_value.id;
        }
        CurrentNode().inputs[i] = fp16_id_to_fp32_id[value.id];
      }
    }
  }

  xnn_subgraph_clean_up(subgraph);
  return xnn_status_success;
}

enum xnn_status xnn_subgraph_alias_fp16_fp32_fallback_data(
    xnn_subgraph_t subgraph, xnn_weights_cache_t cache) {
  if (cache) {
    for (uint32_t i = 0; i < subgraph->num_values; ++i) {
      const xnn_value& value = subgraph->values[i];
      if (value.fp16_to_fp32_fallback.original_data) {
        XNN_RETURN_IF_ERROR(xnn_weights_cache_alias_data(
            cache, value.fp16_to_fp32_fallback.original_data, value.data));
      }
    }
  }
  return xnn_status_success;
}
