# Shandong91 heterogeneous Raw Body — CPU Gate 0

- Overall: **PASS**
- CUDA/AMP: **NOT RUN**
- Smoke/formal training: **NOT RUN**
- Formal generation: **NOT RUN**

## Checks

- explicit_resource_type_encoding: **PASS**
- legacy_lead_branch_disabled: **PASS**
- tail_aux_event_partial_disabled: **PASS**
- forward_shape: **PASS**
- nan_inf: **PASS**
- backward: **PASS**
- overall_gradient_finite_nonzero: **PASS**
- wind_gradient_finite_nonzero: **PASS**
- solar_gradient_finite_nonzero: **PASS**
- load_gradient_finite_nonzero: **PASS**
- mask_invariance: **PASS**
- optimizer_core_update: **PASS**
- checkpoint_save_reload: **PASS**
- tensor_device_dtype_shape: **PASS**
- no_24_node_shape_residue: **PASS**

## Objective diagnostic

```json
{
  "elementwise_masked_mean": 1.0882188081741333,
  "resource_wise_balanced_mean_diagnostic_only": 1.0834976020915634,
  "absolute_difference": 0.004721206082569873,
  "relative_difference": 0.004338471313955089,
  "formal_objective_changed": false,
  "largest_valid_count_resource": "Load"
}
```

## Resource supervision and gradients

```json
{
  "supervision": {
    "Wind": {
      "valid_elements": 9653,
      "valid_share": 0.3033054735122227,
      "unweighted_masked_loss": 1.1255985024085777,
      "elementwise_objective_contribution": 0.3414001867576824,
      "objective_contribution_share": 0.3137238432135724
    },
    "Solar": {
      "valid_elements": 9237,
      "valid_share": 0.29023439954753977,
      "unweighted_masked_loss": 1.0055105951269352,
      "elementwise_objective_contribution": 0.29183376381535536,
      "objective_contribution_share": 0.26817562940766326
    },
    "Load": {
      "valid_elements": 12936,
      "valid_share": 0.40646012694023753,
      "unweighted_masked_loss": 1.1193837087391776,
      "elementwise_objective_contribution": 0.45498484434895997,
      "objective_contribution_share": 0.41810051520094177
    }
  },
  "gradients": {
    "Wind": {
      "finite": true,
      "l2_norm": 4.015636383460865,
      "nonzero_elements": 677892,
      "tensor_count": 121,
      "core_backbone_l2_norm": 1.0016142498100145,
      "loss": 1.1282168626785278
    },
    "Solar": {
      "finite": true,
      "l2_norm": 4.010989532907857,
      "nonzero_elements": 677892,
      "tensor_count": 121,
      "core_backbone_l2_norm": 1.0766838309090543,
      "loss": 1.0018723011016846
    },
    "Load": {
      "finite": true,
      "l2_norm": 3.632351918043302,
      "nonzero_elements": 677892,
      "tensor_count": 121,
      "core_backbone_l2_norm": 1.0124935171941531,
      "loss": 1.1249874830245972
    }
  }
}
```
