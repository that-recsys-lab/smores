# SMORES Configuration Reference

This document describes the `SmoresConfig` structure loaded from a YAML file.

## Top-Level Structure

```yaml
simulation:
data:
output:
summary_logger:
consumer:
provider:
platform:
recommender:
triggers:
trigger_tester:
```

## `simulation`

Required fields:

- `experiment_name` (`str`)
- `num_days` (`int > 0`)
- `num_cycles` (`int > 0`)
- `slate_size` (`int > 0`)
- `seed` (`int > 0`)

## `data`

Required fields:

- `directory` (`str`): directory holding input CSVs
- `consumer_file` (`str`)
- `item_file` (`str`)
- `provider_file` (`str`)

## `output`

Required fields:

- `directory` (`str`)
- `consumer_file` (`str`)
- `provider_file` (`str`)
- `choice_file` (`str`)
- `debug_file` (`str`)
- `debug_level` (`str`)
- `use_timestamp` (`bool`)
- `use_parquet` (`bool`)

Optional fields:

- `cycle_file` (`str`, default: `cycle_metrics`)
- `enable_user_journeys` (`bool`, default: `false`)
- `sampled_user_ids` (`list[int] | null`)
- `sampled_user_count` (`int | null`)
- `sampled_user_file` (`str | null`)
- `enable_item_stats` (`bool`, default: `false`)
- `item_stats_file` (`str | null`)

## `summary_logger`

Optional section.

- `enabled` (`bool`, default: `true`)

## `consumer`

Required fields:

- `initial_recommender` (`str`)
- `models.utility` (`list`)
- `models.item_selection` (`list`)
- `types` (`list`)

### `consumer.models.utility[*]`

Fields:

- `name` (`str`)
- `class_name` (`str`)
- `params` (`dict`, optional)

Supported `class_name` values:

- `fixed_utility`
  - params: `value`
- `list_average` (backward-compatible alias)
- `cosine_similarity`
- `dot_product`

### `consumer.models.item_selection[*]`

Fields:

- `name` (`str`)
- `class_name` (`str`)
- `params` (`dict`)

Supported `class_name` values:

- `category_similarity_logit`
  - params: `threshold`, `selection_utility`

### `consumer.types[*]`

Fields:

- `name` (`str`)
- `utility_model` (`str`): references `consumer.models.utility[*].name`
- `item_selection_model` (`str`): references `consumer.models.item_selection[*].name`
- `recommender_choice_model` (`PythonClassConfig`)

Supported `recommender_choice_model.class_name` values:

- `fixed`
  - params: `recommender_name`
- `threshold`
  - params: `threshold`, `beta`
- `ucb`
  - params: `beta`

## `provider`

Required fields:

- `initial_recommender` (`str`)
- `models.utility` (`list`)
- `types` (`list`)

### `provider.models.utility[*]`

Fields:

- `name` (`str`)
- `class_name` (`str`)
- `params` (`dict`)

Supported `class_name` values:

- `click_fixed`
  - params: `value`

### `provider.types[*]`

Fields:

- `name` (`str`)
- `utility_model` (`str`): references `provider.models.utility[*].name`

## `platform`

Current schema:

- `utility_model` (`PythonClassConfig`)

This section is reserved for platform-level utility modeling and may be
experiment-specific.

## `recommender`

Required fields:

- `initial` (`list[str]`)
- `base_recommenders` (`list[PythonClassConfig]`)
- `fallback_recommenders` (`list[PythonClassConfig]`)

Each recommender entry includes:

- `name` (`str`)
- `class_name` (`str`)
- `params` (`dict`, optional)

### Common recommender params

Many recommenders support these in `params`:

- `cold_start_fallback` (`str`): fallback recommender name
- `cold_user_fallback` (`str`): fallback recommender name
- `candidate_multiplier` (`int >= 1`)
- `recency_threshold` (`int >= 0`)
- `cooldown_cycles` (`int >= 0`)
- `item_sampler` (`dict`)
  - `class_name`: currently `rejection_sampler`
  - `params.file_name`: required
  - `params.sampled_item_count`: optional

### Supported recommender `class_name` values

- `fixed_recommender`
  - no required params
- `file_based`
  - params: `file_name`
- `popular`
  - params: `min_user_count`, `min_interaction_count`
- `item_knn`
  - params:
    - `max_neighbors`, `min_neighbors`, `min_similarity`
    - `min_user_count`, `min_interaction_count`, `min_profile_size`
    - optional common params above
- `implicit_mf`
  - params:
    - `embedding_size`, `epochs`, `regularization`, `positive_weight`
    - `min_user_count`, `min_interaction_count`, `min_profile_size`
    - optional: `model_params` or `als_params`
    - optional: `use_gpu`
- `bpr`
  - params:
    - optional: `min_user_count`, `min_interaction_count`, `min_profile_size`
    - optional: `model_params` or `bpr_params`
    - optional: `use_gpu`
- `genre_knn`
  - `item_knn` params plus `genre_feature`
- `genre_implicit_mf`
  - `implicit_mf` params plus `genre_feature`
- `genre_bpr`
  - `bpr` params plus `genre_feature`

## `triggers`

Optional list of `PythonClassConfig` entries.

Supported `class_name` values:

- `initial_burnin`
  - params: `cycle_count`, `recommenders`
  - `repeating` is forced to `False` internally
- `universal_profile`
  - no params required
- `cold_start`
  - no params required
- `user_ownership`
  - no params required
- `save_switch`
  - no params required (testing/debug trigger)

## `trigger_tester`

Optional section:

- `enabled` (`bool`, default: `false`)
- `sample_count` (`int`, default: `5`)
- `seed` (`int | null`)
- `scenario` (`str | null`)

## Quickstart Example

Use the included config:

```bash
python quickstart/bin/run_smores.py --config_file quickstart/ml1m.yaml
```

This writes outputs to `quickstart/output/ml1m-mini/`.
