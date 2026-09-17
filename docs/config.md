# SMORES Configuration Guide

This guide explains the YAML file loaded into `SmoresConfig`. It is meant for
people editing experiments, comparing recommender behavior, or adding new
consumer, provider, recommender, and trigger configurations.

For a complete runnable example, see `quickstart/ml1m.yaml`.

## Contents

- [Quick Mental Model](#quick-mental-model)
- [Minimal Runnable Shape](#minimal-runnable-shape)
- [Config Sections At A Glance](#config-sections-at-a-glance)
- [Common Recipes](#common-recipes)
- [Full Configuration Reference](#full-configuration-reference)
- [Supported Models And Parameters](#supported-models-and-parameters)
- [Name References](#name-references)
- [Validation And Troubleshooting](#validation-and-troubleshooting)
- [Quickstart Example](#quickstart-example)

## Quick Mental Model

A SMORES config describes one simulation run:

```text
simulation   how long the run lasts and which seed to use
data         where input CSV files live
output       where logs and metrics are written
consumer     user types, user utility models, and recommender switching
provider     provider types and provider utility models
platform     platform-level utility model
recommender  available recommendation algorithms and fallbacks
triggers     events that change behavior during a run, or null
```

Several fields reference named objects defined elsewhere in the same config.
Those references must match exactly.

```text
consumer.types[*].utility_model
    -> consumer.models.utility[*].name

consumer.types[*].item_selection_model
    -> consumer.models.item_selection[*].name

provider.types[*].utility_model
    -> provider.models.utility[*].name

recommender.initial
    -> recommender.base_recommenders[*].name

cold_start_fallback / cold_user_fallback
    -> recommender.fallback_recommenders[*].name
```

## Minimal Runnable Shape

Use this as the starting point for a new config. It omits optional logging,
advanced recommenders, and trigger definitions.

```yaml
simulation:
  experiment_name: my_experiment
  num_days: 2
  num_cycles: 2
  slate_size: 10
  seed: 20250702

data:
  directory: data/simulation
  consumer_file: consumers.csv
  item_file: items.csv
  provider_file: providers.csv

output:
  directory: output/my_experiment
  consumer_file: consumer_utility
  provider_file: provider_utility
  choice_file: rchoice_utility
  debug_file: debug
  debug_level: debug
  use_timestamp: false
  use_parquet: true

summary_logger:
  enabled: true

consumer:
  initial_recommender: Generic
  models:
    utility:
      - name: Dot Product
        class_name: dot_product
    item_selection:
      - name: Category Similarity
        class_name: category_similarity_logit
        params:
          threshold: 0.2
          selection_utility: 1.0
  types:
    - name: Generic
      utility_model: Dot Product
      item_selection_model: Category Similarity
      recommender_choice_model:
        class_name: fixed
        params:
          recommender_name: Generic

provider:
  initial_recommender: __all__
  models:
    utility:
      - name: Click Fixed 1.0
        class_name: click_fixed
        params:
          value: 1.0
  types:
    - name: Generic
      utility_model: Click Fixed 1.0

platform:
  utility_model:
    class_name: null_model

recommender:
  initial: [Generic]
  base_recommenders:
    - name: Generic
      class_name: item_knn
      params:
        max_neighbors: 20
        min_neighbors: 2
        min_similarity: 0.001
        min_user_count: 20
        min_interaction_count: 500
        min_profile_size: 5
        cold_user_fallback: Popular File Generic
        cold_start_fallback: Popular File Generic

  fallback_recommenders:
    - name: Popular File Generic
      class_name: file_based
      params:
        file_name: popular_generic_items.csv

triggers: null
```

## Config Sections At A Glance

| Section | Required | Purpose |
|---|---:|---|
| `simulation` | yes | Run length, slate size, random seed, and experiment name. |
| `data` | yes | Input CSV directory and file names. |
| `output` | yes | Output directory, output file names, debug logging, and optional sampled logs. |
| `summary_logger` | no | Enables or disables console summary logging. |
| `consumer` | yes | Consumer utility, item selection, and recommender choice behavior. |
| `provider` | yes | Provider utility behavior. |
| `platform` | yes | Platform-level utility model. |
| `recommender` | yes | Base recommenders, fallback recommenders, and initially active recommenders. |
| `triggers` | yes | Events that change simulation behavior, or `null`. |
| `trigger_tester` | no | Optional debug checks for trigger behavior. |

## Common Recipes

### Run A Short Experiment

Use small values while developing a config.

```yaml
simulation:
  experiment_name: smoke_test
  num_days: 1
  num_cycles: 1
  slate_size: 5
  seed: 20250702
```

### Write CSV Instead Of Parquet

Set `use_parquet` to `false` when you want easier inspection with standard CSV
tools.

```yaml
output:
  use_parquet: false
```

### Enable User Journey Logging

Use this when you want to inspect sampled user behavior across cycles.

```yaml
output:
  enable_user_journeys: true
  sampled_user_count: 5
  sampled_user_file: user_journeys
```

To log specific users instead of a random sample:

```yaml
output:
  enable_user_journeys: true
  sampled_user_ids: [1, 5, 10]
  sampled_user_file: user_journeys
```

### Enable Item Statistics

Use this when you want per-item output metrics.

```yaml
output:
  enable_item_stats: true
  item_stats_file: item_stats
```

### Add A Consumer Type

Define the models first, then reference them from `consumer.types`.

```yaml
consumer:
  models:
    utility:
      - name: Dot Product
        class_name: dot_product
    item_selection:
      - name: Category Similarity
        class_name: category_similarity_logit
        params:
          threshold: 0.2
          selection_utility: 1.0
  types:
    - name: Niche
      utility_model: Dot Product
      item_selection_model: Category Similarity
      recommender_choice_model:
        class_name: threshold
        params:
          threshold: 0.2
          beta: 2
```

### Add A File-Based Fallback Recommender

Fallback recommenders are used by base recommenders for cold-start or cold-user
cases.

```yaml
recommender:
  base_recommenders:
    - name: Generic
      class_name: item_knn
      params:
        cold_start_fallback: Popular File Generic
        cold_user_fallback: Popular File Generic

  fallback_recommenders:
    - name: Popular File Generic
      class_name: file_based
      params:
        file_name: popular_generic_items.csv
```

### Sample Candidate Items

Use an item sampler to restrict candidate generation for a recommender.

```yaml
recommender:
  base_recommenders:
    - name: Generic
      class_name: item_knn
      params:
        item_sampler:
          class_name: rejection_sampler
          params:
            file_name: popular_generic_items.csv
            sampled_item_count: 2
```

### Add An Initial Burn-In Trigger

Use `initial_burnin` when a recommender should become available only after a
number of cycles.

```yaml
triggers:
  - name: Cycle2Freeze
    class_name: initial_burnin
    params:
      cycle_count: 2
      recommenders: [Niche]
```

`initial_burnin` is forced to non-repeating internally.

### Enable Trigger Tester

Use `trigger_tester` only when debugging trigger behavior.

```yaml
trigger_tester:
  enabled: true
  sample_count: 5
  seed: 20250702
  scenario: initial_burnin_check
```

## Full Configuration Reference

### Top-Level Structure

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

### `simulation`

| Field | Type | Required | Default | Description |
|---|---|---:|---|---|
| `experiment_name` | `str` | yes | - | Human-readable name for the run. |
| `num_days` | `int > 0` | yes | - | Number of simulated days. |
| `num_cycles` | `int > 0` | yes | - | Number of cycles per day. |
| `slate_size` | `int > 0` | yes | - | Number of recommended items shown per slate. |
| `seed` | `int > 0` | yes | - | Random seed for reproducibility. |

### `data`

| Field | Type | Required | Default | Description |
|---|---|---:|---|---|
| `directory` | `str` | yes | - | Directory holding input CSV files. |
| `consumer_file` | `str` | yes | - | Consumer CSV file name. |
| `item_file` | `str` | yes | - | Item CSV file name. |
| `provider_file` | `str` | yes | - | Provider CSV file name. |

### `output`

| Field | Type | Required | Default | Description |
|---|---|---:|---|---|
| `directory` | `str` | yes | - | Directory where output files are written. |
| `consumer_file` | `str` | yes | - | Consumer utility output base name. |
| `provider_file` | `str` | yes | - | Provider utility output base name. |
| `choice_file` | `str` | yes | - | Recommender choice utility output base name. |
| `debug_file` | `str` | yes | - | Debug log base name. |
| `debug_level` | `str` | yes | - | Debug log level, such as `debug`. |
| `use_timestamp` | `bool` | yes | - | Whether output names include timestamps. |
| `use_parquet` | `bool` | yes | - | Whether metric outputs are written as parquet. |
| `cycle_file` | `str` | no | `cycle_metrics` | Cycle metrics output base name. |
| `enable_user_journeys` | `bool` | no | `false` | Enables sampled user journey logging. |
| `sampled_user_ids` | `list[int] \| null` | no | `null` | Specific user IDs to log. |
| `sampled_user_count` | `int \| null` | no | `null` | Number of users to sample for journey logging. |
| `sampled_user_file` | `str \| null` | no | `null` | User journey output base name. |
| `enable_item_stats` | `bool` | no | `false` | Enables item statistics logging. |
| `item_stats_file` | `str \| null` | no | `null` | Item statistics output base name. |

Cycle metric population fields use explicit timing semantics. `users_at_cycle_start`
is the assignment snapshot taken before training and serving, `served_users` counts
distinct consumers who actually requested recommendations during the cycle, and
`users_at_cycle_end` is recorded after end-of-cycle recommender choice.
`new_assignments` and `departures` are the set differences between the end and
start snapshots. Traffic fields such as `rec_requests`, `fallback_used`, and `ctr`
therefore align with `served_users`, not with the next cycle's assignments.
Raw `clicks`, `sampled_items`, and `slate_items_total` counts are included so
summaries can calculate exact request-weighted rates. `platform_coverage_pct`
uses the union of item IDs served by every active recommender during the cycle;
`coverage_pct` remains the coverage of the individual recommender row.

### `summary_logger`

Optional section.

| Field | Type | Required | Default | Description |
|---|---|---:|---|---|
| `enabled` | `bool` | no | `true` | Enables or disables summary logging. |

### `consumer`

| Field | Type | Required | Default | Description |
|---|---|---:|---|---|
| `initial_recommender` | `str` | yes | - | Initial recommender assigned to consumers. |
| `models.utility` | `list[PythonClassConfig]` | yes | - | Available consumer utility models. |
| `models.item_selection` | `list[PythonClassConfig]` | yes | - | Available consumer item-selection models. |
| `types` | `list[ConsumerTypeConfig]` | yes | - | Consumer type definitions. |

#### `consumer.models.utility[*]`

| Field | Type | Required | Default | Description |
|---|---|---:|---|---|
| `name` | `str` | yes | - | Name referenced by `consumer.types[*].utility_model`. |
| `class_name` | `str` | yes | - | Utility model implementation key. |
| `params` | `dict` | no | `{}` | Model-specific parameters. |

#### `consumer.models.item_selection[*]`

| Field | Type | Required | Default | Description |
|---|---|---:|---|---|
| `name` | `str` | yes | - | Name referenced by `consumer.types[*].item_selection_model`. |
| `class_name` | `str` | yes | - | Item-selection model implementation key. |
| `params` | `dict` | yes | - | Model-specific parameters. |

#### `consumer.types[*]`

| Field | Type | Required | Default | Description |
|---|---|---:|---|---|
| `name` | `str` | yes | - | Consumer type name. |
| `utility_model` | `str` | yes | - | References `consumer.models.utility[*].name`. |
| `item_selection_model` | `str` | yes | - | References `consumer.models.item_selection[*].name`. |
| `recommender_choice_model` | `PythonClassConfig` | yes | - | Model controlling recommender switching. |

### `provider`

| Field | Type | Required | Default | Description |
|---|---|---:|---|---|
| `initial_recommender` | `str` | yes | - | Initial recommender assigned to providers. Use `__all__` to include all recommenders. |
| `models.utility` | `list[PythonClassConfig]` | yes | - | Available provider utility models. |
| `types` | `list[ProviderTypeConfig]` | yes | - | Provider type definitions. |

#### `provider.models.utility[*]`

| Field | Type | Required | Default | Description |
|---|---|---:|---|---|
| `name` | `str` | yes | - | Name referenced by `provider.types[*].utility_model`. |
| `class_name` | `str` | yes | - | Utility model implementation key. |
| `params` | `dict` | yes | - | Model-specific parameters. |

#### `provider.types[*]`

| Field | Type | Required | Default | Description |
|---|---|---:|---|---|
| `name` | `str` | yes | - | Provider type name. |
| `utility_model` | `str` | yes | - | References `provider.models.utility[*].name`. |

### `platform`

| Field | Type | Required | Default | Description |
|---|---|---:|---|---|
| `utility_model` | `PythonClassConfig` | yes | - | Platform-level utility model. |

This section is reserved for platform-level utility modeling and may be
experiment-specific.

### `recommender`

| Field | Type | Required | Default | Description |
|---|---|---:|---|---|
| `initial` | `list[str]` | yes | - | Names of base recommenders active at the start of the run. |
| `base_recommenders` | `list[PythonClassConfig]` | yes | - | Primary recommender definitions. |
| `fallback_recommenders` | `list[PythonClassConfig]` | yes | - | Recommenders used as fallbacks by base recommenders. |

Each recommender entry uses `PythonClassConfig`.

| Field | Type | Required | Default | Description |
|---|---|---:|---|---|
| `name` | `str` | yes | - | Recommender name used by references. |
| `class_name` | `str` | yes | - | Recommender implementation key. |
| `params` | `dict` | no | `{}` | Recommender-specific parameters. |

### `triggers`

Nullable list of `PythonClassConfig` entries. Use `triggers: null` when no
triggers are configured.

| Field | Type | Required | Default | Description |
|---|---|---:|---|---|
| `name` | `str` | no | `null` | Trigger name for readability. |
| `class_name` | `str` | yes | - | Trigger implementation key. |
| `params` | `dict` | no | `{}` | Trigger-specific parameters. |

### `trigger_tester`

Optional section.

| Field | Type | Required | Default | Description |
|---|---|---:|---|---|
| `enabled` | `bool` | no | `false` | Enables trigger tester checks. |
| `sample_count` | `int` | no | `5` | Number of samples used in checks. |
| `seed` | `int \| null` | no | `null` | Optional seed for trigger tester sampling. |
| `scenario` | `str \| null` | no | `null` | Optional scenario label. |

## Supported Models And Parameters

### Consumer Utility Models

#### `fixed_utility`

Returns a fixed utility value.

```yaml
- name: Fixed utility 0.5
  class_name: fixed_utility
  params:
    value: 0.5
```

Required params:

| Param | Type | Description |
|---|---|---|
| `value` | number | Fixed utility value. |

#### `list_average`

Backward-compatible alias for list-average utility behavior.

```yaml
- name: List Average
  class_name: list_average
```

#### `cosine_similarity`

Uses cosine similarity as the consumer utility model.

```yaml
- name: Cosine Similarity
  class_name: cosine_similarity
```

#### `dot_product`

Uses dot product as the consumer utility model.

```yaml
- name: Dot Product
  class_name: dot_product
```

### Consumer Item Selection Models

#### `category_similarity_logit`

Selects items using category similarity and logit behavior.

```yaml
- name: Category Similarity
  class_name: category_similarity_logit
  params:
    threshold: 0.2
    selection_utility: 1.0
```

Required params:

| Param | Type | Description |
|---|---|---|
| `threshold` | number | Similarity threshold used by the selection model. |
| `selection_utility` | number | Utility value used during item selection. |

### Recommender Choice Models

#### `fixed`

Always chooses the same recommender.

```yaml
recommender_choice_model:
  class_name: fixed
  params:
    recommender_name: Generic
```

Required params:

| Param | Type | Description |
|---|---|---|
| `recommender_name` | `str` | Recommender name to use. |

#### `threshold`

Switches recommenders based on a threshold.

```yaml
recommender_choice_model:
  class_name: threshold
  params:
    threshold: 0.2
    beta: 2
```

Required params:

| Param | Type | Description |
|---|---|---|
| `threshold` | number | Switching threshold. |
| `beta` | number | Sensitivity parameter. |

#### `ucb`

Uses upper confidence bound behavior for recommender choice.

```yaml
recommender_choice_model:
  class_name: ucb
  params:
    beta: 2
```

Required params:

| Param | Type | Description |
|---|---|---|
| `beta` | number | Exploration sensitivity parameter. |

#### `epsilon_greedy`

Chooses the active recommender whose representation has the highest dot-product
alignment with the consumer preference vector. The current recommender is
included in the comparison. Exact ties keep the current recommender when it is
one of the best options; other ties are resolved with the seeded simulation RNG.

```yaml
recommender_choice_model:
  class_name: epsilon_greedy
  params:
    epsilon: 0.0
    beta: 2
```

Required params:

| Param | Type | Description |
|---|---|---|
| `epsilon` | number in `[0, 1]` | Probability of staying with the current recommender instead of making the greedy preview choice. Use `0.0` for deterministic highest-preview choice. |
| `beta` | number | Smoothing weight for experienced recommender utility. Preview-based decisions do not use this historical value. |

### Provider Utility Models

#### `click_fixed`

Assigns fixed provider utility per click.

```yaml
- name: Click Fixed 1.0
  class_name: click_fixed
  params:
    value: 1.0
```

Required params:

| Param | Type | Description |
|---|---|---|
| `value` | number | Fixed provider utility value. |

### Recommender Models

#### Common Recommender Params

Many recommenders support these optional params:

| Param | Type | Description |
|---|---|---|
| `cold_start_fallback` | `str` | Fallback recommender for cold-start cases. |
| `cold_user_fallback` | `str` | Fallback recommender for cold-user cases. |
| `candidate_multiplier` | `int >= 1` | Multiplier used when generating candidate items. |
| `recency_threshold` | `int >= 0` | Recency cutoff used by supported recommenders. |
| `cooldown_cycles` | `int >= 0` | Number of cycles before a cooled-down item can reappear. |
| `item_sampler` | `dict` | Optional item sampler config. |

Supported `item_sampler.class_name` values:

| `class_name` | Required params | Optional params |
|---|---|---|
| `rejection_sampler` | `file_name` | `sampled_item_count` |

Example:

```yaml
item_sampler:
  class_name: rejection_sampler
  params:
    file_name: popular_generic_items.csv
    sampled_item_count: 2
```

#### `fixed_recommender`

Fixed recommender with no required params.

```yaml
- name: Dummy Fixed
  class_name: fixed_recommender
```

#### `file_based`

Reads recommendations from a file.

```yaml
- name: Popular File Generic
  class_name: file_based
  params:
    file_name: popular_generic_items.csv
```

Required params:

| Param | Type | Description |
|---|---|---|
| `file_name` | `str` | File containing recommendations. |

#### `popular`

Recommends popular items.

```yaml
- name: Popular Generic
  class_name: popular
  params:
    min_user_count: 20
    min_interaction_count: 500
```

Required params:

| Param | Type | Description |
|---|---|---|
| `min_user_count` | `int` | Minimum user count threshold. |
| `min_interaction_count` | `int` | Minimum interaction count threshold. |

#### `item_knn`

Item-nearest-neighbor recommender.

```yaml
- name: Generic
  class_name: item_knn
  params:
    max_neighbors: 20
    min_neighbors: 2
    min_similarity: 0.001
    min_user_count: 20
    min_interaction_count: 500
    min_profile_size: 5
    cold_user_fallback: Popular File Generic
    cold_start_fallback: Popular File Generic
```

Required params:

| Param | Type |
|---|---|
| `max_neighbors` | `int` |
| `min_neighbors` | `int` |
| `min_similarity` | number |
| `min_user_count` | `int` |
| `min_interaction_count` | `int` |
| `min_profile_size` | `int` |

#### `implicit_mf`

Implicit matrix factorization recommender.

Required params:

| Param | Type |
|---|---|
| `embedding_size` | `int` |
| `epochs` | `int` |
| `regularization` | number |
| `positive_weight` | number |
| `min_user_count` | `int` |
| `min_interaction_count` | `int` |
| `min_profile_size` | `int` |

Optional params:

| Param | Type |
|---|---|
| `model_params` | `dict` |
| `als_params` | `dict` |
| `use_gpu` | `bool` |

#### `bpr`

Bayesian personalized ranking recommender.

Optional params:

| Param | Type |
|---|---|
| `min_user_count` | `int` |
| `min_interaction_count` | `int` |
| `min_profile_size` | `int` |
| `model_params` | `dict` |
| `bpr_params` | `dict` |
| `use_gpu` | `bool` |

#### Genre Recommenders

Genre recommenders extend the corresponding base recommender with
`genre_feature`.

| `class_name` | Base params | Additional required params |
|---|---|---|
| `genre_knn` | `item_knn` params | `genre_feature` |
| `genre_implicit_mf` | `implicit_mf` params | `genre_feature` |
| `genre_bpr` | `bpr` params | `genre_feature` |

Example:

```yaml
- name: Niche
  class_name: genre_knn
  params:
    genre_feature: 10
    max_neighbors: 20
    min_neighbors: 2
    min_similarity: 0.001
    min_user_count: 20
    min_interaction_count: 500
    min_profile_size: 5
```

### Triggers

#### `initial_burnin`

Makes specified recommenders unavailable until a cycle count is reached.

```yaml
- name: Cycle2Freeze
  class_name: initial_burnin
  params:
    cycle_count: 2
    recommenders: [Niche]
```

Required params:

| Param | Type | Description |
|---|---|---|
| `cycle_count` | `int` | Number of cycles before the trigger takes effect. |
| `recommenders` | `list[str]` | Recommender names affected by the trigger. |

`repeating` is forced to `False` internally.

#### Other Supported Triggers

| `class_name` | Required params | Notes |
|---|---|---|
| `universal_profile` | none | Uses universal profile behavior. |
| `cold_start` | none | Cold-start trigger behavior. |
| `user_ownership` | none | User-ownership trigger behavior. |
| `save_switch` | none | Testing/debug trigger. |

## Name References

The most common config mistake is a misspelled reference. Use names
consistently and keep them stable.

Example:

```yaml
consumer:
  models:
    utility:
      - name: Dot Product
        class_name: dot_product

  types:
    - name: Generic
      utility_model: Dot Product
```

`utility_model: Dot Product` must match `name: Dot Product` exactly.

The same rule applies to:

- `consumer.types[*].item_selection_model`
- `provider.types[*].utility_model`
- `recommender.initial`
- `cold_start_fallback`
- `cold_user_fallback`
- trigger `params.recommenders`

## Validation And Troubleshooting

### Config Does Not Load

Check that all required sections exist:

```yaml
simulation:
data:
output:
consumer:
provider:
platform:
recommender:
triggers:
```

`summary_logger` and `trigger_tester` are optional. `triggers` must be present,
but can be `null`.

### A Referenced Model Is Not Found

Verify exact spelling and capitalization for referenced names. YAML strings
such as `Generic`, `generic`, and `"Generic "` are different values.

### User Journey Logging Fails

If `enable_user_journeys` is `true`, provide `sampled_user_file`.

```yaml
output:
  enable_user_journeys: true
  sampled_user_count: 5
  sampled_user_file: user_journeys
```

### No Fallback Recommendations Appear

Check that fallback names point to entries in `fallback_recommenders`, not just
`base_recommenders`.

```yaml
recommender:
  base_recommenders:
    - name: Generic
      params:
        cold_start_fallback: Popular File Generic

  fallback_recommenders:
    - name: Popular File Generic
      class_name: file_based
```

### Old Config Uses `recommender.definitions`

Current configs should use:

```yaml
recommender:
  initial: [Generic]
  base_recommenders: []
  fallback_recommenders: []
```

Older examples may use `recommender.definitions`; update those configs before
running with the current schema.

## Quickstart Example

Run the included config:

```bash
python quickstart/bin/run_smores.py --config_file quickstart/ml1m.yaml
```

This writes outputs to `quickstart/output/ml1m-mini/`.
