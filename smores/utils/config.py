from pydantic import BaseModel, PositiveInt, ConfigDict
from typing import Optional, Any

class SimulationConfig(BaseModel):
    experiment_name: str
    num_days: PositiveInt
    num_cycles: PositiveInt
    slate_size: PositiveInt
    seed: PositiveInt

class DataConfig(BaseModel):
    directory: str
    consumer_file: str
    item_file: str
    provider_file: str

class LoggerConfig(BaseModel):
    directory: str
    consumer_file: str
    provider_file: str
    choice_file: str
    debug_file: str
    debug_level: str
    use_timestamp: bool
    use_parquet: bool
    cycle_file: Optional[str] = "cycle_metrics"
    enable_user_journeys: bool = False
    sampled_user_ids: Optional[list[int]] = None
    sampled_user_count: Optional[int] = None
    sampled_user_file: Optional[str] = None
    enable_item_stats: bool = False
    item_stats_file: Optional[str] = None


class SummaryLoggerConfig(BaseModel):
    enabled: bool = True

class TriggerTesterConfig(BaseModel):
    enabled: bool = False
    sample_count: int = 5
    seed: Optional[int] = None
    scenario: Optional[str] = None

class PythonClassConfig(BaseModel):
    name: Optional[str] = None
    class_name: str
    params: Optional[dict[str, Any]] = {}

class ConsumerModelsConfig(BaseModel):
    utility: list[PythonClassConfig]
    item_selection: list[PythonClassConfig]

class ConsumerTypeConfig(BaseModel):
    name: str
    utility_model: str
    item_selection_model: str
    recommender_choice_model: PythonClassConfig

class ConsumerConfig(BaseModel):
    initial_recommender: str
    models: ConsumerModelsConfig
    types: list[ConsumerTypeConfig]

class ProviderModelsConfig(BaseModel):
    utility: list[PythonClassConfig]

class ProviderTypeConfig(BaseModel):
    name: str
    utility_model: str

class ProviderConfig(BaseModel):
    initial_recommender: str
    models: ProviderModelsConfig
    types: list[ProviderTypeConfig]

class PlatformConfig(BaseModel):
    utility_model: PythonClassConfig

class RecommenderConfig(BaseModel):
    initial: list[str]
    base_recommenders: list[PythonClassConfig]
    fallback_recommenders: list[PythonClassConfig]

class SmoresConfig(BaseModel):
    simulation: SimulationConfig
    data: DataConfig
    output: LoggerConfig
    summary_logger: SummaryLoggerConfig = SummaryLoggerConfig()
    consumer: ConsumerConfig
    provider: ProviderConfig
    platform: PlatformConfig
    recommender: RecommenderConfig
    triggers: Optional[list[PythonClassConfig]]
    trigger_tester: Optional[TriggerTesterConfig] = None
