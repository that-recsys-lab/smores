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

class PythonClassConfig(BaseModel):
    name: Optional[str] = None
    class_name: str
    params: dict[str, Any] = {}

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
    definitions: list[PythonClassConfig]

class SmoresConfig(BaseModel):
    simulation: SimulationConfig
    data: DataConfig
    consumer: ConsumerConfig
    provider: ProviderConfig
    platform: PlatformConfig
    recommender: RecommenderConfig
    triggers: list[PythonClassConfig]

