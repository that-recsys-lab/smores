from pydantic import BaseModel, PositiveInt, ConfigDict
from typing import Optional, Dict, Any

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
    params: Optional[Dict[str, Any]] = None


class ConsumerConfig(BaseModel):
    recommender_assignment: PythonClassConfig
    utility_model: PythonClassConfig
    item_selection_model: PythonClassConfig
    recommender_choice_model: PythonClassConfig


class ProviderConfig(BaseModel):
    utility_model: PythonClassConfig


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


