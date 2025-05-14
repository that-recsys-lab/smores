from pydantic import BaseModel, create_model, Field, PositiveInt, DirectoryPath, \
    FilePath, ConfigDict
from typing import Optional

class SimulationConfig(BaseModel):
    experiment_name: str
    num_days: PositiveInt
    num_cycles: PositiveInt
    slate_size: PositiveInt
    seed: PositiveInt

class DataConfig(BaseModel):
    directory: DirectoryPath
    consumer_file: str
    item_file: str
    provider_file: str

class PythonClassConfig(BaseModel):
    name: Optional[str] = None
    class_name: str
    model_config = ConfigDict(extra='allow')

class ConsumerConfig(BaseModel):
    recommender_assignment: PythonClassConfig
    utility_model: PythonClassConfig
    selection_model: PythonClassConfig
    choice_model: PythonClassConfig

class ProviderConfig(BaseModel):
    utility_model: PythonClassConfig

class PlatformConfig(BaseModel):
    utility_model: PythonClassConfig

class SmoresConfig(BaseModel):
    simulation: SimulationConfig
    data: DataConfig
    consumer: ConsumerConfig
    provider: ProviderConfig
    platform: PlatformConfig
    recommenders: list[PythonClassConfig]
    triggers: list[PythonClassConfig]



