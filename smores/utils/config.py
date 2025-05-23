from pydantic import BaseModel, PositiveInt, ConfigDict
from typing import Optional

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
    model_config = ConfigDict(extra='allow')


class ConsumerConfig(BaseModel):
    recommender_assignment: PythonClassConfig
    utility_model: PythonClassConfig
    item_selection_model: PythonClassConfig
    recommender_choice_model: PythonClassConfig


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

'''
class ConfigUtils():
    @staticmethod
    def config_walk(config: dict, keys, on_missing='error'):
        ic(config, keys, on_missing)
        if len(keys) == 0:
            ic('Returning', config)
            return config
        else:
            next_key = keys[0]
            if next_key in config:
                next_config = config[next_key]
                result = ConfigUtils.config_walk(next_config, keys[1:], on_missing=on_missing)
                return result
            else:
                if on_missing == 'none':
                    return None
                else:
                    raise ConfigMissingKeyError(config, next_key)


class ConfigMissingKeyError(Exception):
    def __init__(self, config, key):
        self.message = f'Config collection {config} does not contain key {key}.'
        super().__init__(self.message)
'''
