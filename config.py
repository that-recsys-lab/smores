from pydantic import BaseModel

class SimulationConfig(BaseModel):
    experiment_name: str
    dataset_directory: str
    num_days: int
    num_cycles: int
    slate_size: int
    recommenders: list[dict] 