from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class API:
    HF: Optional[str]

@dataclass
class Param:
    iteration: int
    sampling: bool
    temp: float
    top_k: int
    top_p: float

@dataclass
class Select:
    dataset: str
    task: str
    model: str
    eval_model: str
    method: str
    start_prompt: str
    iterate_prompt: str

@dataclass
class MainConfig:
    api: API
    param: Param
    model: Dict[str, str]
    select: Select
