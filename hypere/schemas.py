from ranx import Qrels
from typing import List, Dict
from pydantic import BaseModel


class ValidationSet(BaseModel):
    queries: List[str]
    passages: List[str]
    query2id: Dict[str, int]
    passage2id: Dict[str, int]
    qrels: Qrels

    class Config:
        arbitrary_types_allowed = True