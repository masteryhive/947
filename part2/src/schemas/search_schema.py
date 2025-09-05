from typing import Union, List
from pydantic import BaseModel

class MatchAnyOrInterval(BaseModel):
    any: List[Union[int, str]] = None
    eq: Union[int, str] = None
    gt: Union[int, str] = None
    gte: Union[int, str] = None
    lt: Union[int, str] = None
    lte: Union[int, str] = None

    class Config:
        arbitrary_types_allowed = True