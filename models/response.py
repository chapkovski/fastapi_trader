from pydantic import BaseModel, Field
from typing import List
from datetime import datetime
from .payload import SessionSettings
class HistoryItem(BaseModel):
    trader_id: int = Field(..., example=0, description='Trader ID (0 - means noise trader)')
    timestamp: float=Field(..., example=7.012312)
    price: float = Field(..., example=35.4)
    direction: int=Field(..., example=-1)
     
class TransactionHistoryResponse(BaseModel):
    session_id: str
    is_active: bool
    num_users: int
    num_noise_traders: int
    num_algo_traders: int
    transactions: List[HistoryItem]

class Session(BaseModel):
    session_id: str = Field(..., example="3fa85f64-5717-4562-b3fc-2c963f66afa6")
    is_active: bool = Field(..., example=True)
    when_created: datetime = Field(..., example="2023-07-29T10:20:30.400Z")
    params: SessionSettings = Field(..., )