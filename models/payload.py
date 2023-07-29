from pydantic import BaseModel, Field


class Item(BaseModel):
    name: str
    description: str = None
    price: float
    is_offer: bool = None


class Trade(BaseModel):
    agression:int
    direction:int
    order_type:int
    price:float
    timestamp:float
    trader_id:int

