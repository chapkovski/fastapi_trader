from pydantic import BaseModel, Field, root_validator
from enum import Enum
from uuid import UUID
from typing import Optional



class Item(BaseModel):
    name: str
    description: str = None
    price: float
    is_offer: bool = None


class Trade(BaseModel):
    agression: int
    direction: int
    order_type: int
    price: float
    timestamp: float
    trader_id: int


class OrderType(Enum):
    market = 0
    limit = 1


class EventType(Enum):
    aggressive_buy = 0
    aggressive_sell = 1
    passive_buy = 2
    passive_buy_improv = 2.5  # buy with price improvements
    passive_sell = 3
    passive_sell_improv = 3.5  # sell with price improvements


class NewOrder(BaseModel):
    event_time: float = Field(..., example=1234.567)
    event_type: EventType = Field(..., example=EventType.aggressive_buy)
    flag_price_improvement: int = Field(..., example=0)
    previous_mid_price: float = Field(..., example=50.0)
    id_trader_in_session: Optional[int] = Field(None, example=1)
    uuid_trader: Optional[UUID] = Field(None, example=UUID('12345678123456781234567812345678'))
    uuid_session: UUID = Field(..., example=UUID("12345678123456781234567812345678"))

    @root_validator(pre=True)
    def check_trader_id_or_uuid(cls, values):
        id_trader_in_session = values.get("id_trader_in_session")
        uuid_trader = values.get("uuid_trader")
        if id_trader_in_session is None and uuid_trader is None:
            raise ValueError(
                'At least one of "id_trader_in_session" or "uuid_trader" must be provided'
            )
        return values


class SessionSettings(BaseModel):
    T_day: int = Field(..., example=1, description="Trading day length in minutes.")
    freq_agg: int = Field(
        ..., example=1, description="Number of aggressive orders per minute."
    )
    freq_pass: int = Field(
        ..., example=1, description="Number of passive orders per minute."
    )
    N_depth: int = Field(..., example=5, description="Depth of the order book.")
    sigma: float = Field(
        ...,
        example=100.0,
        description="Standard deviation for the normal distribution of price.",
    )
    previous_mid_price: int = Field(
        ...,
        example=10000,
        description="The mid-price from the previous trading session.",
    )
    seed: int = Field(
        ..., example=2023, description="Seed for the random number generator."
    )


class TraderInSession(BaseModel):
    id_in_session: int = Field(
        ...,
        example=1,
        description="Id in session (from 0 - noise trader, up to actual number of traders in this session)",
    )
    session_uuid: UUID = Field(
        ...,
        description="Unique UUID to find the session data in the DB",
        example="3fa85f64-5717-4562-b3fc-2c963f66afa6",
    )


class TraderNoSession(BaseModel):
    trader_uuid: UUID = Field(
        ...,
        description="Unique UUID to find the trader data in the DB",
        example="3fa85f64-5717-4562-b3fc-2c963f66afa6",
    )

class CancelOrder(BaseModel):
    id_trader_in_session: int= Field(..., example=1)
    trader_uuid: UUID = Field(..., example=UUID('12345678123456781234567812345678'))
    id_order: int = Field(..., example=1)
    order_uuid: UUID = Field(..., example=UUID('12345678123456781234567812345678'))