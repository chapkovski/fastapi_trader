from pydantic import BaseModel, Field


class HistoryItem(BaseModel):
    name: str = Field(..., example="Foo")
    description: str = Field(None, example="A very nice Item")
    price: float = Field(..., example=35.4)
     
