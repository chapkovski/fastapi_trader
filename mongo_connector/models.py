from mongoengine import Document, FloatField, IntField


class Trade(Document):
    agression=IntField()
    direction= IntField()
    order_type = FloatField()
    price=FloatField()
    timestamp=FloatField()
    trader_id=IntField()

    meta = {"ordering": ["-event_time"]}  # descending order
