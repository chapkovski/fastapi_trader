from mongoengine import connect
from os import environ
from log import logger
mongo_uri = environ.get('MONGO_URL')
if mongo_uri:
    db=connect(host=mongo_uri)
else:
    logger.warning('No MONGO URI FOUND IN ENV. Assume we run locally...')
    DB_NAME = environ.get("MONGO_DB_NAME", "mydb")
    db = connect(DB_NAME)
print(f"Successfully connected to MOngo: {db}")
from .models import Trade


def register_trade(trade_info):
    """
    We don't check anything here (it is implicitly done by mongoengine doc structure),
    we just through the dict there and save
    """
    trade = Trade(**trade_info)
    trade.save()
