# Import the FastAPI library
from fastapi import FastAPI, Request, Query, Body, Path
from fastapi.middleware.cors import CORSMiddleware
import json
from uuid import UUID
from typing import List, Optional
from models.response import (
    HistoryItem,
    TransactionHistoryResponse,
    Session,
)
from collections import OrderedDict
from pprint import pprint
import exogeneous_model as exog_fun
import numpy as np
import pandas as pd
from beautifier import PrettyJSONResponse
from datetime import datetime
from temp_wrapper import data
import random
from models.payload import (
    OrderType,
    EventType,
    SessionSettings,
    TraderInSession,
    TraderNoSession,
    NewOrder,
    CancelOrder
)


def read_markdown_file(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


tags = [
    "traders",
    "sessions",
]
DESC_PATH = "./descriptions"


def get_tag_meta(tag):
    return dict(
        name=tag.capitalize(), description=read_markdown_file(f"{DESC_PATH}/{tag}.md")
    )


tags_metadata = [get_tag_meta(i) for i in tags]


# Create an instance of the FastAPI class
app = FastAPI(
    title="Trading platform API",
    description=read_markdown_file(f"{DESC_PATH}/general.md"),
    version="1.0.0",
    openapi_tags=tags_metadata,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get(
    "/",
    response_class=PrettyJSONResponse,
    summary="get the overall data for platform",
    description="For now it serves as a single entry point for all front clients.",
)
async def read_root(
    request: Request,
    position: int = Query(
        0, description="At which point out of N available we will return the data"
    ),
):
    max_position = len(data) - 1
    day_over = position >= max_position

    position = min(position, max_position)
    res = data[position].copy()
    last_history_item = res["history"][-1]
    last_history_item[
        "timestamp"
    ] = datetime.now()  # int(datetime.now().timestamp() * 1000)
    # pprint(f'{last_history_item=}')
    res["day_over"] = day_over
    res["last_history_item"] = last_history_item

    return res


@app.get(
    "/session/{session_uuid}/transactions",
    tags=["Sessions", "Transactions"],
    response_model=TransactionHistoryResponse,
    response_class=PrettyJSONResponse,
    responses={200: {"model": TransactionHistoryResponse}},
    summary="Get  history of transactions for the trading session",
    description="Returns a list with full trading history of all completed transactions",
)
async def get_history(
    request: Request,
    session_uuid: str = Path(..., example="550e8400-e29b-41d4-a716-446655440000"),
):
    res = []
    for i in range(100):
        res.append(
            HistoryItem(
                trader_id=1,
                timestamp=random.uniform(0, 30),
                price=random.uniform(10000, 20000),
                direction=random.choice((-1, 1)),
            )
        )
    transaction_history = sorted(res, key=lambda x: x.timestamp)

    resp = TransactionHistoryResponse(
        session_id=session_uuid,
        is_active=random.choice(
            (
                False,
                True,
            )
        ),
        num_users=10,
        num_noise_traders=5,
        num_algo_traders=4,
        transactions=transaction_history,
    )
    return resp


@app.get(
    "/session/{session_uuid}/exists",
    tags=[
        "Sessions",
    ],
    response_model=bool,
    response_class=PrettyJSONResponse,
    responses={200: {"model": bool}},
    summary="Checks if session exists",
    description="Returns a boolean defining whether the session with this id already exists in db",
)
async def check_session(
    *,
    session_uuid: str = Path(..., example="550e8400-e29b-41d4-a716-446655440000"),
):
    return random.choice([True, False])


@app.get("/session/{session_uuid}")
async def session_data(session_uuid: str):
    pass


@app.get(
    "/session/get_or_create/{session_uuid}",
    summary="Get or create new trading session",
    description="For a given UUID of a trading session it either creates if it is not found or returns full data if the "
    "session with a give UUID already exists",
)
async def get_or_create(session_uuid: str):
    pass


@app.get(
    "/params",
    tags=["Parameters"],
    response_class=PrettyJSONResponse,
    summary="Get possible parameter values",
    description="This endpoint provides possible values for OrderType and EventType parameters with their descriptions.",
)
async def get_params():
    return {
        "OrderType": {
            e.name: {"value": e.value, "description": "Your description here"}
            for e in OrderType
        },
        "EventType": {
            e.name: {"value": e.value, "description": "Your description here"}
            for e in EventType
        },
    }


@app.post(
    "/session/create",
    response_model=Session,
    response_class=PrettyJSONResponse,
    responses={200: {"model": Session}},
    tags=["Sessions"],
    summary="Create a new session",
    description="This endpoint creates a new trading session with the provided settings. <br>"
    "<i>TODO: add an authorization header here.",
)
async def create_session(settings: SessionSettings):
    # Here you can use the settings to actually create a session.
    # Replace the pass statement with your own code.
    pass


@app.get(
    "/trader/{uuid}",
    response_model=TraderInSession,
    tags=["Traders"],
    summary="Get Trader by UUID",
    description="This endpoint returns the Trader's ID in the session and the session UUID it belongs to.",
)
async def get_trader(uuid: UUID):
    # Functionality to get the trader by UUID
    pass


@app.get(
    "/session/{session_uuid}/trader/{id_in_session}",
    response_model=TraderNoSession  ,
    tags=["Traders", "Sessions"],
    summary="Get Trader by Session ID and Trader ID",
    description="This endpoint returns the Trader UUID that corresponds to the given session UUID and Trader ID.",
)
async def get_session_trader(session_uuid: UUID, id_in_session: int):
    # Functionality to get the trader UUID by session ID and trader ID
    pass


@app.post("/orders/new", response_model=NewOrder, tags=["Orders"],
          summary="Create a new order",
          description="This endpoint creates a new order with the given parameters. "
          "<br> Either <code>id_trader_in_session</code> or <code>trader_uuid</code> is required (both are also ok)")
async def create_order(order: NewOrder):
    # Functionality to create a new order
    pass


@app.post("/orders/cancel", response_model=CancelOrder, tags=["Orders"],
            summary="Cancel an order",
            description="This endpoint cancels an order with the given parameters. "
            "<br> Either <code>id_trader_in_session</code> or <code>trader_uuid</code> is required (both are also ok)")
async def cancel_order(order: CancelOrder):
    # Functionality to cancel an order
    pass
