# Import the FastAPI library
from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
import json
from typing import List
from models.response import HistoryItem
from collections import OrderedDict
from pprint import pprint
import exogeneous_model as exog_fun
import numpy as np
import pandas as pd
from beautifier import PrettyJSONResponse
from datetime import datetime
from temp_wrapper import data
def read_markdown_file(path):
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()

tags = ['traders', 'sessions', 'asks', 'bids', 'transactions']
DESC_PATH = './descriptions'
def get_tag_meta(tag):
    return dict(
        name=tag.capitalize(),
        description=read_markdown_file(f'{DESC_PATH}/{tag}.md')
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


# that we need for local testing of a wrapper
if __name__ == "__main__":
    pass


@app.get(
    "/history",
    response_model=List[HistoryItem],
    response_class=PrettyJSONResponse,
    responses={200: {"model": List[HistoryItem]}},
    summary="get full history for the trading session",
    description="it returns a dataset with full trading history",
)
async def get_history(request: Request):
    history = [{"a": 1}]
    return history


@app.get("/session/create/{session_uuid}", tags=['Sessions', 'Users'],
           summary="Create a new trading session",
    description="Create new session, creates data for new noise traders",)
async def create_session(session_uuid: str):
    pass


@app.get("/session/{session_uuid}/exists")
async def check_session(session_uuid: str):
    pass


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
