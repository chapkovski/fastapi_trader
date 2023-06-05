# Import the FastAPI library
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import json
from collections import OrderedDict
from pprint import pprint
import exogeneous_model as exog_fun
import numpy as np
import pandas as pd
from beautifier import PrettyJSONResponse
from datetime import datetime

# Create an instance of the FastAPI class
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


settings = {
    # 'hours': 60,  # minutes
    "T_day": 1,  # 60 * 6.5,  # mins in a trading day; trading day length in munutes
    # 60/3*0.5,  # 1/min   # 1 = 1min freq,  2  = a trading occurs every 30 sec, 0.5 = traders every 2 min
    "freq_agg": 1,  # 60/2*0.5,  # number of aggresive orders per minute
    "freq_pass": 1,  # 60/2,  # 60/3, number of passive orders per minute
    "N_depth": 5,
    "sigma": 100.0,  # price are multiplied by 100, 1/1e2 is one basis point
    "previous_mid_price": 10000,
    "seed": 2023,
}

# Define a route for the root URL ("/") using the get() decorator


@app.get("/", response_class=PrettyJSONResponse)
async def read_root(request: Request, position: int = 0):
    max_position = len(data) - 1
    day_over = position >= max_position

    position = min(position, max_position)
    res = data[position].copy()
    last_history_item=res['history'][-1]
    last_history_item['timestamp']=datetime.now() #int(datetime.now().timestamp() * 1000)
    # pprint(f'{last_history_item=}')
    res["day_over"] = day_over
    res['last_history_item']=last_history_item
    return res


def order_book_wrapper(settings):
    int_lambda = [
        settings["freq_agg"],
        settings["freq_agg"],
        settings["freq_pass"],
        settings["freq_pass"],
    ]

    # create the exogenous order book
    np.random.seed(seed=settings["seed"])

    settings["T_day"] = 10
    exogenous_events = exog_fun.create_event_times(
        J_traders=1, int_lambda=int_lambda, day=settings["T_day"]
    )

    zeros_col = np.zeros((exogenous_events.shape[0], 1))
    event_book = np.column_stack((exogenous_events, zeros_col))
    # the last column is the idenity of the trader  1 is Algo

    # at each update of the book the algortihm check the state of the book
    arg_input = {
        "inventory": 200,
        "granularity_frequency": 1 / 10000,
        "T_day": settings["T_day"],
    }

    out = exog_fun.create_limit_order_book(
        event_book,
        settings["N_depth"],
        settings["sigma"],
        settings["previous_mid_price"],
        algo_trading=False,
        arg_input=arg_input,
    )

    res = dict(settings=settings, pass_trader=dict(), agg_trader=dict(), data=list())
    # our goal here is to glue together book_state, trade_history and trader_id
    # the

    def retrieve_book_state(_type):
        res = []

        for j, i in enumerate(out["book_state"]):
            arr = i[0].get(_type)
            arr_sub = arr[:, :2]
            df = pd.DataFrame(arr_sub, columns=["x", "y"])
            res.append(df.to_dict(orient="records"))

        return res

    def generate_chart_prices(data, position):
        types = ["bid", "ask"]
        res=[]
        for i in data[:position]:
            temp_res = {}
            for _type in types:
                arr = i[0].get(_type)
                arr_sub = arr[:, :3]
                df = pd.DataFrame(arr_sub, columns=["price", "action", "trader"])
                temp_res[_type] = df.price.mean()

            temp_res["median"] = (temp_res["bid"] + temp_res["ask"]) / 2
            res.append(temp_res)
        max_len=len(data)            
        rest_len=len(data)-len(res)
        rest=[None]*rest_len
        ask_data=[i.get('ask') for i in res]+rest
        bid_data=[i.get('bid') for i in res]+rest
        median_data=[i.get('median') for i in res]+rest
        ask_series=dict(name='Ask', data=ask_data)
        bid_series=dict(name='Bid', data=bid_data)
        median_series=dict(name='Median', data=median_data)
        res=[ask_series, bid_series, median_series]
        max_len=len(data)
        
        return res

    bid_list = retrieve_book_state("bid")
    ask_list = retrieve_book_state("ask")
    full_data_for_chart=[]
    for i,j in enumerate(out["book_state"]):
        data_for_chart=generate_chart_prices(out["book_state"], position=i)
        full_data_for_chart.append(data_for_chart)
    traders_identity_df = pd.DataFrame(out["traders_identity"], columns=["trader_id"])

    trade_history_df = pd.DataFrame(
        out["trade_history"],
        columns=["timestamp", "direction", "order_type", "price", "agression"],
    )

    df = traders_identity_df.merge(trade_history_df, left_index=True, right_index=True)
    trade_history = df.to_dict(orient="records")
    timestamps = df["timestamp"].tolist()
    increment_trade_history = [
        trade_history[: i + 1] for i in range(len(trade_history))
    ]

    full_data_zip = zip(timestamps, increment_trade_history, bid_list, ask_list, full_data_for_chart)
    full_data = [
        dict(timestamp=timestamp, tot_length=len(timestamps), data_for_chart=data_for_chart,
              history=history, bid=bid, ask=ask)
        for timestamp, history, bid, ask, data_for_chart in full_data_zip
    ]
    

    return full_data


data = order_book_wrapper(settings)
# that we need for local testing of a wrapper
if __name__ == "__main__":
    pass
    pprint(data[0])
