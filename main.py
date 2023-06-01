# Import the FastAPI library
from fastapi import FastAPI, Request
import json
from collections import OrderedDict
from pprint import pprint
import exogeneous_model as exog_fun
import numpy as np
import pandas as pd
from beautifier import PrettyJSONResponse

# Create an instance of the FastAPI class
app = FastAPI()


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
async def read_root(
    request: Request,
    position: int=0
):
    max_position=len(data)-1
    day_over=position>=max_position
        
    position=min(position, max_position)
    res = data[position].copy()
    res['day_over']=day_over
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

    res = dict(
        settings=settings, pass_trader=dict(), agg_trader=dict(), data=list()
    )
    # our goal here is to glue together book_state, trade_history and trader_id
    # the 

    def retrieve_book_state(_type):
        res = []

        for j, i in enumerate(out["book_state"]):
            arr = i[0].get(_type)
            arr_sub = arr[:, :3]
            df = pd.DataFrame(arr_sub, columns=["timestamp", "action", "trader"])
            res.append(df.to_dict(orient="records"))

        return res

    bid_list = retrieve_book_state("bid")
    ask_list = retrieve_book_state("ask")
    traders_identity_df = pd.DataFrame(
        out["traders_identity"], columns=["trader_id"]
    )

    trade_history_df = pd.DataFrame(
        out["trade_history"],
        columns=["timestamp", "direction", "order_type", "price", "agression"],
    )
    
    df = traders_identity_df.merge(
        trade_history_df, left_index=True, right_index=True
    )
    trade_history = df.to_dict(orient="records")
    increment_trade_history = [trade_history[:i+1] for i in range(len(trade_history))]

    full_data_zip=zip(increment_trade_history, bid_list, ask_list)
    full_data=[dict(
        history=history, 
        bid=bid,
        ask=ask
    ) for history, bid, ask in full_data_zip]
    return full_data
      

data=order_book_wrapper(settings)
# that we need for local testing of a wrapper
if __name__ == "__main__":
    print(data[0])
