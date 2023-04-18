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
    'T_day': 60 * 6.5,  # mins in a trading day; trading day length in munutes
    # 60/3*0.5,  # 1/min   # 1 = 1min freq,  2  = a trading occurs every 30 sec, 0.5 = traders every 2 min
    'freq_agg': 60/2*0.5,  # number of aggresive orders per minute
    'freq_pass': 60/2,  # 60/3, number of passive orders per minute
    'N_depth': 5,
    'sigma': 100.0,  # price are multiplied by 100, 1/1e2 is one basis point
    'previous_mid_price': 10000,
    'seed': 2023,
}

# Define a route for the root URL ("/") using the get() decorator


@app.get("/", response_class=PrettyJSONResponse)
async def read_root(request: Request,
                    T_day: int = 60*6.5, freq_agg: int = 60/2*0.5,
                    freq_pass: int = 60/2,
                    N_depth: int = 5,
                    sigma: int = 100,
                    previous_mid_price: int = 10000,
                    seed: int = 2023):

    updated_params = dict(T_day=T_day,
                          freq_agg=freq_agg,
                          freq_pass=freq_pass,
                          N_depth=N_depth,
                          sigma=sigma,
                          previous_mid_price=previous_mid_price,
                          seed=seed
                          )

    new_settings = settings.copy()
    new_settings.update(updated_params)
    res = order_book_wrapper(new_settings)
    return res


def order_book_wrapper(settings):
    int_lambda = [settings['freq_agg'], settings['freq_agg'],
                  settings['freq_pass'], settings['freq_pass']]

    # create the exogenous order book
    np.random.seed(seed=settings['seed'])
    exogenous_events = exog_fun.create_event_times(J_traders=1,
                                                   int_lambda=int_lambda,
                                                   day=settings['T_day'])

    zeros_col = np.zeros((exogenous_events.shape[0], 1))
    event_book = np.column_stack((exogenous_events, zeros_col))
    # the last column is the idenity of the trader  1 is Algo

    # at each update of the book the algortihm check the state of the book
    arg_input = {'inventory': 200,
                 'granularity_frequency': 1/10000,
                 'T_day': settings['T_day']}

    out = exog_fun.create_limit_order_book(event_book, settings['N_depth'], settings['sigma'],
                                           settings['previous_mid_price'], algo_trading=False, arg_input=arg_input)

    res = dict(settings=settings, pass_trader=dict(), agg_trader=dict(), data=list())
    book_state_df = pd.DataFrame(out['book_state'])

    data = out['pass_trader'].copy()

    data['cash_inventory_pass'] = data['cash_inventory_pass'].tolist()
    data['inventory_pass'] = data['inventory_pass'].tolist()

    res['pass_trader'] = data
    data = out['agg_trader'].copy()
    data['cash_inventory_agg'] = data['cash_inventory_agg'].tolist()
    data['inventory_agg'] = data['inventory_agg'].tolist()
    res['agg_trader'] = data

    traders_identity_df = pd.DataFrame(
        out['traders_identity'], columns=['trader_id'])

    trade_history_df = pd.DataFrame(out['trade_history'], columns=[
                                    'timestamp', 'direction', 'order_type', 'price', 'agression'])
    df = traders_identity_df.merge(
        trade_history_df, left_index=True, right_index=True)
    data = df.to_dict(orient='index')
    res['data'] = data
    return res


# that we need for local testing of a wrapper
if __name__ == '__main__':
    order_book_wrapper(settings)
