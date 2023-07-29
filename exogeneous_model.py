import pandas as pd
import numpy as np
from scipy.stats import stats
import sys
from pprint import pprint
import scipy as scipy
from scipy import stats
from update_book_nb import update_book_nb
import scipy.stats as st

import random
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import norm
from datetime import datetime
import statsmodels.api as sm
import copy


from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)




def create_frame(book, frame_num, N_depth, MAX=95, MIN=105):
    return
    # Extract the bid and ask prices and sizes from the book
    bids = [bid[0] for bid in book[frame_num][0]['bid']]
    bid_sizes = [bid[1] for bid in book[frame_num][0]['bid']]
    asks = [ask[0] for ask in book[frame_num][0]['ask']]
    ask_sizes = [ask[1] for ask in book[frame_num][0]['ask']]

    # Create the plot
    fig, ax = plt.subplots(figsize=(30, 5))
    y_formatter = ScalarFormatter(useOffset=False)
    ax.yaxis.set_major_formatter(y_formatter)
    ax.bar(bids[:10], bid_sizes[:10], width=20, color='b', alpha=1)
    ax.bar(asks[:10], ask_sizes[:10], width=20, color='r', alpha=1)

    xticks = np.concatenate([bids[:10], asks[:10]])
    xtick_labels = [str(int(p)) for p in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_ylim(0, N_depth)
    ax.set_xlim(MIN, MAX)
    ax.grid(True)

    ax.set_xlabel('Price')
    ax.set_ylabel('Volume')
    ax.set_title(f'Limit Order Book (Frame {frame_num})')

    # Save the plot as an image and return it as a PIL Image object
    # fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    # image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return Image.fromarray(image)


def create_event_times(J_traders=1, int_lambda=None, day=390):
    # OK!
    # create a list of events for each trader
    # output is a matrix with 2 colm
    # the first row indicates the event time and the second col the event type
    #  day [min] *  freq_u[1/min]
    # event 0  aggressive buy
    #       1  aggressive sell
    #       2  passive buy
    #       3  passive sell

    if int_lambda is None:
        int_lambda = [1, 1, 4, 4]
    events = [1, 2, 3, 4]

    # generate orders events
    for ii, d in enumerate(int_lambda):
        aux_events = arrival_times_trader(Lambda=d,
                                          Number_events_trader=int(d * day))
        i_loc = np.where(aux_events < day)[0]  # cut events out of the day
        aux_events = aux_events.iloc[i_loc]
        aux = np.zeros(shape=(aux_events.shape[0], 2))

        aux[:, 0] = aux_events.values.flatten()
        aux[:, 1] = ii
        events[ii] = aux[1:, ]

    # merged and sort

    events_merged = np.concatenate(events)
    events_merged = events_merged[np.argsort(events_merged[:, 0])]

    # con = np.concatenate((events[0][:, 0], events[1][:, 0]))
    # con1 = np.concatenate((events[0][:, 1], events[1][:, 1]))
    #
    # ix_sort = np.argsort(con)
    # events_merged = np.zeros((con.shape[0], 2))
    # events_merged[:, 0] = con[ix_sort]
    # events_merged[:, 1] = con1[ix_sort]
    #
    # for ii in range(2, 4):
    #     con = np.concatenate((events_merged[:, 0], events[ii][:, 0]))
    #     con1 = np.concatenate((events_merged[:, 1], events[ii][:, 1]))
    #     ix_sort = np.argsort(con)
    #     events_merged_aug = np.zeros((con.shape[0], 2))
    #     events_merged_aug[:, 0] = con[ix_sort]
    #     events_merged_aug[:, 1] = con1[ix_sort]
    #     events_merged = events_merged_aug

    return events_merged


def check_if_agg_order_depletes_bid_or_ask(event_type, bid_side, ask_side):
    flag = False

    if event_type == 0:
        # agg buy
        flag = ask_side[0][1] == 1

    if event_type == 1:
        # agg sell
        flag = bid_side[0][1] == 1

    return flag


def algo_trading_check(previous_bid, previous_ask,
                       bid_side, ask_side, step_size,
                       event_time, event_type, events, index, trade_history, inventory=100,
                       granularity_frequency=1 / 1000,
                       id_trader=1):
    # this simple algorithm will place an aggressive sell when there is a price improvement in the bid side

    # check if the spread was open at the previous state

    if id_trader == 1:
        # agg sell trader
        if previous_ask[0][0] - previous_bid[0][0] > step_size:

            # check if there was a bid improvement
            if previous_bid[0][0] < bid_side[0][0]:
                # schedule an aggressive sell order
                new_event_type = 1
                new_event_time = event_time + granularity_frequency
                pos = np.where(events[:, 0] <= new_event_time)[0][-1]
                new_event = np.array([new_event_time, new_event_type, id_trader])
                # pre-allocate the output array
                events_new = np.zeros((events.shape[0] + 1, events.shape[1]))

                # copy elements
                events_new[:pos + 1] = events[:pos + 1]

                # insert the new event
                events_new[pos + 1] = new_event

                # copy remaining elements
                events_new[pos + 2:] = events[pos + 1:]
            else:
                events_new = events
        else:
            events_new = events
    elif id_trader == 2:
        # passive buy
        if ask_side[0][0] - bid_side[0][0] > step_size:

            # check if the spread is open


            # schedule a passive buy order
            new_event_type = 2

            # passive trader is rapid
            new_event_time = event_time + granularity_frequency/2
            pos = np.where(events[:, 0] <= new_event_time)[0][-1]
            new_event = np.array([new_event_time, new_event_type, id_trader])
            # pre-allocate the output array
            events_new = np.zeros((events.shape[0] + 1, events.shape[1]))

            # copy elements
            events_new[:pos + 1] = events[:pos + 1]

            # insert the new event
            events_new[pos + 1] = new_event

            # copy remaining elements
            events_new[pos + 2:] = events[pos + 1:]
        else:
            events_new = events

    else:
        events_new = events
        # else do nothing

    return events_new

# this one create limit order book for a set of events
# actually this one is not needed when we play interactively
def create_limit_order_book(events, Ndepth, sigma=1, previous_mid_price=100.50, algo_trading=False, arg_input=None):
    step_size = float(sigma)
    time_max = int(arg_input['T_day'])
    if not arg_input is None:
        # inventory = arg_input['inventory']
        inventory_agg = np.array(([arg_input['inventory']]))
        inventory_pass = np.array(([arg_input['inventory']]))
        granularity_frequency = arg_input['granularity_frequency']
    else:
        inventory_agg = 0
        inventory_pass = 0
        granularity_frequency = np.Inf
    cash_inventory_agg = np.array(([0]))
    cash_inventory_pass = np.array(([0]))
    count_pass = 0

    # initializiation of the order book
    bid_side = []
    ask_side = []
    
    # define the data types of the elements in the tuple
    # trade_history_dtype = np.dtype(
    #     [('event_time', np.float64), ('event_type', np.float64), ('order_type', np.float64), ('price', np.float64),
    #      ('order_size', np.float64)])
    # # event type = -1 sell, 1 buy
    # # order_type = 0 market, 1 limit
    # trade_history = np.empty(1, dtype=trade_history_dtype)
    #
    trade_history = []
    trade_history.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
    trade_history = np.stack(trade_history, axis=0)

    # create a book with ten levels ( in the exogenous model there will be no missing points inside the bid and ask side
    # this is the memory of the book

    bid_side.append(np.concatenate(([previous_mid_price - step_size / 2, 1], [0], -1*np.ones(Ndepth - 1))))
    ask_side.append(np.concatenate(([previous_mid_price + step_size / 2, 1], [0], -1 * np.ones(Ndepth - 1))))

    for ii in range(2, 11):
        bid_side.append(np.concatenate(([previous_mid_price - (2 * ii - 1) * step_size / 2, 1],
                                        [0], -1 * np.ones(Ndepth - 1))))
        ask_side.append(np.concatenate(([previous_mid_price + (2 * ii - 1) * step_size / 2, 1],
                                        [0], -1 * np.ones(Ndepth - 1))))

    ask_side = np.stack(ask_side, axis=0)
    bid_side = np.stack(bid_side, axis=0)
    # bid_side.append(np.array([previous_mid_price-2*step_size, 1]))
    # ask_side.append(np.array([previous_mid_price+2*step_size, 1]))

    book_state = []
    book_state.append([{'bid': bid_side.copy(), 'ask': ask_side.copy()}])

    Nevents = len(events[:, 0])
    flag_price_improvement = None
    # to do the distance between Pa and Pb must by equal to sigma always
    traders_identity = np.zeros((1))
    index = 0
    event_time = 0

    while index < len(events) and event_time <= time_max:
        event_time, event_type, id_trader = events[index]
        """
        What we have here as an input:
        1. event time
        2. event_type
        3. ask_side
        4. bid_side
        5. trade_history
        6. flag_price_improvement
        7. step_size
        8. Ndepth
        9. previous_mid_price
        10. id_trader
        """
 
        ask_side, bid_side, trade_history, flag_price_improvement, passive_trader, new_item_dict = update_book_nb(event_time,
                                                                                                   event_type, ask_side,
                                                                                                   bid_side,
                                                                                                   trade_history,
                                                                                                   flag_price_improvement,
                                                                                                   step_size, Ndepth,
                                                                                                   previous_mid_price,
                                                                                                   id_trader)

        # Add trade to trade history
        # register_trade(new_item_dict)


        if len(ask_side) != 0 and len(bid_side) != 0:
            previous_mid_price = (ask_side[0][0] + bid_side[0][0]) / 2

        new_traders_identity = np.zeros(traders_identity.shape[0] + 1)
        new_traders_identity[:-1] = traders_identity
        new_traders_identity[-1] = id_trader
        traders_identity = new_traders_identity.copy()

        if algo_trading and index > 0:
            if id_trader == 1 and trade_history[-1][1] == -1 and trade_history[-1][2] == 0:
                inventory_agg = np.append(inventory_agg, inventory_agg[-1] - 1)
                cash_inventory_agg = np.append(cash_inventory_agg, cash_inventory_agg[-1] + trade_history[-1][3])
            else:
                inventory_agg = np.append(inventory_agg, inventory_agg[-1])
                cash_inventory_agg = np.append(cash_inventory_agg, cash_inventory_agg[-1])

            if passive_trader == 2:
                inventory_pass = np.append(inventory_pass, inventory_pass[-1] - 1)
                cash_inventory_pass = np.append(cash_inventory_pass, cash_inventory_pass[-1] - trade_history[-1][3])
            else:
                inventory_pass = np.append(inventory_pass, inventory_pass[-1])
                cash_inventory_pass = np.append(cash_inventory_pass, cash_inventory_pass[-1])

            if inventory_agg[-1] > 0:
                # schedule a new events if the condition of the algo trading strategies is satisfied
                events = algo_trading_check(book_state[-1][0]['bid'], book_state[-1][0]['ask'],
                                            bid_side, ask_side, step_size,
                                            event_time, event_type, events, index, trade_history, inventory_agg[-1],
                                            granularity_frequency=granularity_frequency, id_trader=1)

            if np.sum(
                    traders_identity == 2) < arg_input['inventory'] and id_trader != 2 and inventory_pass[-1] > 0:
                # the passive trader post only inventory_pass[0] orders, he can't remove past orders
                # schedule a new events if the condition of the algo trading strategies is satisfied

                events = algo_trading_check(book_state[-1][0]['bid'], book_state[-1][0]['ask'],
                                            bid_side, ask_side, step_size,
                                            event_time, event_type, events, index, trade_history, inventory_pass[-1],
                                            granularity_frequency=granularity_frequency, id_trader=2)


        book_state.append([{'bid': bid_side.copy(), 'ask': ask_side.copy()}])
        index += 1

        # compute the mid price

    agg_trader = {'inventory_agg': inventory_agg, 'cash_inventory_agg': cash_inventory_agg}
    pass_trader = {'inventory_pass': inventory_pass, 'cash_inventory_pass': cash_inventory_pass}
    return {"book_state": book_state, "trade_history": trade_history, "traders_identity": traders_identity,
            "agg_trader": agg_trader, 'pass_trader': pass_trader}



def arrival_times_trader(Lambda=0.25, Number_events_trader=10):
    # U = 1 - e ^ {- Lambda X} -->  1 - U = e ^ {-Lambda X}  --> log(1 - U) = - Lambda X
    #  --> X = -log(1 - U) / Lambda = -log(U) / Lambda
    time_intervals = np.random.exponential(scale=1 / Lambda, size=Number_events_trader)

    total_events = time_intervals.cumsum()
    events = pd.DataFrame(np.append(0, total_events),
                          np.cumsum(np.append(0, np.ones(Number_events_trader))))
    return events


