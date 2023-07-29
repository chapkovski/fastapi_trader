import pandas as pd
import numpy as np
from pprint import pprint
from numba import jit
from numba import types, typed
from numba.typed import List
from log import logger

# scroll the queue to the first free level
@jit(nopython=True)
def scroll_queue_update(
    side: np.ndarray, step_size: float, Ndepth: int, direction: int, id_trader: float
) -> tuple:
    # def scroll_queue_update(side, step_size, Ndepth, direction):
    # direction 1 (ask), -1 (bid)
    cond = True
    iqueue = 1
    while cond:
        if iqueue == len(side):
            cond = False
            # create the new level
            new_level = np.empty((1, 2 + Ndepth))
            new_level[0][0] = side[iqueue - 1][0] + step_size * direction
            new_level[0][1] = id_trader
            new_level[0][2] = 0
            new_level[0][3:] = -1
            side = np.vstack((side, new_level))
        else:
            if side[iqueue][1] + 1 > Ndepth:
                iqueue = iqueue + 1  # try next level
            else:
                cond = False
                side[iqueue][1] += 1
                pos = side[iqueue][1]
                side[iqueue][int(pos + 2 - 1)] = id_trader
            # level with the first available spot

    return iqueue, side


# @jit(nopython=True)
def update_book_nb(
    event_time: float,
    event_type: int,
    ask_side: np.ndarray,
    bid_side: np.ndarray,
    trade_history: np.ndarray,
    flag_price_improvement: int,
    step_size: float,
    Ndepth: int,
    previous_mid_price: float,
    id_trader: float,
) -> tuple:
    args = dict(
        event_time=event_time,
        event_type=event_type,
        ask_side=ask_side,
        bid_side=bid_side,
    )

    # event_type
    #       0  aggressive buy
    #       1  aggressive sell
    #       2  passive buy  2.5 # buy with price improvements
    #       3  passive sell 3.5 # sell with price improvements
    #
    # event type = -1 sell, 1 buy
    # order_type = 0 market, 1 limit

    # check if the book is empty
    passive_id_trader = -1  # i.e., there is no passive traders
    if len(bid_side) != 0 or len(ask_side) != 0:
        if event_type == 0:
            # Check if there are any sell orders to match

            # Add trade to trade history
            new_trade = np.array(
                [(event_time, float(1), float(0), ask_side[0][0], event_type)]
            )
            trade_history = np.vstack((trade_history, new_trade))

            # Remove matched order from sell book
            ask_side[0][1] = ask_side[0][1] - 1
            arr = ask_side[0].copy()
            passive_id_trader = arr[2].copy()
            arr[2:(-1)] = arr[3:]
            arr[-1] = -1

            ask_side[0] = arr

            if ask_side[0][1] == 0:
                ask_side = ask_side[1:]
                flag_price_improvement = "bid"

        elif event_type == 1:
            # Add trade to trade history
            new_trade = np.array(
                [(event_time, float(-1), float(0), bid_side[0][0], event_type)]
            )
            trade_history = np.vstack((trade_history, new_trade))

            # Remove matched order from buy book
            bid_side[0][1] = bid_side[0][1] - 1

            arr = bid_side[0].copy()
            passive_id_trader = arr[2].copy()
            arr[2:(-1)] = arr[3:]
            arr[-1] = -1

            bid_side[0] = arr

            if bid_side[0][1] == 0:
                bid_side = bid_side[1:]
                flag_price_improvement = "ask"

        elif event_type == 2:
            # add the bid price

            if flag_price_improvement == "bid":
                if bid_side[0][0] + step_size == ask_side[0][0]:
                    #
                    print(
                        "Spread is already closed with a previous price improvement in the bid"
                    )
                    flag_price_improvement = None
                    if (bid_side[0][1] + 1) > Ndepth:
                        # update the queue in the next free level
                        iqueue, bid_side = scroll_queue_update(
                            bid_side, step_size, Ndepth, -1, id_trader
                        )
                        if bid_side[iqueue][0] < 0:
                            # stack all the bid at last level
                            bid_side = bid_side[:-1]
                            iqueue = iqueue - 1
                        new_trade = np.array(
                            [
                                (
                                    event_time,
                                    float(1),
                                    float(1),
                                    bid_side[iqueue][0],
                                    event_type,
                                )
                            ]
                        )
                        trade_history = np.vstack((trade_history, new_trade))

                    else:
                        bid_side[0][1] = bid_side[0][1] + 1
                        pos = bid_side[0][1]
                        bid_side[0][int(pos + 2 - 1)] = id_trader

                        new_trade = np.array(
                            [
                                (
                                    event_time,
                                    float(1),
                                    float(1),
                                    bid_side[0][0],
                                    event_type,
                                )
                            ]
                        )
                        trade_history = np.vstack((trade_history, new_trade))
                else:
                    new_bid = np.empty((1, 2 + Ndepth))
                    new_bid[0][0] = bid_side[0][0] + step_size
                    new_bid[0][1] = 1
                    new_bid[0][2] = id_trader
                    new_bid[0][3:] = -1
                    bid_side = np.vstack((new_bid, bid_side))

                    if bid_side[0][0] + step_size >= ask_side[0][0]:
                        flag_price_improvement = None

                    new_trade = np.array(
                        [(event_time, float(1), float(1), bid_side[0][0], 2.5)]
                    )
                    trade_history = np.vstack((trade_history, new_trade))

                # if (ask_side[0][0] - bid_side[0][0]) == 2*step_size:
                # if P_a - P_b == spread then the bid is not improved, but this can not be happened, since
                # if we are here, the top ask price was depleted, and the spread was opened
            else:
                if (bid_side[0][1] + 1) > Ndepth:
                    # update the queue in the next free level

                    iqueue, bid_side = scroll_queue_update(
                        bid_side, step_size, Ndepth, -1, id_trader
                    )
                    if bid_side[iqueue][0] < 0:
                        # stack all the bid at last level
                        bid_side = bid_side[:-1]
                        iqueue = iqueue - 1
                    new_trade = np.array(
                        [
                            (
                                event_time,
                                float(1),
                                float(1),
                                bid_side[iqueue][0],
                                event_type,
                            )
                        ]
                    )
                    trade_history = np.vstack((trade_history, new_trade))

                else:
                    bid_side[0][1] = bid_side[0][1] + 1

                    pos = bid_side[0][1]
                    bid_side[0][int(pos + 2 - 1)] = id_trader

                    new_trade = np.array(
                        [(event_time, float(1), float(1), bid_side[0][0], event_type)]
                    )
                    trade_history = np.vstack((trade_history, new_trade))
            # Add trade to trade history
        elif event_type == 3:
            if flag_price_improvement == "ask":
                if ask_side[0][0] - step_size == bid_side[0][0]:
                    print(
                        "Spread is already closed with a previous price improvement in the ask"
                    )
                    flag_price_improvement = None
                    if (ask_side[0][1] + 1) > Ndepth:
                        # update the queue in the next free level

                        iqueue, ask_side = scroll_queue_update(
                            ask_side, step_size, Ndepth, 1, id_trader
                        )
                        new_trade = np.array(
                            [
                                (
                                    event_time,
                                    float(-1),
                                    float(1),
                                    ask_side[iqueue][0],
                                    event_type,
                                )
                            ]
                        )
                        trade_history = np.vstack((trade_history, new_trade))

                    else:
                        ask_side[0][1] = ask_side[0][1] + 1

                        pos = ask_side[0][1]
                        ask_side[0][int(pos + 2 - 1)] = id_trader

                        new_trade = np.array(
                            [
                                (
                                    event_time,
                                    float(-1),
                                    float(1),
                                    ask_side[0][0],
                                    event_type,
                                )
                            ]
                        )
                        trade_history = np.vstack((trade_history, new_trade))
                else:
                    new_ask = np.empty((1, 2 + Ndepth))
                    new_ask[0][0] = ask_side[0][0] - step_size
                    new_ask[0][1] = 1
                    new_ask[0][2] = id_trader
                    new_ask[0][3:] = -1
                    ask_side = np.vstack((new_ask, ask_side))

                    if ask_side[0][0] - step_size <= bid_side[0][0]:
                        flag_price_improvement = None

                    new_trade = np.array(
                        [(event_time, float(-1), float(1), ask_side[0][0], 3.5)]
                    )
                    trade_history = np.vstack((trade_history, new_trade))

                # if (ask_side[0][0] - bid_side[0][0]) == 2*step_size:
                # if P_a - P_b == spread then the ask_side is not improved, but this can not be happened, since
                # if we are here, the top bid price was depleted, and the spread was opened
            else:
                if (ask_side[0][1] + 1) > Ndepth:
                    # update the queue in the next free level

                    iqueue, ask_side = scroll_queue_update(
                        ask_side, step_size, Ndepth, 1, id_trader
                    )
                    new_trade = np.array(
                        [
                            (
                                event_time,
                                float(-1),
                                float(1),
                                ask_side[iqueue][0],
                                event_type,
                            )
                        ]
                    )
                    trade_history = np.vstack((trade_history, new_trade))

                else:
                    ask_side[0][1] = ask_side[0][1] + 1

                    pos = ask_side[0][1]
                    ask_side[0][int(pos + 2 - 1)] = id_trader

                    new_trade = np.array(
                        [(event_time, float(-1), float(1), ask_side[0][0], event_type)]
                    )
                    trade_history = np.vstack((trade_history, new_trade))

    else:
        # check if the book is empty
        if len(bid_side) == 0:
            if len(ask_side) == 0:
                bid_price = previous_mid_price - step_size / 2
            else:
                bid_price = ask_side[0][0] - step_size

            bid_volume = 1
            bid_side = np.array(
                [
                    np.hstack(
                        ([bid_price, bid_volume, id_trader], -1 * np.ones(Ndepth - 1))
                    )
                ],
                dtype=np.float64,
            )

            new_trade = np.array(
                [(event_time, float(1), float(1), bid_side[0][0], 2.25)]
            )
            trade_history = np.vstack((trade_history, new_trade))

        if len(ask_side) == 0:
            if len(bid_side) == 0:
                ask_price = previous_mid_price + step_size / 2
            else:
                ask_price = bid_side[0][0] + step_size

            ask_volume = 1
            ask_side = np.array(
                [
                    np.hstack(
                        ([ask_price, ask_volume, id_trader], -1 * np.ones(Ndepth - 1))
                    )
                ],
                dtype=np.float64,
            )

            new_trade = np.array(
                [(event_time, float(-1), float(1), ask_side[0][0], 3.25)]
            )
            trade_history = np.vstack((trade_history, new_trade))
    trade_cols = ["timestamp", "direction", "order_type", "price", "agression"]

    trade_item_dict = {
        key: value for key, value in zip(trade_cols, new_trade.flatten())
    }
    trade_item_dict["trader_id"] = id_trader
    
    return (
        ask_side,
        bid_side,
        trade_history,
        flag_price_improvement,
        passive_id_trader,
        trade_item_dict,
    )
