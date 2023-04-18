# header
import time
import sys
import exogeneous_model
from pprint import pprint
if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import sys
    import scipy as scipy
    from scipy import stats
    import matplotlib.pyplot as plt
    import scipy.stats as st
    from matplotlib.ticker import ScalarFormatter
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from scipy.stats import norm
    from datetime import datetime
    import statsmodels.api as sm

    import math


    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from PIL import Image
    import imageio
    import time

    # Define the function to create each frame of the animation
    print('hello')
    
    settings = {
        'hours': 60,  # minutes
        'T_day': 60 * 6.5,  # mins in a trading day; trading day length in munutes
        # 60/3*0.5,  # 1/min   # 1 = 1min freq,  2  = a trading occurs every 30 sec, 0.5 = traders every 2 min
        'freq_agg': 60/2*0.5, # number of aggresive orders per minute
        'freq_pass': 60/2,  # 60/3, number of passive orders per minute
        'N_depth': 5,
        'sigma': 100.0,  # price are multiplied by 100, 1/1e2 is one basis point
        'previous_mid_price': 10000,
        'seed': 2023,
    }

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

    len(event_book)
    tic = time.time()
    out = exog_fun.create_limit_order_book(event_book, settings['N_depth'], settings['sigma'],
                                           settings['previous_mid_price'], algo_trading=False, arg_input=arg_input)
    print(time.time()-tic)
    print(out.keys())
    print(type(out['book_state']))
    print(len(out['book_state']))
    for k,v in out.items():
        print(k)
        try:
            print(type(v))
        except Exception as e:
            pass
    # print(out['trade_history'].shape)
    # print(out['trade_history'][:10])
    df = pd.DataFrame(out['trade_history'], columns = ['timestamp','direction','order_type', 'price','agression'])
    df.to_csv('./myfile.csv')
    # print(df.head)
    # print(df['a'].nunique())
    # print(df['b'].nunique())
    # print(df['c'].unique())
    pprint(df.head)
    # print(df['d'].nunique())
   
    print('+'*100)
    sys.exit()

##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################

    # #statistics and plot
    # Create list to store mid prices and times
    mid_prices = []
    times = []

    P_a = []
    P_b = []
    spread = []

    t = 0

    frames = []
    book = out['book_state']

    # Iterate through book data to calculate mid prices and times
    for booki in book:
        if len(booki[0]['bid']) != 0 and len(booki[0]['ask']) != 0:
            bid_price = booki[0]['bid'][0][0]
            ask_price = booki[0]['ask'][0][0]
            mid_price = (bid_price + ask_price) / 2
            mid_prices.append(mid_price)
            P_a.append(ask_price)
            P_b.append(bid_price)
            spread.append(ask_price-bid_price)
        else:
            mid_prices.append(np.nan)
            P_a.append(np.nan)
            P_b.append(np.nan)
            spread.append(np.nan)

        times.append(t)
        t += 1

    mid_ret = np.diff(mid_prices)
    #mid_ret = mid_ret[np.where(mid_ret != 0)]

    # Create a list of frames for the animation
    P_b = np.array(P_b)
    P_a = np.array(P_a)
    #
    # for i in range(0, len(book), int(len(book) / 10)):
    #     frames.append(
    #         exogeneous_model.create_frame(book, i, settings['N_depth'], max(mid_prices)+4*settings['sigma'],
    #                                       min(mid_prices)-4*settings['sigma']))

    # Save the frames as a GIF using imageio
    with imageio.get_writer('lob_animation.gif', mode='I', duration=0.5) as writer:
        for frame in frames:
            writer.append_data(np.array(frame))

    plt.close('all')

    # Create plot
    fig, ax = plt.subplots()
    y_formatter = ScalarFormatter(useOffset=False)
    ax.yaxis.set_major_formatter(y_formatter)
    plt.plot(times, mid_prices, color='black', label='Mid Price')
    plt.plot(times, P_a, color='red', label='P_a')
    plt.plot(times, P_b, color='blue', label='P_b')

    # Add legend and labels
    ax.legend(loc='upper left')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_title('Order Book Dynamics Over Time')

    plt.show()

    from statsmodels.graphics.tsaplots import plot_acf

    # Plot the ACF of the mid price
    mid_prices = np.array(mid_prices)
    fig, ax = plt.subplots()
    plot_acf(mid_prices[~np.isnan(mid_prices)], ax=ax, lags=10)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Autocorrelation Function of Mid Price')
    plt.show()

    fig, ax = plt.subplots()
    plot_acf(mid_ret[~np.isnan(mid_ret)], ax=ax, lags=10)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Autocorrelation Function of Mid Return')
    plt.show()

    fig, ax = plt.subplots()
    plt.plot(spread)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Spread')
    ax.set_title('Spread')
    plt.show()

    # analysis of SPY book
    data_spy = pd.read_csv(
        "/home/fcordoni/Desktop/lobster_data/AAPL_2019-03-18_34200000_57600000_orderbook_10.csv", header=None)
    # data_spy = pd.read_csv("/home/fcordoni/Desktop/lobster_data/SPY_2016-07-27_34200000_57600000_orderbook_1.csv",
    #                        header=None)
    # data_spy = pd.read_csv("/home/fcordoni/Desktop/lobster_data/AMZN_2019-08-27_34200000_57600000_orderbook_10.csv",
    #                        header=None)
    # data_spy = pd.read_csv("/home/fcordoni/Desktop/lobster_data/MSFT_2019-03-18_34200000_57600000_orderbook_10.csv",
    #                        header=None)

    # SPY acf negative

    P_a_spy = data_spy.iloc[:, 0]
    P_b_spy = data_spy.iloc[:, 2]
    mid_prices_spy = (P_a_spy+P_b_spy)/2
    mid_ret_spy = np.diff(mid_prices_spy)

    print(len(P_b_spy))

    fig, ax = plt.subplots()
    plot_acf(mid_ret_spy[20000:50000], ax=ax, lags=10)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Autocorrelation Function of Mid Return AAPL')
    plt.show()

    spread = P_a_spy-P_b_spy
    fig, ax = plt.subplots()
    plt.plot(spread[20000:50000])
    ax.set_xlabel('Lag')
    ax.set_ylabel('Spread')
    ax.set_title('Spread Function of Mid Return AAPL')
    plt.show()

    # spread over time

    # # Iterate through book data to plot asks and bids
    # for booki in book:
    #     for i, ask in enumerate(booki[0]['ask']):
    #         ax.yaxis.set_major_formatter(y_formatter)
    #         plt.plot([times[0], times[-1]], [ask[0], ask[0]], color='blue', alpha=0.3)
    #     for i, bid in enumerate(booki[0]['bid']):
    #         ax.yaxis.set_major_formatter(y_formatter)
    #         plt.plot([times[0], times[-1]], [bid[0], bid[0]], color='red', alpha=0.3)

    # # Set x and y limits
    # ax.set_xlim(times[0], times[-1])
    #

    # book = [{'bid': [np.array([99.985, 9]), np.array([99.98, 10])],
    #          'ask': [np.array([99.99, 10]), np.array([99.995, 10]), np.array([100., 10]),
    #                  np.array([100.005, 10]), np.array([100.01, 10]), np.array([100.015, 10])]},
    #         {'bid': [np.array([99.985, 10]), np.array([99.98, 10])],
    #          'ask': [np.array([99.99, 10]), np.array([99.995, 10]), np.array([100., 10]),
    #                  np.array([100.005, 10]), np.array([100.01, 10]), np.array([100.015, 10])]}]
    #
    # bids = [bid[0] for bid in book[0]['bid']]
    # bid_sizes = [bid[1] for bid in book[0]['bid']]
    # asks = [ask[0] for ask in book[0]['ask']]
    # ask_sizes = [ask[1] for ask in book[0]['ask']]
    #
    # fig, ax = plt.subplots()
    #
    # # plt.plot(ask_sizes, asks)
    # y_formatter = ScalarFormatter(useOffset=False)
    # # plt.plot(bid_sizes, bids)
    # ax.yaxis.set_major_formatter(y_formatter)
    # # Plot the bid side
    # ax.barh(bids, bid_sizes, height=0.005, color='g', alpha=0.8)
    #
    # # Plot the ask side
    # ax.barh(asks, ask_sizes, height=0.005, color='r', alpha=0.8)
    # ax.set_xlabel('Volume')
    # ax.set_ylabel('Price')
    # ax.set_title('Limit Order Book')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7))

    ax1.plot(out['agg_trader']['inventory_agg'])
    ax1.set_title('Inventory Agg')
    ax2.plot(out['agg_trader']['cash_inventory_agg'])
    ax2.set_title('Cash Agg')
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7))

    ax1.plot(out['pass_trader']['inventory_pass'])
    ax1.set_title('Inventory Pass')
    ax2.plot(out['pass_trader']['cash_inventory_pass'])
    ax2.set_title('Cash Pass')
    plt.show()
