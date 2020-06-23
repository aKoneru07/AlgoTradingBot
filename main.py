import alpaca
import keras
from keras.models import Sequential, Model
from keras.layers import LSTM, Dropout, Dense, TimeDistributed, concatenate, Activation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import time

import history_processing
import simulation

api = alpaca.api

def MSE(real, predicted):
    assert(len(real) == len(predicted))
    mse = 0
    for i in range(len(predicted)):
        mse += pow((real[i]-predicted[i]), 2)
    return float((1/len(real)) * mse)

def train_model(TICKER, look_back_period = 50, SIMUL=True):

    # Pre-processing

    raw_data = history_processing.saveToCSV(TICKER)
    lead_infos, open_vals, indic_data, data_normalizer, real_data, CURRENT_SET = \
        history_processing.dataframeToData(raw_data, look_back_period)     # 50 day look-back period

    # Training and Testing Data Split
    test_split = 0.15
    # X_train, X_test, y_train, y_test = train_test_split(lead_infos, open_vals,
    #                                                     test_size=test_split, shuffle=False, stratify=None)
    # add random_state for reproducibility

    n = int(len(open_vals)*(1-test_split))
    indic_train = indic_data[0:n]
    indic_test = indic_data[n::]
    X_train = lead_infos[0:n]
    X_test = lead_infos[n::]
    y_train = open_vals[0:n]
    y_test = open_vals[n::]

    # Model

    indic_input = keras.Input(shape=(indic_data.shape[1],), name='indic_input')
    hidden_i = Dense(units=128, name='indic_dense_0')(indic_input)
    hidden_i = Activation(activation='relu', name='relu_0')(hidden_i)
    hidden_i = Dense(units=32, name='indic_dense_1')(hidden_i)
    hidden_i = Activation(activation='relu', name='relu_1')(hidden_i)
    hidden_i = Dropout(0.2, name='indic_dropout_0')(hidden_i)

    ohlcv_input = keras.Input(shape=(lead_infos.shape[1], lead_infos.shape[2]), name='ohlcv_input')
    hidden_h = LSTM(units=look_back_period, name='lstm_0')(ohlcv_input)
    hidden_h = Dropout(0.2, name='lstm_dropout_0')(hidden_h)

    combine = concatenate([hidden_i, hidden_h])

    combine = Dense(units=64, name='dense_0')(combine)
    combine = Activation(activation='sigmoid', name='sigmoid_0')(combine)
    combine = Dense(units=1, name='output')(combine)
    output = Activation(activation='linear', name='output_act')(combine)
    model = Model(inputs=[indic_input, ohlcv_input], outputs=output)
    model.compile(loss='mse', optimizer='adam')

    model.fit(x=[indic_train, X_train], y=y_train, batch_size=32, epochs=10, shuffle=True, validation_split=0.1)
    evaluation = model.evaluate([indic_test, X_test], y_test)
    print("Eval: " + str(evaluation))

    # Converting percent change predictions to stock value predictions

    y_test_prediction = model.predict([indic_test, X_test])
    y_test_prediction = data_normalizer.inverse_transform(y_test_prediction)     # scale back from 0 to 1
    y_test = data_normalizer.inverse_transform(y_test)
    mse_run = MSE(y_test, y_test_prediction)
    print("MSE: " + str(mse_run))

    y_test = [real_data[i + n] * (1 + y_test[i]) for i in range(len(y_test))]   # convert percent change to stock value
    y_test_prediction = [real_data[i + n] * (1 + y_test_prediction[i]) for i in range(len(y_test_prediction))]

    mse_run = MSE(y_test, y_test_prediction)
    print("MSE: " + str(mse_run))

    # Running simulation with basic strategy
    if SIMUL:
        # profit, buys, sells, big_buys = simulation.run_simul(y_test, y_test_prediction, verbose=False)
        profit, buys, sells, big_buys = simulation.run_two_day_simul(y_test, y_test_prediction, verbose=False)

    # Graphing Model predictive performance
    if SIMUL:
        plt.gcf().set_size_inches(22, 15, forward=True)

        start = 0
        end = -1

        plt.title(TICKER + str(' Stock Prediction_rsi_stoch_wpr_128_32_50e_SIMUL    2_day    MSE:') + str(round(mse_run, 2)))

        plt.plot(y_test[start:end], label='Real')
        plt.plot(y_test_prediction[start:end], label='Predicted')

        plt.scatter(buys[0], buys[1], c='r')
        plt.scatter(sells[0], sells[1], c='g')
        plt.scatter(big_buys[0], big_buys[1], c='m')

        plt.legend(loc='upper left')

        plt.show()

    current_prediction = model.predict([[CURRENT_SET[1]], [CURRENT_SET[0]]])
    current_prediction = data_normalizer.inverse_transform(current_prediction)[0][0]


    return model, current_prediction

def get_last_close(ticker):
    barset = api.get_barset(ticker, 'day', limit=1)
    barset = barset[ticker]

    return barset[-1].c

def main():
    print("Hello World!")
    account = api.get_account()
    # Check if our account is restricted from trading.
    if account.trading_blocked:
        print('Account is currently restricted from trading.')

    # Check how much money we can use to open new positions.
    print('${} is available as buying power.'.format(account.buying_power))

    # Train Model
    # model, predict = train_model('MSFT', look_back_period=50)

    tickers = ['MSFT', 'AAPL', 'AAL', 'UAL', 'CAKE', 'PENN', 'SNAP', 'TWTR']

    nextTrain = dt.datetime.now()
    nextTrain = nextTrain.replace(hour=16, minute=30, second=0)       # trains at 4:30 p.m. every day

    while True:

        time_to_train = (nextTrain - dt.datetime.now()).total_seconds()
        print("Next Training in " + str(time_to_train) + " seconds")
        # time.sleep(time_to_train)

        if dt.datetime.now() > nextTrain:
            tick_models = []
            tick_predictions = []

            # TRAIN MODEL
            for tick in tickers:
                print("Evaluating: " + tick)
                model, predict = train_model(tick, look_back_period=50, SIMUL=False)
                tick_models.append(model)
                tick_predictions.append(predict)

            # EXECUTE TRADES
            print("Analyzing Potential Trades")
            portfolio = api.list_positions()
            positions = []
            for p in portfolio:
                positions.append(p.symbol)
            for i in range(len(tickers)):
                print(tickers[i] + " | Prediction: " + str(tick_predictions[i]))
                buying_power = int(float(api.get_account().buying_power))
                if tick_predictions[i] > 0.01 and buying_power > 1100:                      # Buy $1000 of shares max
                    current_price = get_last_close(tickers[i])
                    order_size = int(min(1000 / current_price, (buying_power / current_price)-1))
                    api.submit_order(
                        symbol=tickers[i],
                        qty=order_size,
                        side='buy',
                        type='market',
                        time_in_force='gtc'
                    )
                    print("Buying " + str(order_size) + " shares of " + tickers[i])

                elif tick_predictions[i] < -0.005 and tickers[i] in positions:         # Sell ALL shares
                    api.submit_order(
                        symbol=tickers[i],
                        qty=api.get_position(tickers[i]).qty,
                        side='sell',
                        type='market',
                        time_in_force='gtc'
                    )
                    print("Selling " + str(api.get_position(tickers[i]).qty) + " shares of " + tickers[i])

            nextTrain += dt.timedelta(days = 1)
        else:
            print("Not yet: " + str(dt.datetime.now().hour))
            time.sleep(600)

if __name__ == "__main__":
    main()
