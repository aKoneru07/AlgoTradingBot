import alpaca
import history_processing
import keras
from keras.models import Sequential, Model
from keras.layers import LSTM, Dropout, Dense, TimeDistributed, concatenate, Activation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


api = alpaca.api
LOOK_BACK_PERIOD = 50
TICKER = 'MSFT'

def MSE(real, predicted):
    assert(len(real) == len(predicted))
    mse = 0
    for i in range(len(predicted)):
        mse += pow((real[i]-predicted[i]), 2)
    return (1/len(real)) * mse

def run_simul(real, predicted):
    assert (len(real) == len(predicted))
    buys = [[],[]]
    sells = [[],[]]
    spent = 0
    num_shares = 0
    revenue = 0
    for i in range(len(predicted)-1):
        change = (predicted[i+1] - real[i]) / (real[i])
        # if change > 0.005:
        #     print("CHANGE: " + str(change))
        if change > 0.005:
            spent += real[i]
            num_shares += 1
            buys[0].append(i)
            buys[1].append(real[i])
            print("BOUGHT AT: " + str(real[i]))
        if change < -0.005 and num_shares > 0:
            revenue += num_shares * real[i]
            num_shares = 0
            sells[0].append(i)
            sells[1].append(real[i])
            print("SOLD AT: " + str(real[i]))

    return revenue - spent, buys, sells

def main():
    print("Hello World!")
    account = api.get_account()
    # Check if our account is restricted from trading.
    if account.trading_blocked:
        print('Account is currently restricted from trading.')

    # Check how much money we can use to open new positions.
    print('${} is available as buying power.'.format(account.buying_power))

    # Pre-processing

    raw_data = history_processing.saveToCSV(TICKER)
    lead_infos, open_vals, indic_data, data_normalizer = history_processing.dataframeToData(raw_data, LOOK_BACK_PERIOD)     # 50 day look-back period

    print("Look-back Data Shape: " + str(lead_infos.shape))
    print("Open Val Data Shape: " + str(open_vals.shape))
    print("TRANSFORMED")
    print(data_normalizer.inverse_transform(open_vals))
    # print("RAW")
    # print(raw_open_data)

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
    hidden_h = LSTM(units=LOOK_BACK_PERIOD, name='lstm_0')(ohlcv_input)
    hidden_h = Dropout(0.2, name='lstm_dropout_0')(hidden_h)

    combine = concatenate([hidden_i, hidden_h])

    combine = Dense(units=64, name='dense_0')(combine)
    combine = Activation(activation='sigmoid', name='sigmoid_0')(combine)
    combine = Dense(units=1, name='output')(combine)
    output = Activation(activation='linear', name='output_act')(combine)
    model = Model(inputs=[indic_input, ohlcv_input], outputs=output)
    model.compile(loss='mse', optimizer='adam')

    model.fit(x=[indic_train, X_train], y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)
    evaluation = model.evaluate([indic_test, X_test], y_test)
    print("Eval: " + str(evaluation))

    y_test_prediction = model.predict([indic_test, X_test])
    y_test_prediction = data_normalizer.inverse_transform(y_test_prediction)     # scale back from 0 to 1
    mse_run = MSE(data_normalizer.inverse_transform(y_test), y_test_prediction)[0]
    print("MSE: " + str(mse_run))

    profit, buys, sells = run_simul(data_normalizer.inverse_transform(y_test), y_test_prediction)
    print("Profited: " + str(profit))
    plt.gcf().set_size_inches(22, 15, forward=True)

    start = 0
    end = -1

    plt.title(TICKER + str(' Stock Prediction_rsi_stoch_wpr_128_32_50e_SIMUL    MSE:') + str(round(mse_run, 2)))

    plt.plot(data_normalizer.inverse_transform(y_test)[start:end], label='Real')
    plt.plot(y_test_prediction[start:end], label='Predicted')

    plt.scatter(buys[0], buys[1], c='r')
    plt.scatter(sells[0], sells[1], c='g')

    plt.legend(loc='upper left')

    plt.show()

# model = Sequential()
#
# #model.add(Embedding(input_dim=1173, output_dim=64, input_length=1173))
# model.add(LSTM(units=16, input_shape=(dataX.shape[1], dataX.shape[2])))
#
# model.add(Dense(1173, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# model.fit(dataX, dataY, epochs=1)


if __name__ == "__main__":
    main()