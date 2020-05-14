import alpaca
import history_processing
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


api = alpaca.api
LOOK_BACK_PERIOD = 50
TICKER = 'MSFT'

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
    lead_infos, open_vals, data_normalizer = history_processing.dataframeToData(raw_data, LOOK_BACK_PERIOD)     # 50 day look-back period

    print("Look-back Data Shape: " + str(lead_infos.shape))
    print("Open Val Data Shape: " + str(open_vals.shape))
    print("TRANSFORMED")
    print(data_normalizer.inverse_transform(open_vals))
    # print("RAW")
    # print(raw_open_data)

    # Training and Testing Data Split
    test_split = 0.15
    X_train, X_test, y_train, y_test = train_test_split(lead_infos, open_vals,
                                                        test_size=test_split, shuffle=False, stratify=None)
    # add random_state for reproducibility


    # Model

    model = Sequential(name=TICKER+str(' LSTM Model'))
    model.add(LSTM(units=LOOK_BACK_PERIOD, input_shape=(lead_infos.shape[1], lead_infos.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(units=LOOK_BACK_PERIOD, activation="sigmoid"))
    model.add(Dense(units=1, activation="linear"))
    model.compile(loss='mse', optimizer='adam')
    print(model.summary())

    model.fit(x=X_train, y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)
    evaluation = model.evaluate(X_test, y_test)
    # print("Eval: " + str(evaluation))

    y_test_prediction = model.predict(X_test)
    y_test_prediction = data_normalizer.inverse_transform(y_test_prediction)     # scale back from 0 to 1

    # y_test_prediction = []
    # initial_test = X_test[0]
    # for i in range(50):
    #     print("PREDICTED: " + str(model.predict(initial_test)))

    plt.gcf().set_size_inches(22, 15, forward=True)

    start = 0
    end = -1

    plt.title(TICKER + str(' Stock Prediction'))

    plt.plot(data_normalizer.inverse_transform(y_test)[start:end], label='real')
    plt.plot(y_test_prediction[start:end], label='predicted')

    # print(y_test_prediction[start:end])
    # print(data_normalizer.inverse_transform(y_test)[start:end])

    plt.legend(['Real', 'Predicted'])

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