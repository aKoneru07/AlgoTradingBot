import numpy as np

def get_indicators(ohlcv_data, length = 14):
    high_price = ohlcv_data[len(ohlcv_data)-1][1]
    low_price  = ohlcv_data[len(ohlcv_data)-1][2]
    for i in range(len(ohlcv_data) - length, len(ohlcv_data)):
        if ohlcv_data[i][2] < low_price:
            low_price = ohlcv_data[i][2]

        if ohlcv_data[i][1] > high_price:
            high_price = ohlcv_data[i][1]

    return np.array([RSI(ohlcv_data), Stochastic_Oscillator(ohlcv_data, high_price, low_price),
                     WPR(ohlcv_data, high_price, low_price), MACD(ohlcv_data), EMA(ohlcv_data)])


def RSI(ohlcv_data, length = 14):               # based on closing price
    gain = 0
    loss = 0
    for i in range(len(ohlcv_data) - length, len(ohlcv_data)-1):
        if ohlcv_data[i][3] < ohlcv_data[i+1][3]:
            gain += ohlcv_data[i+1][3] - ohlcv_data[i][3]

        else:
            loss += ohlcv_data[i+1][3] - ohlcv_data[i][3]

    if loss == 0:
        return 100
    return 100 - (100/(1 + (gain/loss)))

def Stochastic_Oscillator(ohlcv_data, high_price, low_price):
    cp = ohlcv_data[len(ohlcv_data)-1][3]

    return 100 * (cp - low_price) / (high_price - low_price)

def WPR(ohlcv_data, high_price, low_price):
    cp = ohlcv_data[len(ohlcv_data) - 1][3]

    return -100 * (high_price - cp) / (high_price - low_price)

def MACD(ohlcv_data, length = 14):
    macd = EMA(ohlcv_data, 12) - EMA(ohlcv_data, 26)

    return macd - EMA(ohlcv_data, 9)

def EMA(ohlcv_data, length = 14):
    ema = SMA(ohlcv_data, length)
    multiplier = 2 / (length + 1)

    for i in range(len(ohlcv_data) - length, len(ohlcv_data)):
        ema = (ohlcv_data[i][3] * multiplier) + (ema * (1 - multiplier))
    return ema

def SMA(ohlcv_data, length = 14):
    sum_cp = 0
    for i in range(len(ohlcv_data) - length, len(ohlcv_data)):
        sum_cp += ohlcv_data[i][3]

    return sum_cp / length
