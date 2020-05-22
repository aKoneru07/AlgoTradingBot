from alpha_vantage.timeseries import TimeSeries
import yfinance as yf
import json
import pandas as pd
from sklearn import preprocessing
import numpy as np
from indicators import get_indicators


def saveToCSV(ticker):
    creds = json.load(open('creds.json', 'r'))

    ts = TimeSeries(key=creds["alpha_vantage_key"], output_format='pandas')
    data, meta_data = ts.get_daily(ticker, outputsize='full')
    data.to_csv(f'./{ticker}_daily.csv')

    # data = yf.Ticker(ticker).history(period="10y")
    # data.to_csv(f'./{ticker}_daily_yf.csv')

    print(data.head())

    # data = csv_to_dataset(f'./{ticker}_daily.csv')           #####
    return data


def csv_to_dataset(csv_path):
    data = pd.read_csv(csv_path)
    data = data.drop('date', axis=1)
    # data = data.drop(0, axis=0)
    print(data.head())
    return data

def dataframeToData(data, length):
    data = data.values
    data = data[::-1]           # reverses list to increasing time
    data = data[1::]            # removes first day (often outlier data for recently IPO companies
    data_normaliser = preprocessing.MinMaxScaler()
    data_normalised = data_normaliser.fit_transform(data)

    ohlcv_data = np.array([data_normalised[i:i+length] for i in range(len(data_normalised) - length)])
    open_data_normal = np.array([data_normalised[i + length][0] for i in range(len(data_normalised) - length)])
    # open_data_normal = np.array([data_normalised[i + length][0:3] for i in range(len(data_normalised) - length - 3)])
    # print(open_data_normal)

    open_data_normal = np.expand_dims(open_data_normal, -1)
    open_data_normal = np.reshape(open_data_normal, (open_data_normal.shape[0], open_data_normal.shape[1]))
    print(open_data_normal)

    open_data = np.array([data[i+length][0] for i in range(len(data) - length)])
    open_data = np.expand_dims(open_data, -1)

    y_normaliser = preprocessing.MinMaxScaler()
    y_normaliser.fit(open_data)

    indic_data = np.array([get_indicators(ohlcv_data[i], 14) for i in range(len(ohlcv_data))])
    print(indic_data)

    # print("high")
    # print(ohlcv_data)
    # print("low")
    # print(open_data_normal)
    # print(ohlcv_data[1][length-1])

    return ohlcv_data, open_data_normal, indic_data, y_normaliser