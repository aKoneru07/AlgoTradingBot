from alpha_vantage.timeseries import TimeSeries
import yfinance as yf
import json
import pandas as pd
from sklearn import preprocessing
import numpy as np


def saveToCSV(ticker):
    creds = json.load(open('creds.json', 'r'))

    ts = TimeSeries(key=creds["alpha_vantage_key"], output_format='pandas')
    data, meta_data = ts.get_daily(ticker, outputsize='full')
    data.to_csv(f'./{ticker}_daily.csv')

    # data = yf.Ticker(ticker).history(period="10y")
    # data.to_csv(f'./{ticker}_daily_yf.csv')

    # print(data.head())

    return data



def csv_to_dataset(csv_path):
    data = pd.read_csv(csv_path)
    data = data.drop('date', axis=1)
    data = data.drop(0, axis=0)
    print(data.head())

def dataframeToData(data, length):
    data = data.values
    data_normaliser = preprocessing.MinMaxScaler()
    data_normalised = data_normaliser.fit_transform(data)

    ohlcv_data = np.array([data_normalised[i:i+length] for i in range(len(data_normalised) - length)])
    open_data_normal = np.array([data_normalised[i + length][0] for i in range(len(data_normalised) - length)])
    open_data_normal = np.expand_dims(open_data_normal, -1)

    open_data = np.array([data[i+length][0] for i in range(len(data) - length)])
    open_data = np.expand_dims(open_data, -1)

    y_normaliser = preprocessing.MinMaxScaler()
    open_data = y_normaliser.fit(open_data)

    # print("high")
    # print(ohlcv_data)
    # print("low")
    # print(open_data)
    #
    # print(ohlcv_data[1][length-1])

    return ohlcv_data, open_data_normal, y_normaliser