# Stock Prediction Algorithm

This algorithm is a neural network-based algorithm which accurately predicts the opening price of a stock given a variable lookback period.

The algorithm is based on two fundamental principles: LSTM units and momentum technical indicators. Long short-term memory (LSTM) units are ideal for time series predictions so they were a natural choice for extracting information from historical price and volume data. In the finance world, there are many technical indicators used for stock analysis. In this project, I have implemented a diverse set to maximize the information extraction. This set includes RSI, Stochastic Oscillator, MACD, and EMA.

NOTE: Algorithm is currently under beta testing for paper trading and will eventually live-trade autonomously.
