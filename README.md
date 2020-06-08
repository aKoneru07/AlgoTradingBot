# Stock Prediction Algorithm

This algorithm is a neural network-based algorithm which accurately predicts the opening price of a stock given a variable lookback period.

The algorithm is based on two fundamental principles: LSTM units and momentum technical indicators. Long short-term memory (LSTM) units are ideal for time-series predictions so they were a natural choice for extracting information from historical price and volume data. In the finance world, there are many technical indicators used for stock analysis. In this project, I have implemented a diverse set to maximize the information extraction. This set includes RSI, Stochastic Oscillator, MACD, and EMA.

_Note: Algorithm is currently under beta testing for paper trading and will eventually live-trade autonomously._

## Development Overview

This project was split into three Phases:

* Phase 1 _(complete)_
  * Implementing an accurate stock prediction Algorithm
* Phase 2 _(current)_
  * Backtesting algorithm over the last 2 years 
  * Running paper trading simulations to reveal insight on optimal buy/sell levels and order quantities
* Phase 3 _(TO DO)_
  * Launching Live Trading

### Phase 1
Phase 1 serves as the foundation of the entire project and thus required the most experimentation. The overall Neural Network architecture was two-pronged. The first input branch focuses on extracting information from common technical indicators used in market analysis. The following indicators were implemented: RSI, Stochastic Oscillator, WPR, MACD, and EMA. 

In early experimentation, it was noted that abundant information does _*not*_ equal impressive results. Rather the opposite. The model continuously struggled to predict accurately when provided data from many indicators. Instead, I selected a few distinct indicators to maximize information in a few metrics. The currently used indicators are RSI, Stochastic Oscillator, and WPR.

The second branch was treated the stock market as a predictable time-series function. The branch was centered around a LSTM layer. This layer is fed the daily open/close, high/low, and volume data for a variable lookback period. It is then trained to predict the next day's opening price.

### Phase 2 
Phase 2a was backtesting a basic algorithm in the market across the last ~2 years. The purpose here was not to create the most optimal algorithm but rather as a proof of concept. After several weeks of backtesting, I found that viable trading strategy could indeed be implemented from the Stock Predictions of Phase 1.

Phase 2b is to test the algorithm rigorously on the current market. To avoid potential losses, I decided to make this phase a paper-trading simulation. Unlike Phase 2a, these trials are meant to optimize the trading strategy based on the Phase 1 predications.

Please note that this phase is currently in progress. Make sure to check for updates!

### Phase 3
Phase 3 will be finally releasing the algorithm on the live market.

Make sure to check for updates!

