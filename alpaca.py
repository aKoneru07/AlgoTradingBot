import alpaca_trade_api as tradeapi
import json
# pip install alpaca-trade-api

global api

creds = json.load(open('creds.json', 'r'))

api = tradeapi.REST(creds["alpaca_key_id"],
                    creds["alpaca_secret_key"],
                    creds["alapca_baseurl"])
