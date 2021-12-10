import yfinance as yf
import numpy as np
import pandas as pd


yf_ticker = yf.Ticker("^SPX")
expirations = yf_ticker.options
m = expirations.__len__()
strikes = np.array(yf_ticker.option_chain(expirations[0]).calls.strike)
dates = np.array(expirations, dtype=np.datetime64)
n = strikes.shape[0]
prices = np.zeros((m, n))
for i in range(m):
    yf_chain = yf_ticker.option_chain(expirations[i])
    s = np.array(yf_chain.calls.strike)
    _, i1, i2 = np.intersect1d(strikes, s, assume_unique=True, return_indices=True)
    p = np.array(yf_chain.calls.ask)
    prices[i, i1] = p[i2]

date_mask = prices[:, 0] >= 0
strike_mask = prices[0, :] >= 0
for i in range(m):
    for j in range(n):
        if prices[i, j] <= 0 and strike_mask[j] and date_mask[i]:
            if np.sum(prices[i, :] > 0) / n > np.sum(prices[:, j] > 0) / m:
                strike_mask[j] = False
            elif dates[i]-dates[0] < 100 or np.sum(prices[i, :] > 0) < 50:
                date_mask[i] = False
            else:
                strike_mask[j] = False

result = prices[date_mask, :]
result = result[:, strike_mask]
result_strike = strikes[strike_mask]
result_dates = dates[date_mask]
s0 = yf_ticker.info["bid"]
print(s0)
np.savez("data/spx.npz", result, result_strike, result_dates, s0, np.datetime64('today'))
