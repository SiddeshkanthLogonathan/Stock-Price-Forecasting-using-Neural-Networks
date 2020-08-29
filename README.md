# Forecasting Financial Time Series using Neural Networks
Using Neural Networks for Financial Time Series Forecasting

---

Financial Time Series Forecasting Neural Network Model is built mainly on the Hypothesis that price on day `x` is affected by the the number of previous days `tau`. For example: when `tau = 4`, the price of the Day 5 is affected by the last 4 days. 

---

# Installation

Just navigate into the root of this directory and execute 

```
pip3/pip install -r requirements.txt
```

This installs pytorch along the required libraries to run the program.

# How to run the app 

In the main directory, execute the program with: `python(3) main.py --ticker (INSERT TICKER)` 

# Data 

The data is queried real time from [yahoo finance](https://finance.yahoo.com/) using yfinance and is not stored locally.