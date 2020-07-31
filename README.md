# Forecasting Financial Time Series using Neural Networks
Using Neural Networks for Financial Time Series Forecasting

Financial Time Series Forecasting is divided into four stages which starts with data_preprocessing by preprocessing the data following by using the model to train and test the data which follows by the visualization. 



# Installation

Just cd into the root of this directory and execute 

```
pip3/pip install -r requirements.txt
```

This install pytorch along the required libraries to run the program

# How to run the app 

In the main directory, execute the program with: `python3 main.py --ticker (INSERT TICKER)` 

# Data 

The data is queried real time from [yahoo finance](https://finance.yahoo.com/) using DataReader and is not stored locally to improve storage cost.