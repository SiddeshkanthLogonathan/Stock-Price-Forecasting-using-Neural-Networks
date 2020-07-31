import pandas as pd
import torch
from plotly.subplots import make_subplots
import plotly.graph_objects as go


## TODO: Make subplots for training and testing

## TODO: Combines train and test for features and labels
class FinancialDataBuilder:

    def __init__(self, initial_data, model_data, tau):

        train_data = model_data[0]
        test_data = model_data[1]

        self.train_df = initial_data.iloc[tau: len(train_data)]
        self.add_column(dataframe=self.train_df, column_name='trained data', column_data=train_data)
        self.test_df = initial_data.iloc[len(train_data)+1+tau:]
        self.add_column(dataframe=self.test_df, column_name='test data', column_data=test_data)


    def concat_data(self, training_data, testing_data):
        return torch.cat((training_data, testing_data), dim=0)

    def add_column(self, dataframe, column_name, column_data):
        dataframe[column_name] = column_data

    def get_dataframes(self):
        return self.train_df, self.test_df

## TODO: on hardcoding which dimension to be plotted
class FinancialDataVisualizer:

    def __init__(self, initial_data, model_data, tau):
       self.train_df, self.test_df = FinancialDataBuilder(initial_data, model_data, tau).get_dataframes()

    def visualize(self, title):
        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(
            go.Scatter(x=self.train_df.index, y=self.train_df.iloc[:, 0], name='Actual price'),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=self.train_df.index, y=self.train_df.iloc[:, 1], name='Model trained price'),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=self.test_df.index, y=self.test_df.iloc[:, 0], name='Actual price'),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=self.test_df.index, y=self.test_df.iloc[:, 1], name='Model tested price'),
            row=2, col=1
        )

        self.beautify_plot(fig, title)
        fig.show()

    def add_plot(self, fig, x, y, name):
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=name))

    def beautify_plot(self, fig, title):
        fig.update_layout(
            title=title + ' Pricing', 
            xaxis_title='Date', 
            yaxis_title='Price ($)', 
            legend_title='*',
            font=dict(
                family="Courier New, Monospace",
                size=18,
                color="slategrey"
            ))