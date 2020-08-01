import pandas as pd
import torch
from plotly.subplots import make_subplots
import plotly.graph_objects as go


## TODO: Make subplots for training and testing

## TODO: Combines train and test for features and labels
class FinancialDataBuilder:

    def __init__(self, df_index, data, model_data, tau):
        # print(data[tau:len(train_label) + tau] == train_label)
        # print(data[len(train_label) + 2*tau:] == test_label)
        train_data = model_data[0]
        test_data = model_data[1]

        self.train_df_x = df_index[tau: len(train_data)+tau]
        self.train_df_y = data[tau: len(train_data)+tau]
        self.train_df_y = self.concat_data(a=self.train_df_y, b=train_data)
        
        self.test_df_x = df_index[len(train_data)+2*tau:]
        self.test_df_y = data[len(train_data)+2*tau:]
        self.test_df_y = self.concat_data(a=self.test_df_y, b= test_data)  

    def concat_data(self, a, b):
        return torch.cat((a, b), axis=1)

    def get_axis(self):
        return self.train_df_x, self.train_df_y, self.test_df_x, self.test_df_y

class FinancialDataVisualizer:

    def __init__(self, df_index, data, model_data, tau):
       self.axis = FinancialDataBuilder(df_index=df_index, data=data, model_data=model_data, tau=tau).get_axis()

    def visualize(self, title):
        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(
            go.Scatter(x=self.axis[0], y=self.axis[1].T[0], name='Actual price'),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=self.axis[0], y=self.axis[1].T[1], name='Model train prediction'),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=self.axis[2], y=self.axis[3].T[0], name='Actual price'),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=self.axis[2], y=self.axis[3].T[1], name='Model test prediction'),
            row=2, col=1
        )

        self.beautify_plot(fig, title)
        fig.show()


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