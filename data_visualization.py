import pandas as pd
import torch
import plotly.graph_objects as go

class FinancialDataBuilder:

    def __init__(self, training_data, dataframe):
        # self.tensor_data = self.concat_data(training_data, testing_data)
        self.build_dataframe(dataframe)
        self.add_column(column_name='Predicted Close', column_data=training_data)

    def concat_data(self, training_data, testing_data):
        return torch.cat((training_data, testing_data), dim=0)

    def build_dataframe(self, dataframe):
        self.final_dataframe = dataframe

    def add_column(self, column_name, column_data):
        self.final_dataframe[column_name] = column_data

    def get_dataframe(self):
        return self.final_dataframe


class FinancialDataVisualizer:

    def __init__(self, training_data, FData_object):
        self.df = FinancialDataBuilder(training_data, FData_object.get_dataset()).get_dataframe()

    def visualize(self, title):
        X_AXIS = self.df.index
        column_names = list(self.df.columns)
        fig = go.Figure()
        self.add_plot(fig, x=X_AXIS, y=self.df[column_names[0]], name=column_names[0])
        self.add_plot(fig, x=X_AXIS, y=self.df[column_names[1]], name=column_names[1])
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