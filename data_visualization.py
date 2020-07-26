import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

class FinancialDataBuilder:

    def __init__(self, training_data, testing_data, dataframe):
        self.tensor_data = self.concat_data(training_data, testing_data)
        self.build_dataframe(dataframe)
        self.add_column(column_name='Predicted Close', column_data=self.tensor_data.numpy())
        
    def concat_data(self, training_data, testing_data):
        return torch.cat((training_data, testing_data), dim=0)

    def build_dataframe(self, dataframe):
        self.final_dataframe = dataframe

    def add_column(self, column_name, column_data):
        self.final_dataframe[column_name] = column_data

    def get_dataframe(self):
        return self.final_dataframe

        
class FinancialDataVisualizer:
    
    def __init__(self, training_data, testing_data, FData_object):
        self.df = FinancialDataBuilder(training_data, testing_data, FData_object.get_dataset()).get_dataframe()

    def visualize(self):
        self.df.plot()
        plt.show()
