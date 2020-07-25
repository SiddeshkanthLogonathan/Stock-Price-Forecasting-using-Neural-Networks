import plotly.express as px
import pandas as pd
import torch

class FinancialDataBuilder:

    def __init__(self, trained_data, tested_data):
        self.concatenated_data = self.concat_data(trained_data, tested_data)
        self.builed_data = self.convert_to_dataframe()

    def concat_data(self, feature, label):
        return torch.cat((feature, label), dim=0)
    
## TODO: FinancialDataBuilder has to have the same structure to FinancialDataLoader, so think of
## finding a way to get its columns and index which u can re-use

## TODO: complete FinancialDataVisualizer under the asumption that I have a new pandas DataFrame as an input, creating
## is done in Builder