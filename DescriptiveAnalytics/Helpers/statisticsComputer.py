import numpy as np
import pandas as pd


class StatisticsComputer:

    def __init__(self, df):

        self.df = df
        self.total_days_value = self.total_days()
        self.ask_mean_value = self.ask_mean()
        self.bid_mean_value = self.bid_mean()

    def total_days(self):

        """
        Compute total days (number of rows in the DataFrame).
        """

        return len(self.df)

    def ask_mean(self):

        """
        Compute the mean of the 'Ask' column.
        """

        return np.mean(self.df['Ask'])

    def bid_mean(self):
        """
        Compute the mean of the 'Bid' column.
        """

        return np.mean(self.df['Bid'])

    def get_statistics(self):
        """
        Return a summary of computed statistics.
        """

        return {
            'Total Days': self.total_days_value,
            'Ask Mean': self.ask_mean_value,
            'Bid Mean': self.bid_mean_value
        }
