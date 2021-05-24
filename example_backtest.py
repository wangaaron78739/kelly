from dataclasses import dataclass

import numpy as np

from backtester import Backtester

# always implement the base class to create the model
@dataclass
class StrategyA(Backtester):
    def _predict(self):
        """assume bull win rate to be default (50%) every round"""
        bull_wr = self.bull_win_rate
        self.df = self.df.assign(bull_wr=bull_wr)

@dataclass
class StrategyB(Backtester):
    p_adjustment: float = 0.1

    def _predict(self):
        """increase bull win rate when bull bet amount > bear bet amount, vice versa"""
        bull_wr = self.bull_win_rate+np.where(self.df['bullAmount']>self.df['bearAmount'], 1, -1)*self.p_adjustment
        self.df = self.df.assign(bull_wr=bull_wr)


if __name__ == '__main__':
    # backtest -> get statistics -> plot -> save
    bt = StrategyA(csv_path="epoches2k.csv")
    bt.run().analytics().plot().save()

    # backtest -> get statistics in a for-loop
    bt = StrategyB(csv_path="epoches2k.csv")
    params = {"p_adjustment": [0.05, 0.07, 0.1]}
    for combo in Backtester.combinations(params):
        vars(bt).update(combo)
        bt.run().analytics()
        print(f"Params:\t\t{combo}\n")