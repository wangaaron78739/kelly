from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
from itertools import product

import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt


@dataclass
class Backtester(ABC):
    csv_path: str                       # path for rounds data
    bull_win_rate: float = 0.50         # bull win rate
    min_prize_pool: float = 3.0         # min prize pool size allowed
    min_kelly: float = 0.01             # min kelly fraction to ensure positive bet size
    gas_price: int = 6                  # default gas price
    gas: int = 200000                   # default gas quantity
    kelly_multiplier: float = 0.5       # 0.5 = using half-kelly
    fees: bool = True                   # True = enable fees
    
    _status: int = 0                    # runtime status

    def __post_init__(self) -> None:
        self.df = pd.read_csv(self.csv_path)
        self.df_size = len(self.df)
        self._preprocess_data()
    
    def _check_df_size(self) -> None:
        if len(self.df) < 10:
            raise Exception("Rounds data is inadequate!")

    def _preprocess_data(self) -> None:
        """data preprocessing"""
        columns_in_gwei = ['totalAmount', 'bullAmount', 'bearAmount', 'rewardBaseCalAmount', 'rewardAmount']
        self.df[columns_in_gwei] = self.df[columns_in_gwei].apply(lambda a : a / 1e18)  # gwei to ether
        self.df = self.df[self.df['rewardAmount'] > self.min_prize_pool]
        self.df = self.df[self.df['oracleCalled']]
        self._check_df_size()

    # implement this abstract method to model bull win rate
    # we should do some work to create a bull_wr column in df
    # i.e. bull_wr should be a series have the same length of df
    @abstractmethod
    def _predict(self) -> None:
        pass

    def _run(self) -> None:
        if not 'bull_wr' in self.df.columns:
            raise Exception("_predict is not correctly implemented!")

        self.df['bull_odds'] = self.df['rewardAmount']/self.df['bullAmount']
        self.df['bear_odds'] = self.df['rewardAmount']/self.df['bearAmount']

        self.df['bull_kelly'] = (self.df['bull_wr']*self.df['bull_odds']-1)/(self.df['bull_odds']-1)
        self.df['bear_kelly'] = ((1-self.df['bull_wr'])*self.df['bear_odds']-1)/(self.df['bear_odds']-1)

        self.df['bet'] = np.where(self.df['bull_kelly']>self.df['bear_kelly'], 'bull', 'bear')
        self.df['result'] = np.where(self.df['closePrice']>self.df['lockPrice'], 'bull', 'bear')
        self.df.loc[self.df['lockPrice']==self.df['closePrice'], 'result'] = 'tie'  # tie results
        self.df['net_odds'] = np.where(self.df['bet']=='bull', self.df["bull_odds"], self.df["bear_odds"])-1
        self.df['kelly'] = self.df[["bull_kelly", "bear_kelly"]].max(axis=1)

        self.df = self.df[self.df['kelly']>self.min_kelly]
        self._check_df_size()

        pnl = []
        capital = []
        for i in range(len(self.df)):
            prev_capital = 1 if i == 0 else capital[i-1]                                                          # get capital available for this round
            prev_capital -= self.gas*self.gas_price/1e9/2 if self.fees == True else 0                             # minus fees if enabled
            bet_size = prev_capital*self.df['kelly'].values[i]*self.kelly_multiplier                              # optimal bet size by kelly criterion
            ret = self.df['net_odds'].values[i] if self.df['bet'].values[i]==self.df['result'].values[i] else -1  # return of this round
            pnl.append(ret*bet_size)
            capital.append(prev_capital+pnl[i])
        self.df = self.df.assign(pnl=pnl, capital=capital)
        self.df = self.df[self.df['capital']>0]
        self._status = 1

    def run(self) -> Backtester:
        try:
            if self._status == 1:
                self.__post_init__()
            self._predict()
            if not (0 <= self.df['bull_wr'].max() <= 1):
                raise Exception("Extreme win rate is observed")
            self._run()
        except Exception as e:
            self._status = -1
            print(e.message)
        finally:
            return self

    def plot(self) -> Backtester:
        if self._status == 0:
            print("Please run before plotting")
        if self._status == 1:
            sns.lineplot(data=self.df, x="epoch", y="capital")
            plt.show()
        return self

    def analytics(self) -> Backtester:
        if self._status == 0:
            print("Please run before analytics")
        if self._status == 1:
            win_rate = len(self.df[self.df['bet']==self.df['result']])/len(self.df['bet'])
            pct_change = self.df['capital'].pct_change()
            t_test = stats.ttest_1samp(self.df['capital'].pct_change().dropna(), 0)

            print(f"Betted {len(self.df)} games over {self.df_size} games:\n")
            print(f"Start Epoch:\t{self.df['epoch'].iloc[0]}")
            print(f"End Epoch:\t{self.df['epoch'].iloc[-1]}\n")
            print(f"Bet Rate:\t{len(self.df)/self.df_size:.0%}")
            print(f"Win Rate:\t{win_rate:.0%}")
            print(f"Return:\t\t{self.df['capital'].iloc[-1]-1:.0%}")
            print(f"Mean Return:\t{pct_change.mean():.2%}")
            print(f"Volatility:\t{pct_change.std():.2%}")
            print(f"T-Test:\t\t{t_test[0]:.2f}")
            print(f"P-Value:\t{t_test[1]:.2f}")
            print(f"MDD:\t\t{self._get_max_dd(self.df['capital']):.2%}\n")
        return self
    
    def save(self, output_dir='./results', output_filename='result.csv', index=False) -> Backtester:
        if self._status == 0:
            print("Please run before saving")
        if self._status == 1:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self.df.to_csv(output_dir/output_filename, index=index)
        return self

    @staticmethod
    def _get_max_dd(nvs: pd.Series, window=None) -> float:
        """
        :param nvs: net value series
        :param window: lookback window, int or None
        if None, look back entire history
        """
        n = len(nvs)
        if window is None:
            window = n
        # rolling peak values
        peak_series = nvs.rolling(window=window, min_periods=1).max()
        return (nvs/peak_series-1.0).min()

    @staticmethod
    def combinations(params: dict) -> list:
        params_names = list(params.keys())
        params_values = list(params.values())
        dot_product = list(map(list, product(*params_values)))
        return [dict(zip(params_names, v)) for v in dot_product]

