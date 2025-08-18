import gym
from gym import spaces
import numpy as np
import pandas as pd

class MultiTimeframeForexEnv(gym.Env):
    """Forex trading environment using multiple timeframe features.

    The environment loads pre-computed features for 15m, 30m, 1h and 4h
    timeframes from parquet files. At each step the agent chooses a trade
    direction and lot size. Trades are held for a single step (15 minutes)
    to approximate a scalping strategy.

    Observation: concatenated feature vector from all timeframes plus the
    current account balance.
    Action: [direction, lot_fraction]
        direction in [-1, 1] -> sell/buy
        lot_fraction in [0, 1] scaled to [0.01, max_lot]
    Reward: profit/loss in account currency after the step.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        data_15m_path: str,
        data_30m_path: str,
        data_1h_path: str,
        data_4h_path: str,
        initial_balance: float = 1000.0,
        leverage: int = 500,
        max_lot: float = 1.0,
    ):
        super().__init__()
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.max_lot = max_lot

        # Load data for all timeframes
        df15 = pd.read_parquet(data_15m_path)
        df30 = pd.read_parquet(data_30m_path)
        df1h = pd.read_parquet(data_1h_path)
        df4h = pd.read_parquet(data_4h_path)

        # Merge on timestamp column
        for df in (df30, df1h, df4h):
            if "time" not in df.columns:
                raise ValueError("Dataframes must contain a 'time' column")
        merged = (
            df15
            .merge(df30, on="time", suffixes=("_15m", "_30m"))
            .merge(df1h, on="time", suffixes=("", "_1h"))
            .merge(df4h, on="time", suffixes=("", "_4h"))
            .sort_values("time")
            .reset_index(drop=True)
        )

        if "close_15m" in merged.columns:
            price_col = "close_15m"
        elif "close" in df15.columns:
            price_col = "close"
        else:
            raise ValueError("15m dataframe must contain a 'close' or 'close_15m' column")
        self.prices = merged[price_col].values

        # Remove non-feature columns
        self.feature_df = merged.drop(columns=["time", price_col])
        self.features = self.feature_df.values.astype(np.float32)

        # Observation: features + balance
        feature_dim = self.features.shape[1]
        obs_low = np.full(feature_dim + 1, -np.inf, dtype=np.float32)
        obs_high = np.full(feature_dim + 1, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action: [direction, lot_fraction]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.step_idx = 0
        return self._get_obs()

    def _get_obs(self):
        obs = np.append(self.features[self.step_idx], self.balance)
        return obs.astype(np.float32)

    def step(self, action):
        direction = np.sign(action[0])  # -1 sell, 1 buy
        lot_fraction = np.clip(action[1], 0.0, 1.0)
        lot_size = 0.01 + (self.max_lot - 0.01) * lot_fraction

        price_now = self.prices[self.step_idx]
        price_next = self.prices[self.step_idx + 1]
        pip_diff = (price_next - price_now) * 10000.0
        profit = direction * pip_diff * lot_size * 10.0  # 1 lot -> $10 per pip

        self.balance += profit
        self.step_idx += 1
        done = self.step_idx >= len(self.prices) - 1 or self.balance <= 0
        reward = profit
        obs = self._get_obs()
        info = {"balance": self.balance, "profit": profit, "pips": pip_diff}
        return obs, reward, done, info

    def render(self, mode="human"):
        print(f"Step: {self.step_idx} Balance: {self.balance:.2f}")
