import argparse
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from forex_sac_env import MultiTimeframeForexEnv

DATA15 = "EURUSD_15m_with_features.parquet"
DATA30 = "EURUSD_30m_with_features.parquet"
DATA1H = "EURUSD_1h_with_features.parquet"
DATA4H = "EURUSD_4h_with_features.parquet"


def evaluate(model, env, n_steps=1000):
    obs = env.reset()
    balances = [env.balance]
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        balances.append(env.balance)
        if done:
            break
    profit = env.balance - env.initial_balance
    return profit, np.array(balances)


def train_until_profitable(args):
    attempt = 0
    profit = -np.inf
    best_model = None
    best_balances = None
    lr = args.learning_rate

    while profit <= 0 and attempt < args.max_attempts:
        env = DummyVecEnv([lambda: MultiTimeframeForexEnv(DATA15, DATA30, DATA1H, DATA4H)])
        model = SAC("MlpPolicy", env, learning_rate=lr, verbose=0)
        model.learn(total_timesteps=args.timesteps)

        eval_env = MultiTimeframeForexEnv(DATA15, DATA30, DATA1H, DATA4H)
        profit, balances = evaluate(model, eval_env, n_steps=args.eval_steps)
        print(f"Attempt {attempt+1} profit: {profit:.2f}")
        if profit > 0:
            best_model = model
            best_balances = balances
            break
        lr *= 0.5
        attempt += 1

    if best_model is None:
        best_model = model
        best_balances = balances

    best_model.save(args.model_path)
    np.save(args.equity_path, best_balances)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--eval-steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--model-path", type=str, default="sac_scalper")
    parser.add_argument("--equity-path", type=str, default="equity.npy")
    args = parser.parse_args()

    train_until_profitable(args)


if __name__ == "__main__":
    main()
