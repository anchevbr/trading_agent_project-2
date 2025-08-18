import numpy as np
import matplotlib.pyplot as plt


def plot_equity_curve(equity):
    equity = np.asarray(equity)
    peaks = np.maximum.accumulate(equity)
    drawdown = (equity - peaks) / peaks

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(equity)
    ax1.set_title("Equity Curve")
    ax1.set_ylabel("Equity")

    ax2.plot(drawdown)
    ax2.set_title("Drawdown")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Step")

    plt.tight_layout()
    plt.show()


def main(path: str = "equity.npy"):
    equity = np.load(path)
    plot_equity_curve(equity)


if __name__ == "__main__":
    main()
