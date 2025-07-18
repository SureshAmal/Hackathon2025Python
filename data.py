import pandas as pd
import numpy as np
import os


def calculate_dema(series, span):
    ema = series.ewm(span=span, adjust=False).mean()
    dema = 2 * ema - ema.ewm(span=span, adjust=False).mean()
    return dema


def calculate_sma(series, window):
    return series.rolling(window=window).mean()


def backtest_dema_strategy(file_path, capital=100000):
    df = pd.read_csv(file_path)
    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df["Date"] = df["Date"].str.split(" ").str[0]
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df.set_index("Date", inplace=True)
    df = df.dropna()

    df["DEMA_20"] = calculate_sma(df["Close"], 20)
    df["DEMA_30"] = calculate_sma(df["Close"], 30)

    df["Signal"] = "Hold"
    df.loc[
        (df["DEMA_20"] > df["DEMA_30"])
        & (df["DEMA_20"].shift(1) <= df["DEMA_30"].shift(1)),
        "Signal",
    ] = "Buy"
    df.loc[
        (df["DEMA_20"] < df["DEMA_30"])
        & (df["DEMA_20"].shift(1) >= df["DEMA_30"].shift(1)),
        "Signal",
    ] = "Sell"

    position = None
    trades = []
    buy_price = 0
    shares = 0

    for date, row in df.iterrows():
        if row["Signal"] == "Buy" and position is None:
            buy_price = row["Close"]
            shares = int(capital / buy_price)
            position = "Long"
            trades.append(
                {"Date": date, "Type": "Buy", "Price": buy_price, "Shares": shares}
            )

        elif row["Signal"] == "Sell" and position == "Long":
            sell_price = row["Close"]
            profit_per_share = sell_price - buy_price
            total_profit = profit_per_share * shares
            trades.append(
                {
                    "Date": date,
                    "Type": "Sell",
                    "Price": sell_price,
                    "Shares": shares,
                    "Profit_per_share": profit_per_share,
                    "Total_Profit": total_profit,
                }
            )
            position = None

    trades_df = pd.DataFrame(trades)
    sell_trades = trades_df[trades_df["Type"] == "Sell"]
    total_profit = sell_trades["Total_Profit"].sum()
    win_trades = sell_trades[sell_trades["Total_Profit"] > 0]
    loss_trades = sell_trades[sell_trades["Total_Profit"] <= 0]
    if not sell_trades.empty:
        total_win = win_trades["Total_Profit"].sum()
        total_loss = abs(loss_trades["Total_Profit"].sum())
        win_rate = (
            (total_win / (total_win + total_loss)) * 100
            if (total_win + total_loss) > 0
            else 0
        )
    else:
        win_rate = 0

    return {
        "Stock": os.path.basename(file_path).replace(".csv", ""),
        "Total Trades": len(sell_trades),
        "Total Profit": total_profit,
        "Win Rate (%)": win_rate,
        "Trade Log": trades_df,
    }


folder_path = "./Data"
summary = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        result = backtest_dema_strategy(file_path)
        summary.append(
            {
                "Stock": result["Stock"],
                "Total Trades": result["Total Trades"],
                "Total Profit": result["Total Profit"],
                "Win Rate (%)": result["Win Rate (%)"],
            }
        )
        result["Trade Log"].to_csv(
            f"./Results/{result['Stock']}_trades.csv", index=False
        )


summary_df = pd.DataFrame(summary)
summary_df = summary_df.sort_values(by="Total Profit", ascending=False)

print("\n📊 Summary Backtest Results (All Stocks):")
print(summary_df)

summary_df.to_csv("./Results/summary_backtest.csv", index=False)
