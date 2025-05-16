import yfinance as yf
import pandas as pd

# Download 1-minute SOXL data for past 7 days
data = yf.download("SOXL", period="7d", interval="1m")
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# Create 10-minute rolling range
data['RollingHigh'] = data['High'].rolling(window=10).max()
data['RollingLow'] = data['Low'].rolling(window=10).min()

# Simple Buy/Sell Signals
data['BuySignal'] = data['Close'] <= data['RollingLow'] + 0.05
data['SellSignal'] = data['Close'] >= data['RollingHigh'] - 0.05

# Simulated trade logic
capital = 10000
position = 0
entry_price = 0
trades = []

for i in range(10, len(data)):
    price = data['Close'].iloc[i]
    time = data.index[i]

    if position == 0 and data['BuySignal'].iloc[i]:
        position = capital / price
        entry_price = price
        trades.append(('BUY', time, price))

    elif position > 0:
        if data['SellSignal'].iloc[i] or price >= entry_price * 1.01:
            capital = position * price
            trades.append(('SELL', time, price))
            position = 0

# Final results
final = capital if position == 0 else position * data['Close'].iloc[-1]
print(f"Final Capital: ${final:.2f}")
print(f"Total Trades: {len(trades) // 2}")

# Save trades
trades_df = pd.DataFrame(trades, columns=['Action', 'Timestamp', 'Price'])
trades_df.to_csv("soxl_trades.csv", index=False)
