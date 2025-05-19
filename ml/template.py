import yfinance as yf
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest
from alpaca.trading.enums import OrderSide, TimeInForce

trading_client = TradingClient('api-key', 'secret-key')

# Get our account information.
account = trading_client.get_account()



# Download 1-minute SOXL data for past 7 days
data = yf.download("SOXL", period="7d", interval="1m")
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()


state          = "INIT"
entry_price    = None
stop_loss      = None
take_profit    = None
baseline_low   = None
baseline_high  = None
prices         = []         # will hold the last N prices
WINDOW         = 5          # how many bars to look for “sticking”
TOL            = 0.002      # 0.2% tolerance to call it a stable baseline

def detect_stabilized(prices, window, tol):
    recent = prices[-window:]
    if len(recent) < window:
        return None
    mn, mx = min(recent), max(recent)
    if (mx - mn) / mn < tol:
        return mn, mx
    return None

while True:
    raw = get_latest_price()                     # pull your 1-min bars
    df  = create_features(raw.copy())            
    price = df['Close'].iloc[-1]
    prices.append(price)
    if len(prices) > WINDOW:                     # keep only last WINDOW bars
        prices.pop(0)

    # ─────────────────────────────────────────
    # INIT: always try to find a fresh baseline
    if state == "INIT":
        result = detect_stabilized(prices, WINDOW, TOL)
        if result:
            # we found a new stable micro-range → update baseline
            baseline_low, baseline_high = result
            print(f"New baseline detected: {baseline_low:.2f}–{baseline_high:.2f}")
            # immediately go scalp that range
            entry_price = baseline_low
            take_profit = baseline_low + 0.02
            # place your BUY order here (limit or market)
            print(f"BUY @ {entry_price:.2f}, TP @ {take_profit:.2f}")
            state = "SCALP_DOWN"  
        else:
            # No stable range yet—stay in INIT, keep updating prices
            pass

    # ─────────────────────────────────────────
    # SCALP_DOWN: we’re long at the bottom of a baseline
    elif state == "SCALP_DOWN":
        # if price dips below baseline, abandon and re-INIT
        if price < baseline_low:
            print("Range broken lower → reset")
            state = "INIT"
        # if price hits TP, take profit and re-INIT
        elif price >= take_profit:
            print(f"TP hit @ {price:.2f} → SELL")
            # place your SELL order here…
            state = "INIT"

    # … you could also re-implement your UP_OPEN logic here if you want that branch …
    # elif state == "UP_OPEN":
    #     …

    # ─────────────────────────────────────────
    # Sleep until next minute
    time.sleep(60)


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
