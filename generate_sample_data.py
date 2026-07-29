import pandas as pd
import numpy as np

def generate_stock_data(symbol="TCS", days=60, start_date="2026-01-01"):
    """
    Generates realistic 5-minute intraday data for testing the backtester.
    """
    date_range = pd.date_range(start=start_date, periods=days, freq='B')
    all_data = []

    np.random.seed(42)
    base_price = 3500.0

    for current_date in date_range:
        # Trading hours: 09:15 to 15:30 (75 five-minute bars per day)
        times = pd.date_range(
            start=f"{current_date.strftime('%Y-%m-%d')} 09:15:00",
            end=f"{current_date.strftime('%Y-%m-%d')} 15:25:00",
            freq='5min'
        )
        
        n_bars = len(times)
        returns = np.random.normal(loc=0.0001, scale=0.002, size=n_bars)
        price_path = base_price * np.exp(np.cumsum(returns))
        
        highs = price_path * (1 + np.abs(np.random.normal(0, 0.001, n_bars)))
        lows = price_path * (1 - np.abs(np.random.normal(0, 0.001, n_bars)))
        opens = price_path * (1 + np.random.normal(0, 0.0005, n_bars))
        closes = price_path
        volumes = np.random.randint(1000, 50000, size=n_bars)

        df_day = pd.DataFrame({
            'datetime': times,
            'symbol': symbol,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        all_data.append(df_day)
        base_price = closes[-1]

    df_total = pd.concat(all_data, ignore_index=True)
    df_total.to_csv("sample_data.csv", index=False)
    print("sample_data.csv successfully generated.")

if __name__ == "__main__":
    generate_stock_data()
