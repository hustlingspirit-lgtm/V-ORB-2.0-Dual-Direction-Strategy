import pandas as pd
import numpy as np

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    df['time'] = df['datetime'].dt.time

    df['vol_sma_20'] = df['volume'].rolling(window=20, min_periods=1).mean()

    prev_close = df['close'].shift(1)
    tr = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - prev_close).abs(), (df['low'] - prev_close).abs()))
    df['atr_14'] = tr.rolling(window=14, min_periods=1).mean()

    tp = (df['high'] + df['low'] + df['close']) / 3.0
    df['tp_vol'] = tp * df['volume']
    df['cum_tp_vol'] = df.groupby('date')['tp_vol'].cumsum()
    df['cum_vol'] = df.groupby('date')['volume'].cumsum()
    df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
    df.drop(columns=['tp_vol', 'cum_tp_vol', 'cum_vol'], inplace=True)
    
    return df

def run_backtest(df: pd.DataFrame, initial_capital=1_000_000.0, risk_per_trade_pct=0.01):
    df = calculate_indicators(df)
    trades = []
    equity = initial_capital
    equity_curve = []

    for trade_date, day_df in df.groupby('date'):
        day_df = day_df.sort_values('datetime').reset_index(drop=True)
        
        or_mask = (day_df['time'] >= pd.to_datetime('09:15').time()) & (day_df['time'] <= pd.to_datetime('09:50').time())
        or_df = day_df[or_mask]

        if or_df.empty:
            continue

        or_high = or_df['high'].max()
        or_low = or_df['low'].min()
        active_position = None

        for row in day_df.itertuples(index=False):
            current_time = row.time
            equity_curve.append({'datetime': row.datetime, 'equity': equity})

            if active_position is not None:
                pos = active_position
                
                if current_time >= pd.to_datetime('15:15').time():
                    exit_price = row.close
                    pnl = (exit_price - pos['entry_price']) * pos['qty'] if pos['type'] == 'LONG' else (pos['entry_price'] - exit_price) * pos['qty']
                    equity += pnl
                    trades.append({**pos, 'exit_time': row.datetime, 'exit_reason': 'Square-off', 'pnl': pnl})
                    active_position = None
                    continue

                if pos['type'] == 'LONG':
                    if row.low <= pos['sl_price']:
                        pnl = (pos['sl_price'] - pos['entry_price']) * pos['qty']
                        equity += pnl
                        trades.append({**pos, 'exit_time': row.datetime, 'exit_reason': 'Stop Loss', 'pnl': pnl})
                        active_position = None
                    elif row.high >= pos['target_price']:
                        pnl = (pos['target_price'] - pos['entry_price']) * pos['qty']
                        equity += pnl
                        trades.append({**pos, 'exit_time': row.datetime, 'exit_reason': 'Target', 'pnl': pnl})
                        active_position = None
                        
                elif pos['type'] == 'SHORT':
                    if row.high >= pos['sl_price']:
                        pnl = (pos['entry_price'] - pos['sl_price']) * pos['qty']
                        equity += pnl
                        trades.append({**pos, 'exit_time': row.datetime, 'exit_reason': 'Stop Loss', 'pnl': pnl})
                        active_position = None
                    elif row.low <= pos['target_price']:
                        pnl = (pos['entry_price'] - pos['target_price']) * pos['qty']
                        equity += pnl
                        trades.append({**pos, 'exit_time': row.datetime, 'exit_reason': 'Target', 'pnl': pnl})
                        active_position = None

            if active_position is None:
                if pd.to_datetime('10:00').time() <= current_time <= pd.to_datetime('14:00').time():
                    atr = row.atr_14
                    vol_sma = row.vol_sma_20
                    vwap = row.vwap

                    if pd.isna(atr) or pd.isna(vol_sma) or pd.isna(vwap):
                        continue

                    sl_dist = 1.3 * atr if atr > 0 else 1.0
                    risk_amt = equity * risk_per_trade_pct
                    position_size = max(1, int(risk_amt / sl_dist))

                    # Core V1.0 Logic
                    if (row.close > or_high) and (row.close > vwap) and (row.volume > 1.5 * vol_sma):
                        active_position = {
                            'symbol': getattr(row, 'symbol', 'UNKNOWN'),
                            'type': 'LONG',
                            'entry_time': row.datetime,
                            'entry_price': row.close,
                            'qty': position_size,
                            'sl_price': row.close - sl_dist,
                            'target_price': row.close + (2.0 * sl_dist)
                        }
                    elif (row.close < or_low) and (row.close < vwap) and (row.volume > 1.5 * vol_sma):
                        active_position = {
                            'symbol': getattr(row, 'symbol', 'UNKNOWN'),
                            'type': 'SHORT',
                            'entry_time': row.datetime,
                            'entry_price': row.close,
                            'qty': position_size,
                            'sl_price': row.close + sl_dist,
                            'target_price': row.close - (2.0 * sl_dist)
                        }

    return pd.DataFrame(trades), pd.DataFrame(equity_curve)
    
