import pandas as pd
import numpy as np

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates 5-min ATR, Volume SMA, ADX, Daily VWAP, and Shifted Daily 20 EMA.
    Uses min_periods and bfill to prevent NaN filter-stacking bugs.
    """
    df = df.copy()
    
    # Ensure datetime format and extract date/time
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    df['time'] = df['datetime'].dt.time

    # 1. Volume SMA (20)
    df['vol_sma_20'] = df['volume'].rolling(window=20, min_periods=1).mean()

    # 2. ATR (14)
    prev_close = df['close'].shift(1)
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            (df['high'] - prev_close).abs(),
            (df['low'] - prev_close).abs()
        )
    )
    df['atr_14'] = tr.rolling(window=14, min_periods=1).mean()

    # 3. ADX (14)
    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']
    
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr_smooth = tr.rolling(14, min_periods=1).sum()
    pos_di = 100 * (pd.Series(pos_dm).rolling(14, min_periods=1).sum() / (tr_smooth + 1e-8))
    neg_di = 100 * (pd.Series(neg_dm).rolling(14, min_periods=1).sum() / (tr_smooth + 1e-8))
    
    dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di + 1e-8)
    df['adx_14'] = dx.rolling(14, min_periods=1).mean()

    # 4. Intraday Daily VWAP
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    df['tp_vol'] = tp * df['volume']
    
    df['cum_tp_vol'] = df.groupby('date')['tp_vol'].cumsum()
    df['cum_vol'] = df.groupby('date')['volume'].cumsum()
    df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
    
    df.drop(columns=['tp_vol', 'cum_tp_vol', 'cum_vol'], inplace=True)

    # 5. Shifted Daily 20 EMA (Updated from 50)
    daily_closes = df.groupby('date')['close'].last()
    daily_20_ema = daily_closes.ewm(span=20, adjust=False).mean().shift(1)
    
    # Map back to 5-min and fill NaNs to prevent dropping first-day trades
    df['daily_20_ema'] = df['date'].map(daily_20_ema).ffill().bfill()

    return df

def run_backtest(df: pd.DataFrame, initial_capital=1_000_000.0, risk_per_trade_pct=0.01):
    """
    Executes V-ORB 2.0 dual-direction backtest logic.
    """
    df = calculate_indicators(df)
    
    trades = []
    equity = initial_capital
    equity_curve = []

    for trade_date, day_df in df.groupby('date'):
        day_df = day_df.sort_values('datetime').reset_index(drop=True)
        
        or_mask = (day_df['datetime'].dt.time >= pd.to_datetime('09:15').time()) & \
                  (day_df['datetime'].dt.time <= pd.to_datetime('09:50').time())
        or_df = day_df[or_mask]

        if or_df.empty:
            continue

        or_high = or_df['high'].max()
        or_low = or_df['low'].min()
        or_width = or_high - or_low

        consecutive_losses = 0
        active_position = None

        for idx, row in day_df.iterrows():
            current_time = row['datetime'].time()
            equity_curve.append({'datetime': row['datetime'], 'equity': equity})

            if active_position is not None:
                pos = active_position
                
                if current_time >= pd.to_datetime('15:15').time():
                    exit_price = row['close']
                    pnl = (exit_price - pos['entry_price']) * pos['remaining_qty'] if pos['type'] == 'LONG' else (pos['entry_price'] - exit_price) * pos['remaining_qty']
                    pos['pnl'] += pnl
                    equity += pnl
                    
                    consecutive_losses = consecutive_losses + 1 if pos['pnl'] <= 0 else 0
                    trades.append({**pos, 'exit_time': row['datetime'], 'exit_reason': 'Square-off 3:15 PM'})
                    active_position = None
                    continue

                if pos['type'] == 'LONG':
                    if row['low'] <= pos['sl_price']:
                        pnl = (pos['sl_price'] - pos['entry_price']) * pos['remaining_qty']
                        pos['pnl'] += pnl
                        equity += pnl
                        consecutive_losses = consecutive_losses + 1 if pos['pnl'] <= 0 else 0
                        trades.append({**pos, 'exit_time': row['datetime'], 'exit_reason': 'Stop Loss'})
                        active_position = None
                        continue

                    if not pos['partial_taken'] and row['high'] >= pos['target1_price']:
                        part_pnl = (pos['target1_price'] - pos['entry_price']) * pos['partial_qty']
                        pos['pnl'] += part_pnl
                        equity += part_pnl
                        pos['remaining_qty'] -= pos['partial_qty']
                        pos['sl_price'] = pos['entry_price'] 
                        pos['partial_taken'] = True

                    if pos['partial_taken'] and row['high'] >= pos['target2_price']:
                        final_pnl = (pos['target2_price'] - pos['entry_price']) * pos['remaining_qty']
                        pos['pnl'] += final_pnl
                        equity += final_pnl
                        consecutive_losses = 0
                        trades.append({**pos, 'exit_time': row['datetime'], 'exit_reason': 'Target 2'})
                        active_position = None
                        continue

                elif pos['type'] == 'SHORT':
                    if row['high'] >= pos['sl_price']:
                        pnl = (pos['entry_price'] - pos['sl_price']) * pos['remaining_qty']
                        pos['pnl'] += pnl
                        equity += pnl
                        consecutive_losses = consecutive_losses + 1 if pos['pnl'] <= 0 else 0
                        trades.append({**pos, 'exit_time': row['datetime'], 'exit_reason': 'Stop Loss'})
                        active_position = None
                        continue

                    if not pos['partial_taken'] and row['low'] <= pos['target1_price']:
                        part_pnl = (pos['entry_price'] - pos['target1_price']) * pos['partial_qty']
                        pos['pnl'] += part_pnl
                        equity += part_pnl
                        pos['remaining_qty'] -= pos['partial_qty']
                        pos['sl_price'] = pos['entry_price']
                        pos['partial_taken'] = True

                    if pos['partial_taken'] and row['low'] <= pos['target2_price']:
                        final_pnl = (pos['entry_price'] - pos['target2_price']) * pos['remaining_qty']
                        pos['pnl'] += final_pnl
                        equity += final_pnl
                        consecutive_losses = 0
                        trades.append({**pos, 'exit_time': row['datetime'], 'exit_reason': 'Target 2'})
                        active_position = None
                        continue

            if active_position is None and consecutive_losses < 2:
                if pd.to_datetime('10:00').time() <= current_time <= pd.to_datetime('14:00').time():
                    atr = row['atr_14']
                    adx = row['adx_14']
                    vol_sma = row['vol_sma_20']
                    vwap = row['vwap']
                    ema20 = row['daily_20_ema']

                    if pd.isna(atr) or pd.isna(adx) or pd.isna(vol_sma) or pd.isna(ema20):
                        continue

                    # Updated Regime Filter
                    regime_pass = (adx >= 18.0) and (0.3 * atr <= or_width <= 3.0 * atr)

                    if regime_pass:
                        sl_dist = 1.3 * atr
                        if sl_dist == 0:
                            continue
                            
                        risk_amt = equity * risk_per_trade_pct
                        position_size = max(1, int(risk_amt / sl_dist))

                        if (row['close'] > or_high) and (row['close'] > vwap) and \
                           (row['volume'] > 1.5 * vol_sma) and (row['close'] > ema20):
                            
                            active_position = {
                                'symbol': row.get('symbol', 'UNKNOWN'),
                                'type': 'LONG',
                                'entry_time': row['datetime'],
                                'entry_price': row['close'],
                                'qty': position_size,
                                'remaining_qty': position_size,
                                'partial_qty': position_size // 2,
                                'sl_price': row['close'] - sl_dist,
                                'target1_price': row['close'] + sl_dist,
                                'target2_price': row['close'] + (2.0 * sl_dist),
                                'partial_taken': False,
                                'pnl': 0.0
                            }

                        elif (row['close'] < or_low) and (row['close'] < vwap) and \
                             (row['volume'] > 1.5 * vol_sma) and (row['close'] < ema20):
                            
                            active_position = {
                                'symbol': row.get('symbol', 'UNKNOWN'),
                                'type': 'SHORT',
                                'entry_time': row['datetime'],
                                'entry_price': row['close'],
                                'qty': position_size,
                                'remaining_qty': position_size,
                                'partial_qty': position_size // 2,
                                'sl_price': row['close'] + sl_dist,
                                'target1_price': row['close'] - sl_dist,
                                'target2_price': row['close'] - (2.0 * sl_dist),
                                'partial_taken': False,
                                'pnl': 0.0
                            }

    return pd.DataFrame(trades), pd.DataFrame(equity_curve)
