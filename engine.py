import pandas as pd
import numpy as np

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    df['volume'] = df.get('volume', 0.0)

    df['vol_sma_20'] = df['volume'].rolling(window=20, min_periods=1).mean()

    prev_close = df['close'].shift(1)
    tr = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - prev_close).abs(), (df['low'] - prev_close).abs()))
    df['atr_14'] = tr.rolling(window=14, min_periods=1).mean()

    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_smooth = tr.rolling(14, min_periods=1).sum()
    pos_di = 100 * (pd.Series(pos_dm).rolling(14, min_periods=1).sum() / (tr_smooth + 1e-8))
    neg_di = 100 * (pd.Series(neg_dm).rolling(14, min_periods=1).sum() / (tr_smooth + 1e-8))
    dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di + 1e-8)
    df['adx_14'] = dx.rolling(14, min_periods=1).mean()

    tp = (df['high'] + df['low'] + df['close']) / 3.0
    df['tp_vol'] = tp * (df['volume'] + 1e-8)
    df['cum_tp_vol'] = df.groupby('date')['tp_vol'].cumsum()
    df['cum_vol'] = df.groupby('date')['volume'].cumsum() + 1e-8
    df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
    df.drop(columns=['tp_vol', 'cum_tp_vol', 'cum_vol'], inplace=True)

    daily_closes = df.groupby('date')['close'].last()
    daily_20_ema = daily_closes.ewm(span=20, adjust=False, min_periods=1).mean().shift(1)
    df['daily_20_ema'] = df['date'].map(daily_20_ema).ffill().bfill()

    return df

def run_backtest(df: pd.DataFrame, initial_capital=1_000_000.0, risk_per_trade_pct=0.01):
    if 'symbol' not in df.columns:
        df['symbol'] = 'UNKNOWN'
        
    all_trades = []
    
    # CRITICAL FIX: Loop through each stock independently to prevent data mixing!
    for symbol, sym_df in df.groupby('symbol'):
        sym_df = sym_df.sort_values('datetime').reset_index(drop=True)
        sym_df = calculate_indicators(sym_df)
        
        for trade_date, day_df in sym_df.groupby('date'):
            day_df = day_df.sort_values('datetime').reset_index(drop=True)
            
            if len(day_df) < 15:
                continue

            or_df = day_df.iloc[:7]
            or_high = or_df['high'].max()
            or_low = or_df['low'].min()
            or_width = or_high - or_low

            consecutive_losses = 0
            active_position = None
            last_index = day_df.index[-1]
            has_volume = day_df['volume'].max() > 0

            for row in day_df.iloc[7:].itertuples():
                candle_idx = row.Index

                if active_position is not None:
                    pos = active_position
                    
                    if candle_idx >= 72 or candle_idx == last_index:
                        exit_price = row.close
                        pnl = (exit_price - pos['entry_price']) * pos['remaining_qty'] if pos['type'] == 'LONG' else (pos['entry_price'] - exit_price) * pos['remaining_qty']
                        pos['pnl'] += pnl
                        consecutive_losses = consecutive_losses + 1 if pos['pnl'] <= 0 else 0
                        all_trades.append({**pos, 'exit_time': row.datetime, 'exit_reason': 'Square-off EOD'})
                        active_position = None
                        continue

                    if pos['type'] == 'LONG':
                        if row.low <= pos['sl_price']:
                            pnl = (pos['sl_price'] - pos['entry_price']) * pos['remaining_qty']
                            pos['pnl'] += pnl
                            consecutive_losses = consecutive_losses + 1 if pos['pnl'] <= 0 else 0
                            all_trades.append({**pos, 'exit_time': row.datetime, 'exit_reason': 'Stop Loss'})
                            active_position = None
                            continue

                        if not pos['partial_taken'] and row.high >= pos['target1_price']:
                            part_pnl = (pos['target1_price'] - pos['entry_price']) * pos['partial_qty']
                            pos['pnl'] += part_pnl
                            pos['remaining_qty'] -= pos['partial_qty']
                            pos['sl_price'] = pos['entry_price']
                            pos['partial_taken'] = True

                        if pos['partial_taken'] and row.high >= pos['target2_price']:
                            final_pnl = (pos['target2_price'] - pos['entry_price']) * pos['remaining_qty']
                            pos['pnl'] += final_pnl
                            consecutive_losses = 0
                            all_trades.append({**pos, 'exit_time': row.datetime, 'exit_reason': 'Target 2 (+2R)'})
                            active_position = None
                            continue

                    elif pos['type'] == 'SHORT':
                        if row.high >= pos['sl_price']:
                            pnl = (pos['entry_price'] - pos['sl_price']) * pos['remaining_qty']
                            pos['pnl'] += pnl
                            consecutive_losses = consecutive_losses + 1 if pos['pnl'] <= 0 else 0
                            all_trades.append({**pos, 'exit_time': row.datetime, 'exit_reason': 'Stop Loss'})
                            active_position = None
                            continue

                        if not pos['partial_taken'] and row.low <= pos['target1_price']:
                            part_pnl = (pos['entry_price'] - pos['target1_price']) * pos['partial_qty']
                            pos['pnl'] += part_pnl
                            pos['remaining_qty'] -= pos['partial_qty']
                            pos['sl_price'] = pos['entry_price']
                            pos['partial_taken'] = True

                        if pos['partial_taken'] and row.low <= pos['target2_price']:
                            final_pnl = (pos['entry_price'] - pos['target2_price']) * pos['remaining_qty']
                            pos['pnl'] += final_pnl
                            consecutive_losses = 0
                            all_trades.append({**pos, 'exit_time': row.datetime, 'exit_reason': 'Target 2 (+2R)'})
                            active_position = None
                            continue

                if active_position is None and consecutive_losses < 2:
                    if 9 <= candle_idx <= 57:
                        atr = row.atr_14
                        adx = row.adx_14
                        vol_sma = row.vol_sma_20
                        vwap = row.vwap
                        ema20 = row.daily_20_ema

                        if pd.isna(atr) or pd.isna(adx) or pd.isna(ema20) or pd.isna(vwap):
                            continue

                        regime_pass = (adx >= 18.0) and (0.3 * atr <= or_width <= 3.0 * atr)

                        if regime_pass:
                            sl_dist = 1.3 * atr
                            if sl_dist == 0:
                                continue
                                
                            risk_amt = initial_capital * risk_per_trade_pct
                            position_size = max(1, int(risk_amt / sl_dist))
                            
                            vol_condition = (row.volume > 1.5 * vol_sma) if has_volume else True

                            if (row.close > or_high) and (row.close > vwap) and vol_condition and (row.close > ema20):
                                active_position = {
                                    'symbol': symbol,
                                    'type': 'LONG',
                                    'entry_time': row.datetime,
                                    'entry_price': row.close,
                                    'qty': position_size,
                                    'remaining_qty': position_size,
                                    'partial_qty': position_size // 2,
                                    'sl_price': row.close - sl_dist,
                                    'target1_price': row.close + sl_dist,
                                    'target2_price': row.close + (2.0 * sl_dist),
                                    'partial_taken': False,
                                    'pnl': 0.0
                                }
                            elif (row.close < or_low) and (row.close < vwap) and vol_condition and (row.close < ema20):
                                active_position = {
                                    'symbol': symbol,
                                    'type': 'SHORT',
                                    'entry_time': row.datetime,
                                    'entry_price': row.close,
                                    'qty': position_size,
                                    'remaining_qty': position_size,
                                    'partial_qty': position_size // 2,
                                    'sl_price': row.close + sl_dist,
                                    'target1_price': row.close - sl_dist,
                                    'target2_price': row.close - (2.0 * sl_dist),
                                    'partial_taken': False,
                                    'pnl': 0.0
                                }
                                
    trades_df = pd.DataFrame(all_trades)
    
    if trades_df.empty:
        return trades_df, pd.DataFrame()
        
    # Generate Global Portfolio Equity Curve
    trades_df = trades_df.sort_values('exit_time').reset_index(drop=True)
    trades_df['equity'] = initial_capital + trades_df['pnl'].cumsum()
    
    equity_df = trades_df[['exit_time', 'equity']].rename(columns={'exit_time': 'datetime'})
    start_row = pd.DataFrame([{'datetime': df['datetime'].min(), 'equity': initial_capital}])
    equity_df = pd.concat([start_row, equity_df], ignore_index=True)

    return trades_df, equity_df
