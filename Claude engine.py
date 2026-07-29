import pandas as pd
import numpy as np


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    df['volume'] = df.get('volume', 0.0)

    df['vol_sma_20'] = df['volume'].rolling(window=20, min_periods=1).mean()

    prev_close = df['close'].shift(1)
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum((df['high'] - prev_close).abs(), (df['low'] - prev_close).abs())
    )
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
    # shift(1) avoids using "today's" EMA (which includes today's own close) to judge today's trend
    daily_20_ema = daily_closes.ewm(span=20, adjust=False, min_periods=1).mean().shift(1)
    df['daily_20_ema'] = df['date'].map(daily_20_ema).ffill().bfill()

    return df


def _finalize_r_trade(pos, exit_time, exit_reason, r_multiple, legs):
    """Package a completed trade in R-multiple space (no quantity/rupee amount yet)."""
    return {
        'symbol': pos['symbol'],
        'type': pos['type'],
        'entry_time': pos['entry_time'],
        'entry_price': pos['entry_price'],
        'sl_dist': pos['sl_dist'],
        'exit_time': exit_time,
        'exit_reason': exit_reason,
        'r_multiple': r_multiple,
        'legs': legs,
    }


def _generate_r_multiple_trades(df, risk_per_trade_pct=0.01, min_atr_pct_of_price=0.0005):
    """
    Pass 1: walk each symbol/day and produce trades expressed in R-multiples
    (i.e. "this trade made +1.5x its intended risk" rather than a rupee amount).
    This deliberately decouples signal/exit logic from position sizing, so sizing
    can be computed correctly (and compounding) in a second pass.
    """
    if 'symbol' not in df.columns:
        df['symbol'] = 'UNKNOWN'

    all_trades = []

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
            pending_signal = None  # signal fires this candle, FILLS next candle's open
            last_index = day_df.index[-1]
            has_volume = day_df['volume'].max() > 0

            for row in day_df.iloc[7:].itertuples():
                candle_idx = row.Index

                # --- Fill any pending signal at THIS candle's open (no same-candle lookahead fill) ---
                if pending_signal is not None and active_position is None:
                    sig = pending_signal
                    entry_price = row.open
                    sl_dist = sig['sl_dist']
                    direction = sig['type']
                    if direction == 'LONG':
                        sl_price = entry_price - sl_dist
                        target1_price = entry_price + sl_dist
                        target2_price = entry_price + 2.0 * sl_dist
                    else:
                        sl_price = entry_price + sl_dist
                        target1_price = entry_price - sl_dist
                        target2_price = entry_price - 2.0 * sl_dist

                    active_position = {
                        'symbol': symbol,
                        'type': direction,
                        'entry_time': row.datetime,
                        'entry_price': entry_price,
                        'sl_dist': sl_dist,
                        'sl_price': sl_price,
                        'target1_price': target1_price,
                        'target2_price': target2_price,
                        'partial_taken': False,
                    }
                    pending_signal = None
                    # NOTE: intentionally fall through — this same candle's high/low can still
                    # trigger SL/target immediately after the open-price fill.

                # --- Manage an open position ---
                if active_position is not None:
                    pos = active_position

                    if candle_idx >= 72 or candle_idx == last_index:
                        exit_price = row.close
                        move = (exit_price - pos['entry_price']) if pos['type'] == 'LONG' \
                            else (pos['entry_price'] - exit_price)
                        r_here = move / pos['sl_dist']
                        if not pos['partial_taken']:
                            total_r = r_here
                            legs = 2
                        else:
                            total_r = 0.5 * 1.0 + 0.5 * r_here
                            legs = 3
                        consecutive_losses = consecutive_losses + 1 if total_r <= 0 else 0
                        all_trades.append(_finalize_r_trade(pos, row.datetime, 'Square-off EOD', total_r, legs))
                        active_position = None
                        continue

                    if pos['type'] == 'LONG':
                        if row.low <= pos['sl_price']:
                            total_r = -1.0 if not pos['partial_taken'] else 0.5 * 1.0 + 0.5 * 0.0
                            legs = 2 if not pos['partial_taken'] else 3
                            consecutive_losses = consecutive_losses + 1 if total_r <= 0 else 0
                            all_trades.append(_finalize_r_trade(pos, row.datetime, 'Stop Loss', total_r, legs))
                            active_position = None
                            continue

                        if not pos['partial_taken'] and row.high >= pos['target1_price']:
                            pos['partial_taken'] = True
                            pos['sl_price'] = pos['entry_price']  # move stop to breakeven

                        if pos['partial_taken'] and row.high >= pos['target2_price']:
                            total_r = 0.5 * 1.0 + 0.5 * 2.0
                            consecutive_losses = 0
                            all_trades.append(_finalize_r_trade(pos, row.datetime, 'Target 2 (+2R)', total_r, 3))
                            active_position = None
                            continue

                    elif pos['type'] == 'SHORT':
                        if row.high >= pos['sl_price']:
                            total_r = -1.0 if not pos['partial_taken'] else 0.5 * 1.0 + 0.5 * 0.0
                            legs = 2 if not pos['partial_taken'] else 3
                            consecutive_losses = consecutive_losses + 1 if total_r <= 0 else 0
                            all_trades.append(_finalize_r_trade(pos, row.datetime, 'Stop Loss', total_r, legs))
                            active_position = None
                            continue

                        if not pos['partial_taken'] and row.low <= pos['target1_price']:
                            pos['partial_taken'] = True
                            pos['sl_price'] = pos['entry_price']

                        if pos['partial_taken'] and row.low <= pos['target2_price']:
                            total_r = 0.5 * 1.0 + 0.5 * 2.0
                            consecutive_losses = 0
                            all_trades.append(_finalize_r_trade(pos, row.datetime, 'Target 2 (+2R)', total_r, 3))
                            active_position = None
                            continue

                # --- Look for a new signal (only if flat, and within the entry window) ---
                # Window trimmed to 56 (not 57) so the NEXT candle's fill still lands by 2:00 PM.
                if active_position is None and pending_signal is None and consecutive_losses < 2:
                    if 9 <= candle_idx <= 56:
                        atr = row.atr_14
                        adx = row.adx_14
                        vol_sma = row.vol_sma_20
                        vwap = row.vwap
                        ema20 = row.daily_20_ema
                        close = row.close

                        if pd.isna(atr) or pd.isna(adx) or pd.isna(ema20) or pd.isna(vwap) or close <= 0:
                            continue

                        # BUG FIX: floor out unreliable/near-zero ATR readings instead of only
                        # skipping when it's exactly zero. Prevents absurd position sizing later.
                        if atr < (min_atr_pct_of_price * close):
                            continue

                        sl_dist = 1.3 * atr
                        if sl_dist <= 0:
                            continue

                        # Opening-range width sanity band (part of the original V-ORB 2.0 spec,
                        # was missing from this implementation): skip abnormally narrow or wide ranges.
                        if not (0.5 * atr <= or_width <= 2.5 * atr):
                            continue

                        regime_pass = (adx >= 18.0)
                        if not regime_pass:
                            continue

                        vol_condition = (row.volume > 1.5 * vol_sma) if has_volume else True

                        if (close > or_high) and (close > vwap) and vol_condition and (close > ema20):
                            pending_signal = {'type': 'LONG', 'sl_dist': sl_dist}
                        elif (close < or_low) and (close < vwap) and vol_condition and (close < ema20):
                            pending_signal = {'type': 'SHORT', 'sl_dist': sl_dist}

    return pd.DataFrame(all_trades)


def run_backtest(df: pd.DataFrame, initial_capital=1_000_000.0, risk_per_trade_pct=0.01,
                  max_active_trades=3, cost_per_leg=20.0, max_position_pct_of_equity=0.25):
    """
    Two-pass backtest:
      Pass 1 (_generate_r_multiple_trades): determine WHICH trades happen and their
      outcome in R-multiples, independent of position sizing.
      Pass 2 (below): walk approved trades in chronological order, sizing each one
      off the CURRENT running equity (fixes the non-compounding bug), applying a
      per-trade notional cap (fixes the unbounded-size bug), and deducting a simple
      brokerage/slippage cost per executed leg (fixes the zero-cost bug).
    """
    r_trades = _generate_r_multiple_trades(df, risk_per_trade_pct=risk_per_trade_pct)

    if r_trades.empty:
        return r_trades, pd.DataFrame()

    # Portfolio concurrency filter: at most `max_active_trades` open at once, chronologically.
    r_trades = r_trades.sort_values('entry_time').reset_index(drop=True)
    approved = []
    active_exits = []
    for _, trade in r_trades.iterrows():
        active_exits = [t for t in active_exits if t > trade['entry_time']]
        if len(active_exits) < max_active_trades:
            approved.append(trade)
            active_exits.append(trade['exit_time'])

    if not approved:
        return pd.DataFrame(), pd.DataFrame()

    approved_df = pd.DataFrame(approved).sort_values('entry_time').reset_index(drop=True)

    # --- Pass 2: equity-based (compounding) sizing, applied in entry-time order ---
    running_equity = initial_capital
    final_rows = []

    for _, t in approved_df.iterrows():
        risk_amt = running_equity * risk_per_trade_pct
        sl_dist = t['sl_dist']
        quantity = risk_amt / sl_dist

        # BUG FIX: cap notional exposure so a single trade can't silently eat an
        # outsized share of the account even if sizing math produces a big quantity.
        notional = quantity * t['entry_price']
        max_notional = max_position_pct_of_equity * running_equity
        if notional > max_notional:
            quantity = max_notional / t['entry_price']

        quantity = max(1, int(quantity))
        effective_risk = quantity * sl_dist
        gross_pnl = t['r_multiple'] * effective_risk

        # BUG FIX: real trading has costs. Simple flat cost per executed leg
        # (entry, partial exit if any, final exit).
        cost = t['legs'] * cost_per_leg
        net_pnl = gross_pnl - cost

        running_equity += net_pnl

        final_rows.append({
            'symbol': t['symbol'],
            'type': t['type'],
            'entry_time': t['entry_time'],
            'entry_price': t['entry_price'],
            'exit_time': t['exit_time'],
            'exit_reason': t['exit_reason'],
            'quantity': quantity,
            'r_multiple': t['r_multiple'],
            'pnl': net_pnl,
            'equity_after': running_equity,
        })

    final_trades_df = pd.DataFrame(final_rows).sort_values('exit_time').reset_index(drop=True)

    equity_df = final_trades_df[['exit_time', 'equity_after']].rename(
        columns={'exit_time': 'datetime', 'equity_after': 'equity'}
    )
    start_row = pd.DataFrame([{'datetime': df['datetime'].min(), 'equity': initial_capital}])
    equity_df = pd.concat([start_row, equity_df], ignore_index=True)

    return final_trades_df, equity_df
            
