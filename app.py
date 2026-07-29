import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import zipfile
import io
from engine import run_backtest

st.set_page_config(page_title="V-ORB 2.0 Backtester", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .metric-card {
        background-color: #1a1c23;
        border: 1px solid #2d313e;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .metric-title { font-size: 14px; color: #8e95a5; font-weight: 500; }
    .metric-value { font-size: 24px; color: #ffffff; font-weight: 700; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ V-ORB Strategy: Advanced X-Ray Dashboard")

# Zip File Uploader
uploaded_zip = st.file_uploader("Upload Historical Data (.zip containing CSVs)", type=["zip"])

@st.cache_data
def process_uploaded_zip(uploaded_file):
    dfs = []
    with zipfile.ZipFile(uploaded_file) as z:
        for filename in z.namelist():
            if filename.endswith('.csv'):
                with z.open(filename) as f:
                    df = pd.read_csv(f)
                    dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

if uploaded_zip is None:
    st.info("Please upload a .zip file containing your 5-minute historical CSV data to view results.")
else:
    df_raw = process_uploaded_zip(uploaded_zip)
    
    if df_raw.empty:
        st.error("No valid CSV files found in the uploaded .zip archive.")
    else:
        st.success("Data successfully loaded. Running backtest...")
        
        # Run Engine
        trades_df, equity_df = run_backtest(df_raw)

        if trades_df.empty:
            st.warning("No trades generated with the current dataset filters.")
        else:
            # Calculations
            net_profit = trades_df['pnl'].sum()
            total_trades = len(trades_df)
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] <= 0]
            
            win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
            gross_profit = winning_trades['pnl'].sum()
            gross_loss = abs(losing_trades['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan
            expectancy = net_profit / total_trades if total_trades > 0 else 0

            # Drawdown
            equity_df['peak'] = equity_df['equity'].cummax()
            equity_df['drawdown_pct'] = ((equity_df['equity'] - equity_df['peak']) / equity_df['peak']) * 100
            max_drawdown = equity_df['drawdown_pct'].min()

            # Streaks
            pnl_series = (trades_df['pnl'] > 0).astype(int).tolist()
            max_win_streak = max_loss_streak = cur_win = cur_loss = 0
            for val in pnl_series:
                if val == 1:
                    cur_win += 1
                    cur_loss = 0
                    max_win_streak = max(max_win_streak, cur_win)
                else:
                    cur_loss += 1
                    cur_win = 0
                    max_loss_streak = max(max_loss_streak, cur_loss)

            # UI Metrics Display
            st.subheader("Core Performance")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Net Profit (₹)</div><div class="metric-value">₹{net_profit:,.2f}</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-card"><div class="metric-title">Expectancy (₹)</div><div class="metric-value">₹{expectancy:,.2f}</div></div>', unsafe_allow_html=True)

            with col2:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Win Rate</div><div class="metric-value">{win_rate:.1f}%</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-card"><div class="metric-title">Total Trades</div><div class="metric-value">{total_trades}</div></div>', unsafe_allow_html=True)

            with col3:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Profit Factor</div><div class="metric-value">{profit_factor:.2f}</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-card"><div class="metric-title">Max Win Streak</div><div class="metric-value">{max_win_streak}</div></div>', unsafe_allow_html=True)

            with col4:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Max Drawdown</div><div class="metric-value">{max_drawdown:.2f}%</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-card"><div class="metric-title">Max Loss Streak</div><div class="metric-value">{max_loss_streak}</div></div>', unsafe_allow_html=True)

            st.markdown("---")

            # Dynamic Equity Curve Plot
            st.subheader("Dynamic Equity Curve")
            fig_eq = px.line(equity_df, x='datetime', y='equity', title="Capital Growth Over Time")
            fig_eq.update_traces(line_color="#00e676", line_width=2)
            fig_eq.update_layout(template="plotly_dark", height=400, xaxis_title="Date", yaxis_title="Equity")
            st.plotly_chart(fig_eq, use_container_width=True)

            # Underwater Drawdown Chart
            st.subheader("Underwater Chart (Drawdown Depth)")
            fig_dd = px.area(equity_df, x='datetime', y='drawdown_pct', title="Drawdown (%)")
            fig_dd.update_traces(fillcolor="rgba(255, 82, 82, 0.3)", line_color="#ff5252")
            fig_dd.update_layout(template="plotly_dark", height=300, xaxis_title="Date", yaxis_title="Drawdown (%)")
            st.plotly_chart(fig_dd, use_container_width=True)

            # Advanced Breakdowns
            col_left, col_right = st.columns(2)

            trades_df['entry_hour'] = pd.to_datetime(trades_df['entry_time']).dt.strftime('%H:%M')
            trades_df['day_of_week'] = pd.to_datetime(trades_df['entry_time']).dt.day_name()

            with col_left:
                st.subheader("Profitability by Entry Time")
                time_pnl = trades_df.groupby('entry_hour')['pnl'].sum().reset_index()
                fig_time = px.bar(time_pnl, x='entry_hour', y='pnl', color='pnl',
                                  color_continuous_scale=['#ff5252', '#ffeb3b', '#00e676'])
                fig_time.update_layout(template="plotly_dark", height=350, showlegend=False)
                st.plotly_chart(fig_time, use_container_width=True)

            with col_right:
                st.subheader("Performance by Day of the Week")
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                day_pnl = trades_df.groupby('day_of_week')['pnl'].sum().reindex(day_order).dropna().reset_index()
                fig_day = px.bar(day_pnl, x='day_of_week', y='pnl', color='pnl',
                                 color_continuous_scale=['#ff5252', '#ffeb3b', '#00e676'])
                fig_day.update_layout(template="plotly_dark", height=350, showlegend=False)
                st.plotly_chart(fig_day, use_container_width=True)
