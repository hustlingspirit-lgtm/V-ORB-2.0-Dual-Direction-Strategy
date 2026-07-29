import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import zipfile
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

st.title("⚡ V-ORB 2.0 by claude: Advanced X-Ray Dashboard")

uploaded_file = st.file_uploader(
    "Upload Historical Data (Single .csv OR .zip containing CSVs)",
    type=["zip", "csv"]
)

REQUIRED_COLUMNS = {'open', 'high', 'low', 'close'}


@st.cache_data
def process_uploaded_data(uploaded):
    dfs = []

    if uploaded.name.endswith('.zip'):
        with zipfile.ZipFile(uploaded) as z:
            for filename in z.namelist():
                if filename.endswith('.csv') and not filename.startswith('__MACOSX/'):
                    with z.open(filename) as f:
                        df = pd.read_csv(f)
                        df.columns = df.columns.str.strip().str.lower()
                        if 'symbol' not in df.columns:
                            df['symbol'] = filename.replace('.csv', '')
                        dfs.append(df)
    elif uploaded.name.endswith('.csv'):
        df = pd.read_csv(uploaded)
        df.columns = df.columns.str.strip().str.lower()
        if 'symbol' not in df.columns:
            df['symbol'] = uploaded.name.replace('.csv', '')
        dfs.append(df)

    if dfs:
        df_combined = pd.concat(dfs, ignore_index=True)
        df_combined.rename(columns={
            'date': 'datetime',
            'time': 'datetime',
            'timestamp': 'datetime',
            'date/time': 'datetime'
        }, inplace=True)
        return df_combined
    return pd.DataFrame()


if uploaded_file is None:
    st.info("Please upload a .csv or .zip file to view results.")
else:
    df_raw = process_uploaded_data(uploaded_file)

    if df_raw.empty:
        st.error("No valid CSV data found in the uploaded file.")
    elif 'datetime' not in df_raw.columns:
        st.error("The uploaded CSV must contain a time column (e.g., 'datetime', 'date', 'timestamp').")
    elif not REQUIRED_COLUMNS.issubset(set(df_raw.columns)):
        missing = REQUIRED_COLUMNS - set(df_raw.columns)
        st.error(f"The uploaded CSV is missing required OHLC column(s): {', '.join(sorted(missing))}. "
                  f"Found columns: {', '.join(df_raw.columns)}")
    else:
        st.success("Data successfully loaded. Running backtest...")

        with st.expander("Data Diagnostic Viewer (click to expand)"):
            st.write(f"Total Rows in Dataset: {len(df_raw)}")
            st.write("First 5 Rows of Data:")
            st.dataframe(df_raw.head())

            debug_date = pd.to_datetime(df_raw['datetime']).dt.date
            rows_per_day = df_raw.groupby(debug_date).size()
            st.write("Candles per day (must be > 15 for the engine to trade):")
            st.dataframe(rows_per_day.head(10))

        trades_df, equity_df = run_backtest(df_raw)

        if trades_df.empty:
            st.error("No trades generated with the current dataset filters. "
                     "Ensure your CSV has intraday granularity (5-minute candles) "
                     "and covers enough days (each trading day needs at least 15 candles).")
        else:
            net_profit = trades_df['pnl'].sum()
            total_trades = len(trades_df)
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] <= 0]

            win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
            gross_profit = winning_trades['pnl'].sum()
            gross_loss = abs(losing_trades['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan
            expectancy = net_profit / total_trades if total_trades > 0 else 0

            equity_df['peak'] = equity_df['equity'].cummax()
            equity_df['drawdown_pct'] = ((equity_df['equity'] - equity_df['peak']) / equity_df['peak']) * 100
            max_drawdown = equity_df['drawdown_pct'].min()

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

            st.subheader("Dynamic Equity Curve")
            fig_eq = px.line(equity_df, x='datetime', y='equity', title="Capital Growth Over Time")
            fig_eq.update_traces(line_color="#00e676", line_width=2)
            fig_eq.update_layout(template="plotly_dark", height=400, xaxis_title="Date", yaxis_title="Equity")
            st.plotly_chart(fig_eq, use_container_width=True)

            st.subheader("Underwater Chart (Drawdown Depth)")
            fig_dd = px.area(equity_df, x='datetime', y='drawdown_pct', title="Drawdown (%)")
            fig_dd.update_traces(fillcolor="rgba(255, 82, 82, 0.3)", line_color="#ff5252")
            fig_dd.update_layout(template="plotly_dark", height=300, xaxis_title="Date", yaxis_title="Drawdown (%)")
            st.plotly_chart(fig_dd, use_container_width=True)

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
    
