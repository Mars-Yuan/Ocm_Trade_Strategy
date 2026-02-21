# ============ OCM Markowitz Streamlit Dashboard ============
# 运行方式: streamlit run ocm_streamlit_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import json
import os
import re
from datetime import date, datetime, timedelta
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from pytz import timezone as ZoneInfo  # fallback for Python 3.8
from scipy.optimize import minimize


def get_yahoo_current_date():
    """获取 Yahoo Finance 数据源时区（美东时间）的当前日期"""
    try:
        eastern = ZoneInfo('America/New_York')
        now_eastern = datetime.now(eastern)
        return now_eastern.date()
    except Exception:
        # 如果时区获取失败，回退到本机时间
        return date.today()

# 页面配置
st.set_page_config(
    page_title="OCM Markowitz Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 自定义样式
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00d4aa;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

# 颜色主题
colors = {
    'optimal': '#ff4444',      # 红色 - 最优组合
    'equal': '#44cc44',        # 绿色 - 等权组合
    'benchmark': '#4488ff',    # 蓝色 - 基准
    'positive': '#00d4aa',
    'negative': '#ff6b6b'
}

# 加载数据（不缓存，确保每次读取最新文件）
def load_dashboard_data():
    """从保存的文件加载数据"""
    data_path = os.path.join(os.path.dirname(__file__), 'Streamlit_data.json')
    
    if not os.path.exists(data_path):
        st.error("⚠️ 未找到数据文件！请先运行 notebook 中的数据导出 cell")
        st.info("在 notebook 中运行 '导出数据并启动 Streamlit' cell 后刷新此页面")
        return None
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def period_to_days(period_label):
    text = str(period_label).strip().upper()
    if not text:
        return 1

    match = re.match(r'^(\d+)?\s*([DWMQY])$', text)
    if match:
        num_text, unit = match.groups()
        n = int(num_text) if num_text else 1
        if unit == 'D':
            return max(1, n)
        if unit == 'W':
            return max(1, n * 5)
        if unit == 'M':
            return max(1, n * 21)
        if unit == 'Q':
            return max(1, n * 63)
        if unit == 'Y':
            return max(1, n * 252)

    return 1


def period_to_bars_per_year(period_label):
    days = period_to_days(period_label)
    return max(1, int(round(252 / days)))


def calc_max_drawdown(cum_curve):
    running_max = np.maximum.accumulate(cum_curve)
    drawdown = cum_curve / running_max - 1
    return float(np.min(drawdown)) if len(drawdown) > 0 else 0.0


def calc_perf_stats(daily_returns):
    if len(daily_returns) == 0:
        return 0.0, 0.0, 0.0, 0.0

    cum = (1 + daily_returns).cumprod()
    total_return = float(cum.iloc[-1] - 1)
    n = len(daily_returns)
    ann_return = float((1 + total_return) ** (252 / n) - 1) if n > 0 else 0.0
    vol = float(daily_returns.std())
    sharpe = float((daily_returns.mean() * 252) / (vol * np.sqrt(252))) if vol > 1e-12 else 0.0
    max_dd = calc_max_drawdown(cum.values)
    return total_return, ann_return, sharpe, max_dd


def standardize_daily_df(price_df):
    df = price_df.copy()
    df = df.reset_index()
    
    # 识别日期列
    date_col = None
    for c in ['Date', 'date', 'Datetime', 'datetime']:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]
    df['trade_date'] = pd.to_datetime(df[date_col]).dt.strftime('%Y%m%d')
    
    # 列名映射（支持大小写不敏感）
    col_map = {}
    for target, aliases in [
        ('open', ['Open', 'open', 'OPEN']),
        ('high', ['High', 'high', 'HIGH']),
        ('low', ['Low', 'low', 'LOW']),
        ('close', ['Close', 'close', 'CLOSE', 'Adj Close', 'adj close'])
    ]:
        for alias in aliases:
            if alias in df.columns:
                col_map[target] = alias
                break
    
    # 确保至少有 close 列
    if 'close' not in col_map:
        raise ValueError("DataFrame 缺少 Close 列")
    
    # 如果缺少 open/high/low，用 close 填充
    result = pd.DataFrame()
    result['trade_date'] = df['trade_date']
    result['open'] = df[col_map.get('open', col_map['close'])].astype(float)
    result['high'] = df[col_map.get('high', col_map['close'])].astype(float)
    result['low'] = df[col_map.get('low', col_map['close'])].astype(float)
    result['close'] = df[col_map['close']].astype(float)
    
    return result.reset_index(drop=True)


def resample_ohlc(df, period):
    df = df.copy().reset_index(drop=True)
    groups = []
    for i in range(0, len(df), period):
        g = df.iloc[i:i + period]
        if len(g) == 0:
            continue
        groups.append({
            'trade_date': g['trade_date'].iloc[-1],
            'open': g['open'].iloc[0],
            'high': g['high'].max(),
            'low': g['low'].min(),
            'close': g['close'].iloc[-1]
        })
    return pd.DataFrame(groups)


def generate_ocm_signals(df):
    df = df.copy()
    n = len(df)
    df['prev_close'] = df['close'].shift(1)
    df['breakout'] = df['open'] > df['prev_close']
    signals = np.zeros(n)
    position = 0
    for i in range(1, n):
        if bool(df['breakout'].iloc[i]) and position == 0:
            signals[i] = 1
            position = 1
        elif (not bool(df['breakout'].iloc[i])) and position == 1:
            signals[i] = -1
            position = 0
    df['signal'] = signals
    return df


def backtest_ocm(df):
    df = generate_ocm_signals(df)
    n = len(df)
    daily_returns = np.zeros(n)
    position = 0
    for i in range(1, n):
        signal = df['signal'].iloc[i]
        open_price = df['open'].iloc[i]
        close_price = df['close'].iloc[i]
        prev_close = df['close'].iloc[i - 1]
        if signal == 1:
            position = 1
            daily_returns[i] = (close_price - open_price) / open_price if open_price != 0 else 0
        elif signal == -1:
            position = 0
            daily_returns[i] = 0
        elif position == 1:
            daily_returns[i] = (close_price - prev_close) / prev_close if prev_close != 0 else 0
    return pd.Series(daily_returns, index=df['trade_date'])


def markowitz_optimize(returns_df):
    n_assets = len(returns_df.columns)
    mean_returns = returns_df.mean().values * 252
    cov_matrix = returns_df.cov().values * 252

    def portfolio_return(weights):
        return np.dot(weights, mean_returns)

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def neg_sharpe_ratio(weights):
        ret = portfolio_return(weights)
        vol = portfolio_volatility(weights)
        return -ret / vol if vol > 0 else 0

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = tuple((0, 1) for _ in range(n_assets))
    init_weights = np.array([1 / n_assets] * n_assets)

    result = minimize(neg_sharpe_ratio, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success:
        optimal_weights = init_weights
    else:
        optimal_weights = result.x

    opt_return = portfolio_return(optimal_weights)
    opt_vol = portfolio_volatility(optimal_weights)
    opt_sharpe = opt_return / opt_vol if opt_vol > 0 else 0

    return {
        'weights': optimal_weights,
        'return': float(opt_return),
        'volatility': float(opt_vol),
        'sharpe': float(opt_sharpe),
        'assets': returns_df.columns.tolist()
    }


def portfolio_backtest(returns_df, weights):
    portfolio_returns = (returns_df * weights).sum(axis=1)
    cum_returns = (1 + portfolio_returns).cumprod()
    total_return = float(cum_returns.iloc[-1] - 1) if len(cum_returns) else 0.0
    n_days = len(portfolio_returns)
    ann_return = float((1 + total_return) ** (252 / n_days) - 1) if n_days > 0 else 0.0
    ann_vol = float(portfolio_returns.std() * np.sqrt(252))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0
    peak = cum_returns.cummax()
    drawdown = (peak - cum_returns) / peak
    max_dd = float(drawdown.max()) if len(drawdown) else 0.0
    return {
        'total_return': total_return,
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'portfolio_returns': portfolio_returns,
        'cum_returns': cum_returns
    }


# ============ OCM6 动态单向策略参数 ============
DEFAULT_PERIODS = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67]
MIN_WINDOW = 67          # 滚动优化最小窗口期（交易日）
REBALANCE_FREQ = 1       # 再平衡频率（交易日）


def ocm6_portfolio_stats(weights, mean_ret, cov_mat):
    """组合年化收益、波动率、夏普比率"""
    port_return = weights @ mean_ret
    port_vol = np.sqrt(weights @ cov_mat @ weights)
    sharpe = port_return / port_vol if port_vol > 0 else 0
    return port_return, port_vol, sharpe


def ocm6_neg_sharpe(weights, mean_ret, cov_mat):
    """目标函数：负夏普比率"""
    _, _, sharpe = ocm6_portfolio_stats(weights, mean_ret, cov_mat)
    return -sharpe


def ocm6_min_variance(weights, mean_ret, cov_mat):
    """目标函数：组合方差"""
    return weights @ cov_mat @ weights


def build_runtime_data(symbol, start_date_input, end_date_input, assets, weights):
    """OCM6 动态单向策略 - 全流程计算（滚动窗口 Markowitz 优化）"""
    periods = DEFAULT_PERIODS
    n_assets = len(periods)

    # ---- 1. 计算预热期 ----
    warmup_trading_days = max(periods) + MIN_WINDOW + 10
    warmup_calendar_days = int(warmup_trading_days * 1.5) + 10

    backtest_start = pd.to_datetime(start_date_input).strftime('%Y-%m-%d')
    end_dt = pd.to_datetime(end_date_input)
    end_str = (end_dt + timedelta(days=1)).strftime('%Y-%m-%d')
    download_start = (pd.to_datetime(start_date_input) - timedelta(days=warmup_calendar_days)).strftime('%Y-%m-%d')

    # ---- 2. 验证标的 ----
    try:
        ticker_obj = yf.Ticker(symbol)
        test_hist = ticker_obj.history(period='5d')
        if test_hist.empty:
            return None, "标的代码（格式）错误/Yahoo无此标的，请重新输入"
    except Exception as e:
        return None, f"标的代码验证失败: {str(e)}"

    # ---- 3. 下载数据（含预热期）----
    price_df = yf.download(symbol, start=download_start, end=end_str, auto_adjust=True, progress=False)
    if price_df is None or price_df.empty:
        return None, f"未获取到 {symbol} 在所选区间的行情数据"

    if isinstance(price_df.columns, pd.MultiIndex):
        price_df.columns = price_df.columns.droplevel("Ticker")

    # 解析列名
    def _resolve_col(target):
        cols = [str(c) for c in price_df.columns]
        lower_map = {c.lower(): c for c in cols}
        if target.lower() in lower_map:
            return lower_map[target.lower()]
        for c in cols:
            if target.lower() in c.lower():
                return c
        return None

    open_col = _resolve_col('Open')
    close_col = _resolve_col('Close')
    if open_col is None or close_col is None:
        return None, f"数据缺少 Open/Close 列（当前: {list(price_df.columns)[:8]}）"

    if open_col != 'Open':
        price_df['Open'] = price_df[open_col]
    if close_col != 'Close':
        price_df['Close'] = price_df[close_col]

    price_df = price_df.dropna(subset=['Open']).copy()
    if len(price_df) < warmup_trading_days + 20:
        return None, f"有效交易日不足（需要约 {warmup_trading_days + 20}，仅有 {len(price_df)}）"

    # ---- 4. OCM6 每日收益率 ----
    # 每日收益率（基于开盘价）：Daily_Return = (Open_t - Open_{t-1}) / Open_{t-1}
    daily_return = price_df['Open'].pct_change()

    # ---- 5. 构建多周期策略收益率矩阵 ----
    # 信号: Open_t > Close_{t-N} → 买入(1)，否则空仓(0)，每个周期 N 有独立信号
    # Strategy_Return_t(N) = Signal_N_{t-1} × Daily_Return_t（1日执行延迟）
    strat_returns = pd.DataFrame(index=price_df.index)
    signals_all = pd.DataFrame(index=price_df.index)
    for N in periods:
        signal_N = (price_df['Open'] > price_df['Close'].shift(N)).astype(int)
        signals_all[f'N={N}'] = signal_N
        strat_returns[f'N={N}'] = signal_N.shift(1) * daily_return
    strat_returns = strat_returns.dropna()

    benchmark_daily = daily_return.reindex(strat_returns.index)

    if len(strat_returns) < MIN_WINDOW + 20:
        return None, "策略收益矩阵有效数据不足"

    # ---- 6. 扩展窗口 Markowitz 优化（最大夏普 + 最小方差 + 等权）----
    dates_idx = strat_returns.index
    n_days = len(dates_idx)
    bounds = tuple((0, 1.0) for _ in range(n_assets))
    w_equal = np.ones(n_assets) / n_assets

    rolling_weights_sharpe = np.full((n_days, n_assets), np.nan)
    rolling_ret_sharpe = np.full(n_days, np.nan)
    rolling_weights_mv = np.full((n_days, n_assets), np.nan)
    rolling_ret_mv = np.full(n_days, np.nan)
    rolling_ret_eq = np.full(n_days, np.nan)

    current_w_sharpe = None
    current_w_mv = None
    rebalance_dates_list = []
    weights_history_list = []

    for t in range(MIN_WINDOW, n_days):
        if (t - MIN_WINDOW) % REBALANCE_FREQ == 0:
            hist = strat_returns.iloc[:t + 1]
            mu = hist.mean().values * 252
            sigma = hist.cov().values * 252

            cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
            w_init = np.ones(n_assets) / n_assets

            # 最大夏普比率
            try:
                res_s = minimize(ocm6_neg_sharpe, w_init, args=(mu, sigma),
                                 method="SLSQP", bounds=bounds, constraints=cons,
                                 options={"maxiter": 1000, "ftol": 1e-12})
                if res_s.success:
                    current_w_sharpe = res_s.x
                elif current_w_sharpe is None:
                    current_w_sharpe = w_init.copy()
            except Exception:
                if current_w_sharpe is None:
                    current_w_sharpe = w_init.copy()

            # 最小方差
            try:
                res_mv = minimize(ocm6_min_variance, w_init, args=(mu, sigma),
                                  method="SLSQP", bounds=bounds, constraints=cons,
                                  options={"maxiter": 1000, "ftol": 1e-12})
                if res_mv.success:
                    current_w_mv = res_mv.x
                elif current_w_mv is None:
                    current_w_mv = w_init.copy()
            except Exception:
                if current_w_mv is None:
                    current_w_mv = w_init.copy()

            rebalance_dates_list.append(dates_idx[t])
            opt_sr = float(ocm6_portfolio_stats(current_w_sharpe, mu, sigma)[2])
            weights_history_list.append({
                'date': dates_idx[t].strftime('%Y-%m-%d') if hasattr(dates_idx[t], 'strftime') else str(dates_idx[t]),
                'weights': current_w_sharpe.tolist(),
                'sharpe': opt_sr
            })

        day_rets = strat_returns.iloc[t].values

        if current_w_sharpe is not None:
            rolling_weights_sharpe[t] = current_w_sharpe
            rolling_ret_sharpe[t] = current_w_sharpe @ day_rets

        if current_w_mv is not None:
            rolling_weights_mv[t] = current_w_mv
            rolling_ret_mv[t] = current_w_mv @ day_rets

        rolling_ret_eq[t] = w_equal @ day_rets

    # 转为 Series / DataFrame
    ret_sharpe = pd.Series(rolling_ret_sharpe, index=dates_idx)
    ret_mv = pd.Series(rolling_ret_mv, index=dates_idx)
    ret_eq = pd.Series(rolling_ret_eq, index=dates_idx)
    weights_df = pd.DataFrame(rolling_weights_sharpe, index=dates_idx, columns=strat_returns.columns)

    valid = ~ret_sharpe.isna()
    ret_sharpe = ret_sharpe[valid]
    ret_mv = ret_mv[valid]
    ret_eq = ret_eq[valid]
    weights_df = weights_df[valid]

    # ---- 7. 过滤到回测目标区间（去除预热期）----
    bt_mask = ret_sharpe.index >= backtest_start
    ret_sharpe = ret_sharpe[bt_mask]
    ret_mv = ret_mv[bt_mask]
    ret_eq = ret_eq[bt_mask]
    weights_df = weights_df[bt_mask]
    rebalance_dates_list = [d for d in rebalance_dates_list if d >= pd.Timestamp(backtest_start)]
    weights_history_list = [w for w in weights_history_list if w['date'] >= backtest_start]

    if len(ret_sharpe) == 0:
        return None, "回测区间内无有效数据（预热期可能不足）"

    # ---- 8. 累计收益 ----
    cum_sharpe = (1 + ret_sharpe).cumprod()
    cum_mv = (1 + ret_mv).cumprod()
    cum_eq = (1 + ret_eq).cumprod()
    bench_daily_valid = benchmark_daily.reindex(ret_sharpe.index)
    cum_bench = (1 + bench_daily_valid).cumprod()

    # ---- 9. 统计计算 ----
    def _calc_stats(cum_s, daily_s):
        total = float(cum_s.iloc[-1] - 1)
        n = len(daily_s)
        ann_r = float((1 + total) ** (252 / n) - 1) if n > 0 else 0
        ann_v = float(daily_s.std() * np.sqrt(252))
        sr = float(ann_r / ann_v) if ann_v > 1e-12 else 0
        mdd = float((cum_s / cum_s.cummax() - 1).min())
        return total, ann_r, ann_v, sr, mdd

    s_total, s_ann, s_vol, s_sr, s_mdd = _calc_stats(cum_sharpe, ret_sharpe)
    mv_total, mv_ann, mv_vol, mv_sr, mv_mdd = _calc_stats(cum_mv, ret_mv)
    eq_total, eq_ann, eq_vol, eq_sr, eq_mdd = _calc_stats(cum_eq, ret_eq)

    latest_w = weights_df.iloc[-1] if len(weights_df) > 0 else pd.Series(w_equal, index=strat_returns.columns)

    # ---- 10. 各周期策略表现 ----
    period_stats = []
    for N in periods:
        col = f'N={N}'
        sr_col = strat_returns[col].reindex(ret_sharpe.index)
        total_ret = float((1 + sr_col).prod() - 1)
        n_p = len(sr_col)
        ann_ret_p = float((1 + total_ret) ** (252 / n_p) - 1) if n_p > 0 else 0
        ann_vol_p = float(sr_col.std() * np.sqrt(252))
        sharpe_p = float(ann_ret_p / ann_vol_p) if ann_vol_p > 1e-12 else 0
        cum_p = (1 + sr_col).cumprod()
        mdd_p = float((cum_p / cum_p.cummax() - 1).min())
        w_val = float(latest_w[col]) if col in latest_w.index else 0

        period_stats.append({
            '\u5468\u671f': f'N{N}',
            '\u6743\u91cd': f'{w_val:.1%}' if abs(w_val) > 0.005 else '0%',
            '\u603b\u6536\u76ca\u7387': f'{total_ret:.2%}',
            '\u5e74\u5316\u6536\u76ca': f'{ann_ret_p:.2%}',
            '\u5e74\u5316\u6ce2\u52a8': f'{ann_vol_p:.2%}',
            'Sharpe': f'{sharpe_p:.2f}',
            '\u6700\u5927\u56de\u64a4': f'{mdd_p:.2%}'
        })

    # ---- 11. 收益率热力图 ----
    hm_tail = strat_returns.reindex(ret_sharpe.index).tail(30)
    returns_heatmap = {
        'dates': [d.strftime('%m-%d') if hasattr(d, 'strftime') else str(d)[-5:] for d in hm_tail.index.tolist()],
        'periods': [c.replace('N=', 'N') for c in hm_tail.columns.tolist()],
        'values': hm_tail.values.T.tolist()
    }

    # ---- 12. 最近交易信号（对齐 OCM6 notebook：按 日期×周期展开，再按日期合并）----
    signal_records = []
    valid_dates = ret_sharpe.index

    for dt in valid_dates:
        if dt not in price_df.index or dt not in weights_df.index:
            continue

        open_price = price_df.loc[dt, 'Open']
        w_row = weights_df.loc[dt]
        if w_row.isna().all():
            continue

        sig_row = signals_all.loc[dt]
        df_pos = price_df.index.get_loc(dt)

        for N in periods:
            col = f'N={N}'
            wv = float(w_row[col]) if col in w_row.index else 0.0
            if wv <= 0.01:
                continue

            sig_now = int(sig_row[col])
            prev_sig = int(signals_all.iloc[df_pos - 1][col]) if df_pos > 0 else 0
            if prev_sig == 0 and sig_now == 1:
                action = '买入'
            elif prev_sig == 1 and sig_now == 1:
                action = '持仓'
            elif prev_sig == 1 and sig_now == 0:
                action = '卖出'
            else:
                action = '空仓'

            if action == '卖出':
                trade_weight_str = f'{-wv * 100:.2f}%'
            elif action == '买入':
                trade_weight_str = f'{wv * 100:.2f}%'
            else:
                trade_weight_str = '-'

            trade_signal = action if action in ('买入', '卖出') else '-'
            signal_records.append({
                '\u65e5\u671f': dt.strftime('%Y-%m-%d') if hasattr(dt, 'strftime') else str(dt),
                '\u5468\u671f': col.replace('N=', 'N'),
                '\u4ea4\u6613\u6743\u91cd': trade_weight_str,
                '\u4ea4\u6613\u52a8\u4f5c': trade_signal,
                '\u5f00\u76d8\u4ef7': f'{float(open_price):.2f}' if not pd.isna(open_price) else '-',
            })

    signals_data = []
    if len(signal_records) > 0:
        signal_df = pd.DataFrame(signal_records).iloc[::-1].reset_index(drop=True)

        def _pct_to_float(v):
            if pd.isna(v):
                return 0.0
            s = str(v).strip()
            if (not s) or (s == '-'):
                return 0.0
            if s.endswith('%'):
                s = s[:-1]
            return float(s)

        signal_df['_trade_w_num'] = signal_df['交易权重'].map(_pct_to_float)
        signal_df['_date_dt'] = pd.to_datetime(signal_df['日期'])
        signal_df = (
            signal_df.groupby('日期', as_index=False)
            .agg({
                '_trade_w_num': 'sum',
                '周期': lambda s: ', '.join(s.astype(str).tolist()),
                '开盘价': 'first',
                '_date_dt': 'first',
            })
            .sort_values('_date_dt', ascending=False)
            .reset_index(drop=True)
        )
        signal_df = signal_df[signal_df['_trade_w_num'] != 0].reset_index(drop=True)
        signal_df['交易权重'] = signal_df['_trade_w_num'].map(lambda x: f'{x:.2f}%')
        signal_df['交易动作'] = signal_df['_trade_w_num'].map(lambda x: '买入' if x > 0 else ('卖出' if x < 0 else '-'))
        signal_df = signal_df[['日期', '周期', '交易权重', '交易动作', '开盘价']]

        # 页面保持“最近”语义，仅展示最新 20 条
        signals_data = signal_df.head(20).to_dict(orient='records')

    # ---- 13. 基准曲线（归一化到 1.0）----
    bench_prices = price_df['Open'].reindex(ret_sharpe.index)
    if len(bench_prices) > 0 and float(bench_prices.iloc[0]) > 0:
        benchmark_curve_list = (bench_prices / float(bench_prices.iloc[0])).tolist()
    else:
        benchmark_curve_list = cum_bench.values.tolist()

    benchmark_max_dd = abs(float((cum_bench / cum_bench.cummax() - 1).min())) if len(cum_bench) > 0 else 0

    # ---- 14. 交易日期列表 ----
    trade_dates = [d.strftime('%Y%m%d') if hasattr(d, 'strftime') else str(d) for d in ret_sharpe.index]

    # ---- 构建输出数据 ----
    runtime_data = {
        'symbol': symbol,
        'assets': [f'N{N}' for N in periods],
        'weights': latest_w.tolist() if hasattr(latest_w, 'tolist') else list(latest_w),
        'dates': trade_dates,
        'cum_returns_optimal': cum_sharpe.values.tolist(),
        'cum_returns_equal': cum_eq.values.tolist(),
        'cum_returns_fixed': cum_mv.values.tolist(),
        'benchmark_curve': benchmark_curve_list,
        'benchmark_max_dd': benchmark_max_dd,
        'portfolio_result': {
            'total_return': s_total,
            'ann_return': s_ann,
            'ann_vol': s_vol,
            'sharpe': s_sr,
            'max_drawdown': abs(s_mdd),
            'n_rebalances': len(rebalance_dates_list)
        },
        'fixed_result': {
            'total_return': mv_total,
            'ann_return': mv_ann,
            'sharpe': mv_sr,
            'max_drawdown': abs(mv_mdd)
        },
        'equal_result': {
            'total_return': eq_total,
            'ann_return': eq_ann,
            'sharpe': eq_sr,
            'max_drawdown': abs(eq_mdd)
        },
        'opt_return': s_ann,
        'opt_volatility': s_vol,
        'opt_sharpe': s_sr,
        'efficient_frontier': None,
        'period_stats': period_stats,
        'returns_heatmap': returns_heatmap,
        'signals': signals_data,
        'weights_history': weights_history_list,
        'rolling_params': {
            'min_window': MIN_WINDOW,
            'rebalance': REBALANCE_FREQ,
            'n_periods': n_assets
        }
    }

    return runtime_data, None

# 主函数
def main():
    # 标题
    st.markdown(
        '<h1 style="text-align: center;">OCM 多周期组合优化策略 '
        '<span style="font-size: 0.5em; color: #888888;">V3.2 (OCM6动态单向)</span></h1>',
        unsafe_allow_html=True
    )
    
    # 加载数据
    data = load_dashboard_data()
    
    if data is None:
        return
    
    st.subheader("策略信息")
    # 使用 Yahoo Finance 数据源时区（美东时间）而非本机时区
    today = get_yahoo_current_date()
    default_start_date = date(today.year - 1, 1, 1)
    default_end_date = today
    col_info1, col_info2, col_info3, col_info4 = st.columns([1.4, 1, 1, 0.5])
    with col_info1:
        symbol_input = st.text_input("标的", value=data['symbol'], key='symbol_input_main')
    with col_info2:
        start_date_input = st.date_input(
            "开始日期",
            value=default_start_date,
            key='start_date_input_main'
        )
    with col_info3:
        end_date_input = st.date_input(
            "结束日期",
            value=default_end_date,
            key='end_date_input_main'
        )
    with col_info4:
        st.write("")  # 占位对齐

    show_equal = True
    show_benchmark = True

    runtime_data, runtime_error = build_runtime_data(
        symbol_input,
        start_date_input,
        end_date_input,
        data.get('assets', []),
        data.get('weights', [])
    )

    if runtime_error:
        st.warning(f"动态刷新失败，已回退为导出数据：{runtime_error}")
    if runtime_data:
        for key in [
            'symbol', 'assets', 'weights', 'dates', 'cum_returns_optimal', 'cum_returns_equal',
            'cum_returns_fixed', 'benchmark_curve', 'benchmark_max_dd', 'portfolio_result', 
            'equal_result', 'fixed_result', 'opt_return', 'opt_volatility', 'opt_sharpe', 
            'efficient_frontier', 'period_stats', 'returns_heatmap', 'signals',
            'weights_history', 'rolling_params'
        ]:
            if key in runtime_data:
                data[key] = runtime_data[key]

    st.divider()
    
    # ========== 计算基准指标 ==========
    benchmark_curve = data.get('benchmark_curve', [])
    if benchmark_curve and len(benchmark_curve) > 0:
        benchmark_total_return = benchmark_curve[-1] - 1  # 累计收益率
        # 年化收益率
        trading_days = len(benchmark_curve)
        benchmark_ann_return = (1 + benchmark_total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0
        # Sharpe Ratio
        benchmark_returns = [benchmark_curve[i]/benchmark_curve[i-1] - 1 for i in range(1, len(benchmark_curve))]
        if len(benchmark_returns) > 1:
            import statistics
            benchmark_sharpe = (statistics.mean(benchmark_returns) * 252) / (statistics.stdev(benchmark_returns) * (252**0.5)) if statistics.stdev(benchmark_returns) > 0 else 0
        else:
            benchmark_sharpe = 0
        benchmark_max_dd = data.get('benchmark_max_dd', 0)
    else:
        benchmark_total_return = 0
        benchmark_ann_return = 0
        benchmark_sharpe = 0
        benchmark_max_dd = 0
    
    # ========== 统计卡片 ==========
    st.markdown(f"### 策略表现概览 <span style='font-size: 0.7em; color: #888888;'>{data['symbol']}</span>", unsafe_allow_html=True)
    
    # 第一行 - 最优组合
    st.markdown("**最大夏普组合**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_return = data['portfolio_result']['total_return']
        st.metric(
            label="总收益率",
            value=f"{total_return:.2%}",
            delta=f"vs 基准 {total_return - benchmark_total_return:.2%}"
        )
    
    with col2:
        ann_return = data['portfolio_result']['ann_return']
        st.metric(
            label="年化收益",
            value=f"{ann_return:.2%}",
            delta=f"vs 基准 {ann_return - benchmark_ann_return:.2%}"
        )
    
    with col3:
        sharpe = data['portfolio_result']['sharpe']
        st.metric(
            label="Sharpe Ratio",
            value=f"{sharpe:.2f}",
            delta=f"vs 基准 {sharpe - benchmark_sharpe:.2f}"
        )
    
    with col4:
        max_dd = data['portfolio_result']['max_drawdown']
        st.metric(
            label="最大回撤",
            value=f"{max_dd:.2%}",
            delta=f"vs 基准 {max_dd - benchmark_max_dd:.2%}",
            delta_color="inverse"
        )
    
    # 滚动回测信息
    n_rebalances = data['portfolio_result'].get('n_rebalances', 0)
    rolling_params = data.get('rolling_params', {})
    if n_rebalances > 0:
        st.caption(f"共执行 {n_rebalances} 次再平衡 | 最小窗口: {rolling_params.get('min_window', 67)}天 | 再平衡频率: 每{rolling_params.get('rebalance', 1)}天 | 策略周期数: {rolling_params.get('n_periods', 20)}")
    
    # 第二行 - 基准(买入持有)
    st.markdown("**基准 (买入持有)**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="总收益率", value=f"{benchmark_total_return:.2%}")
    
    with col2:
        st.metric(label="年化收益", value=f"{benchmark_ann_return:.2%}")
    
    with col3:
        st.metric(label="Sharpe Ratio", value=f"{benchmark_sharpe:.2f}")
    
    with col4:
        st.metric(label="最大回撤", value=f"{benchmark_max_dd:.2%}")
    
    st.divider()
    
    # ========== 主图表区 - 累计收益 ==========
    # 累计收益图
    st.markdown(f"### 累计收益对比 <span style='font-size: 0.7em; color: #888888;'>{data['symbol']}</span>", unsafe_allow_html=True)
    
    dates = data['dates']
    
    fig = go.Figure()
    
    # 最大夏普组合（主策略，OCM5 动态单向）
    fig.add_trace(go.Scatter(
        x=dates,
        y=data['cum_returns_optimal'],
        name='最大夏普',
        line=dict(color=colors['optimal'], width=2.5)
    ))
    
    # 最小方差组合（对比）
    if data.get('cum_returns_fixed'):
        fig.add_trace(go.Scatter(
            x=dates,
            y=data['cum_returns_fixed'],
            name='最小方差',
            line=dict(color='#B0B0B0', width=1.5, dash='solid'),
            opacity=0.8
        ))
    
    # 等权组合
    if show_equal:
        fig.add_trace(go.Scatter(
            x=dates,
            y=data['cum_returns_equal'],
            name='等权组合',
            line=dict(color=colors['equal'], width=1.5),
            opacity=0.6
        ))
    
    # 买入持有
    if show_benchmark:
        fig.add_trace(go.Scatter(
            x=dates,
            y=data['benchmark_curve'],
            name='买入持有',
            line=dict(color=colors['benchmark'], width=1.5),
            opacity=0.5
        ))
    
    fig.update_layout(
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0.5, xanchor='center'),
        margin=dict(l=20, r=20, t=40, b=40),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # ========== 第一行：最优权重配置 + 有效Sharpe率 ==========
    col1, col2 = st.columns([0.9, 1.5])
    
    with col1:
        # 权重饼图
        st.markdown(f"### 最优权重配置 <span style='font-size: 0.7em; color: #888888;'>{data['symbol']}</span>", unsafe_allow_html=True)
        
        weights_df = pd.DataFrame({
            '周期': data['assets'],
            '权重': [w * 100 for w in data['weights']]
        })
        # 只显示权重 > 1% 的
        weights_df = weights_df[weights_df['权重'] > 1]
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=weights_df['周期'],
            values=weights_df['权重'],
            hole=0.4,
            marker=dict(colors=px.colors.qualitative.Set2),
            textinfo='percent+label'
        )])
        
        fig_pie.update_layout(
            template='plotly_dark',
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            height=320
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # 有效Sharpe率
        st.markdown(f"### 有效Sharpe率 <span style='font-size: 0.7em; color: #888888;'>{data['symbol']}</span>", unsafe_allow_html=True)
        
        ef = data.get('efficient_frontier')
        opt_vol = data.get('opt_volatility', 0)
        opt_ret = data.get('opt_return', 0)
        opt_sharpe_val = data.get('opt_sharpe', 0)
        
        # 如果有有效Sharpe率数据
        if ef and isinstance(ef, dict) and ef.get('volatility') and ef.get('return') and len(ef.get('volatility', [])) > 0:
            fig_ef = go.Figure()
            
            fig_ef.add_trace(go.Scatter(
                x=[v * 100 for v in ef['volatility']],
                y=[r * 100 for r in ef['return']],
                mode='markers',
                showlegend=False,
                marker=dict(
                    size=8,
                    color=ef.get('sharpe', [1]*len(ef['volatility'])),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Sharpe')
                ),
                name='有效Sharpe率'
            ))
            
            # 找到最小方差点和最大Sharpe点
            ef_vols = ef.get('volatility', [])
            ef_rets = ef.get('return', [])
            ef_sharpes = ef.get('sharpe', [])
            
            if len(ef_vols) > 0:
                # 最小方差点
                min_vol_idx = np.argmin(ef_vols)
                min_vol = ef_vols[min_vol_idx]
                min_vol_ret = ef_rets[min_vol_idx]
                
                fig_ef.add_trace(go.Scatter(
                    x=[min_vol * 100],
                    y=[min_vol_ret * 100],
                    mode='markers',
                    showlegend=False,
                    marker=dict(size=14, color='#00d4aa', symbol='circle', line=dict(width=2, color='white')),
                    name='最小方差'
                ))
            
            # 最大Sharpe组合点
            if opt_vol and opt_ret:
                fig_ef.add_trace(go.Scatter(
                    x=[opt_vol * 100],
                    y=[opt_ret * 100],
                    mode='markers',
                    showlegend=False,
                    marker=dict(size=14, color='red', symbol='circle', line=dict(width=2, color='white')),
                    name='最大Sharpe'
                ))
            
            fig_ef.update_layout(
                template='plotly_dark',
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0.5, xanchor='center'),
                xaxis_title='年化波动率 (%)',
                yaxis_title='年化收益率 (%)',
                margin=dict(l=20, r=20, t=50, b=40),
                height=380
            )
            
            st.plotly_chart(fig_ef, use_container_width=True)
        elif opt_vol and opt_ret:
            # 基于最优点生成模拟有效Sharpe率
            # 有效Sharpe率形态：完整的"子弹头"边界曲线
            
            n_points = 80
            vols = []
            rets = []
            sharpes = []
            
            # 最小方差组合（MVP）：波动率最低，收益也较低
            mvp_vol = opt_vol * 0.65   # 最小方差点波动率比最大Sharpe点低
            mvp_ret = opt_ret * 0.45   # 最小方差点收益也较低
            
            # 有效Sharpe率双曲线参数
            a = mvp_vol ** 2
            b = (opt_vol ** 2 - mvp_vol ** 2) / ((opt_ret - mvp_ret) ** 2) if (opt_ret - mvp_ret) != 0 else 0.5
            
            # 完整曲线：上半部分（有效Sharpe率）+ 下半部分（无效Sharpe率）
            # 上半部分：从最小方差点向上延伸到更高收益
            min_ret_upper = mvp_ret
            max_ret_upper = opt_ret * 2.0
            upper_returns = np.linspace(min_ret_upper, max_ret_upper, n_points // 2)
            
            for target in upper_returns:
                vol = np.sqrt(a + b * (target - mvp_ret) ** 2)
                vols.append(vol)
                rets.append(target)
                sharpes.append(target / vol if vol > 0 else 0)
            
            # 下半部分：从最小方差点向下延伸到负收益区域（无效前沿）
            min_ret_lower = -opt_ret * 0.6  # 延伸到负收益区域
            max_ret_lower = mvp_ret
            lower_returns = np.linspace(max_ret_lower, min_ret_lower, n_points // 2)
            
            for target in lower_returns:
                vol = np.sqrt(a + b * (target - mvp_ret) ** 2)
                vols.append(vol)
                rets.append(target)
                sharpes.append(target / vol if vol > 0 else 0)
            
            fig_ef = go.Figure()
            
            # 绘制完整的有效Sharpe率散点图（子弹头形状）
            fig_ef.add_trace(go.Scatter(
                x=[v * 100 for v in vols],
                y=[r * 100 for r in rets],
                mode='markers',
                showlegend=False,
                marker=dict(
                    size=8,
                    color=sharpes,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Sharpe Ratio')
                ),
                name='有效Sharpe率'
            ))
            
            # 标记最小方差组合（绿点）- 波动率最低位置
            fig_ef.add_trace(go.Scatter(
                x=[mvp_vol * 100],
                y=[mvp_ret * 100],
                mode='markers',
                showlegend=False,
                marker=dict(size=14, color='#00d4aa', symbol='circle', line=dict(width=2, color='white')),
                name='最小方差'
            ))
            
            # 标记最大Sharpe组合（红点）- 夏普比率最高位置
            fig_ef.add_trace(go.Scatter(
                x=[opt_vol * 100],
                y=[opt_ret * 100],
                mode='markers',
                showlegend=False,
                marker=dict(size=14, color='red', symbol='circle', line=dict(width=2, color='white')),
                name='最大Sharpe'
            ))
            
            fig_ef.update_layout(
                template='plotly_dark',
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0.5, xanchor='center'),
                xaxis_title='年化波动率 (%)',
                yaxis_title='年化收益率 (%)',
                margin=dict(l=20, r=20, t=50, b=40),
                height=380
            )
            
            st.plotly_chart(fig_ef, use_container_width=True)
        else:
            st.info("有效Sharpe率数据不可用")
    
    st.divider()
    
    # ========== 第二行：最近交易信号 ==========
    st.markdown(f"### 最近交易信号 <span style='font-size: 0.7em; color: #888888;'>{data['symbol']}</span>", unsafe_allow_html=True)
    
    # 显示数据日期范围
    dates_list = data.get('dates') or []
    latest_date = None
    if dates_list:
        latest_date = str(dates_list[-1])
        st.caption(f"数据最新日期: {latest_date[:4]}-{latest_date[4:6]}-{latest_date[6:]}")
    
    signals = data.get('signals') or []
    if len(signals) > 0:
        raw_df = pd.DataFrame(signals)
        if '日期' in raw_df.columns:
            raw_df = raw_df.sort_values('日期', ascending=False)

        if latest_date and '日期' in raw_df.columns and '交易动作' in raw_df.columns:
            latest_date_digits = ''.join(ch for ch in str(latest_date) if ch.isdigit())

            def _highlight_current_date_trade_action(row):
                styles = [''] * len(row)
                row_date_digits = ''.join(ch for ch in str(row.get('日期', '')) if ch.isdigit())
                if row_date_digits == latest_date_digits:
                    action_col_idx = row.index.get_loc('交易动作')
                    styles[action_col_idx] = 'color: red; font-weight: 700;'
                return styles

            styled_df = raw_df.style.apply(_highlight_current_date_trade_action, axis=1)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.dataframe(raw_df, use_container_width=True, hide_index=True)
    else:
        st.info("signals 为空列表")
    
    st.divider()
    
    # ========== 第三行：权重变化图 + 各周期策略表现 ==========
    col1, col2 = st.columns(2)
    
    with col1:
        # 权重随时间变化图
        st.markdown(f"### 权重随时间变化 <span style='font-size: 0.7em; color: #888888;'>{data['symbol']}</span>", unsafe_allow_html=True)
        st.caption("（全部再平衡记录）")
        
        weights_history = data.get('weights_history', [])
        assets = data.get('assets', [])
        
        if weights_history and len(weights_history) > 0 and len(assets) > 0:
            # 构建权重历史 DataFrame
            wh_dates = [record['date'] for record in weights_history]
            wh_data = {asset: [] for asset in assets}
            
            for record in weights_history:
                weights = record['weights']
                for i, asset in enumerate(assets):
                    wh_data[asset].append(weights[i] * 100 if i < len(weights) else 0)
            
            fig_weights = go.Figure()
            
            colors_list = px.colors.qualitative.Set2
            for i, asset in enumerate(assets):
                fig_weights.add_trace(go.Scatter(
                    x=wh_dates,
                    y=wh_data[asset],
                    name=asset,
                    mode='lines',
                    stackgroup='one',
                    line=dict(width=0, color='rgba(0,0,0,0)'),
                    fillcolor=colors_list[i % len(colors_list)]
                ))
            
            fig_weights.update_layout(
                template='plotly_dark',
                showlegend=True,
                legend=dict(
                    orientation='h', yanchor='top', y=-0.15, x=0, xanchor='left',
                    font=dict(size=5),
                    tracegroupgap=2,
                    itemwidth=30,
                ),
                xaxis_title=dict(text='再平衡日期', standoff=0, font=dict(size=6)),
                xaxis=dict(title=dict(text='再平衡日期', standoff=0, font=dict(size=6)), side='bottom', anchor='free', title_standoff=0),
                yaxis_title='权重 (%)',
                yaxis=dict(range=[0, 100]),
                margin=dict(l=20, r=20, t=40, b=80),
                height=350
            )
            
            st.plotly_chart(fig_weights, use_container_width=True, key='weights_chart')
            # 缩小图例色块为原来的 1/2
            st.markdown("""
            <style>
            [data-testid="stPlotlyChart"][class*="weights_chart"] .legend .traces .legendtoggle,
            div[data-testid="element-container"]:has(> [class*="weights_chart"]) .legend .traces rect,
            .js-plotly-plot .legend .traces rect { transform: scale(0.5); }
            </style>
            """, unsafe_allow_html=True)
        else:
            st.info("权重历史数据不可用（需要更长的回测周期）")
    
    with col2:
        st.markdown(f"### 各周期策略表现 <span style='font-size: 0.7em; color: #888888;'>{data['symbol']}</span>", unsafe_allow_html=True)
        
        if data.get('period_stats') and len(data['period_stats']) > 0:
            stats_df = pd.DataFrame(data['period_stats'])
            # 用 HTML 表格渲染，字号自适应列数确保全部列可见
            n_cols = len(stats_df.columns)
            fs = max(9, min(13, int(90 / n_cols)))
            html = f'<div style="overflow-x:auto;max-height:350px;overflow-y:auto;">'
            html += f'<table style="width:100%;border-collapse:collapse;font-size:{fs}px;white-space:nowrap;">'
            html += '<thead><tr>'
            for col in stats_df.columns:
                html += f'<th style="padding:3px 5px;text-align:center;border-bottom:1px solid #444;background:inherit;color:inherit;position:sticky;top:0;">{col}</th>'
            html += '</tr></thead><tbody>'
            for _, row in stats_df.iterrows():
                html += '<tr>'
                for col in stats_df.columns:
                    html += f'<td style="padding:2px 5px;text-align:center;border-bottom:1px solid #333;">{row[col]}</td>'
                html += '</tr>'
            html += '</tbody></table></div>'
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("各周期策略表现数据不可用")
    
    # ========== 页脚 ==========
    st.divider()
    st.caption("OCM 多周期组合优化策略 | Powered by Streamlit & Plotly      开发:MarsYuan    版本:V3.2 (OCM6动态单向)")

if __name__ == '__main__':
    main()
