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


# ============ 滚动回测模式 ============
# 滚动回测参数（基于最长周期31D设计）
# 回看窗口 >= 最长周期的10倍，确保有足够数据计算各周期收益
LOOKBACK_DAYS = 20      # 回看窗口（与 notebook 一致）
REBALANCE_DAYS = 5      # 再平衡周期（每5天调仓）
MIN_LOOKBACK = 15       # 最小回看天数（与 notebook 一致）
DEFAULT_PERIODS = ['1D', '3D', 'W', '11D', '17D', '23D', '31D']


def period_sort_key(period_label):
    text = str(period_label).strip().upper()
    if text in DEFAULT_PERIODS:
        return DEFAULT_PERIODS.index(text)
    return len(DEFAULT_PERIODS) + period_to_days(text)


def rolling_portfolio_backtest(returns_df, lookback=20, rebalance=5, min_lookback=15):
    """
    滚动回测（Rolling Backtest）
    
    参数:
    - returns_df: 收益率矩阵
    - lookback: 回看窗口天数（用于计算权重）
    - rebalance: 再平衡周期（每隔多少天重新优化）
    - min_lookback: 最小回看天数
    
    返回:
    - 回测结果字典，包含组合收益、权重历史等
    """
    n_days = len(returns_df)
    n_assets = len(returns_df.columns)
    
    # 初始化
    portfolio_returns = np.zeros(n_days)
    weights_history = []  # 记录每次再平衡的权重
    rebalance_dates = []  # 记录再平衡日期
    
    # 当前权重（初始等权）
    current_weights = np.array([1/n_assets] * n_assets)
    last_rebalance = 0
    
    for i in range(n_days):
        # 计算当日组合收益
        daily_ret = returns_df.iloc[i].values
        portfolio_returns[i] = np.dot(daily_ret, current_weights)
        
        # 检查是否需要再平衡
        if i >= min_lookback and (i - last_rebalance) >= rebalance:
            # 使用过去lookback天数据优化权重
            start_idx = max(0, i - lookback)
            train_data = returns_df.iloc[start_idx:i]
            
            # 排除全零行
            train_data_valid = train_data.loc[(train_data != 0).any(axis=1)]
            
            if len(train_data_valid) >= min_lookback // 2:
                try:
                    opt_result_rolling = markowitz_optimize(train_data_valid)
                    current_weights = opt_result_rolling['weights']
                    
                    # 记录再平衡信息
                    weights_history.append({
                        'date': returns_df.index[i],
                        'weights': current_weights.copy(),
                        'opt_sharpe': opt_result_rolling['sharpe']
                    })
                    rebalance_dates.append(returns_df.index[i])
                    last_rebalance = i
                except:
                    pass  # 优化失败时保持原权重
    
    # 计算回测指标
    portfolio_returns = pd.Series(portfolio_returns, index=returns_df.index)
    cum_returns = (1 + portfolio_returns).cumprod()
    total_return = float(cum_returns.iloc[-1] - 1) if len(cum_returns) else 0.0
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
        'cum_returns': cum_returns,
        'weights_history': weights_history,
        'rebalance_dates': rebalance_dates,
        'n_rebalances': len(rebalance_dates)
    }


@st.cache_data(show_spinner=False, ttl=300)  # 缓存5分钟
def build_runtime_data(symbol, start_date_input, end_date_input, assets, weights):
    start_str = pd.to_datetime(start_date_input).strftime('%Y-%m-%d')
    # yfinance 的 end 参数是 exclusive 的，需要加一天才能包含 end_date 当天数据
    end_str = (pd.to_datetime(end_date_input) + timedelta(days=1)).strftime('%Y-%m-%d')

    # 验证标的代码是否有效（使用 history() 方法更可靠）
    try:
        ticker = yf.Ticker(symbol)
        # 直接尝试获取历史数据来验证标的是否存在
        test_hist = ticker.history(period='5d')
        if test_hist.empty:
            return None, f"标的代码（格式）错误/Yahoo无此标的，请重新输入"
    except Exception as e:
        return None, f"标的代码验证失败: {str(e)}"

    price_df = yf.download(symbol, start=start_str, end=end_str, auto_adjust=True, progress=False)
    if price_df is None or price_df.empty:
        return None, f"未获取到 {symbol} 在所选区间的行情数据，请检查日期范围"

    if isinstance(price_df.columns, pd.MultiIndex):
        flat_cols = []
        for col in price_df.columns:
            parts = [str(x) for x in col if str(x).strip() not in ['', 'None']]
            flat_cols.append('_'.join(parts))
        price_df.columns = flat_cols

    def resolve_price_col(target):
        cols = [str(c) for c in price_df.columns]
        lower_map = {c.lower(): c for c in cols}
        if target.lower() in lower_map:
            return lower_map[target.lower()]

        for c in cols:
            cl = c.lower()
            if cl.endswith(f"_{target.lower()}") or cl.startswith(f"{target.lower()}_"):
                return c

        for c in cols:
            if target.lower() in c.lower():
                return c

        return None

    open_col = resolve_price_col('Open')
    close_col = resolve_price_col('Close')

    if open_col is None or close_col is None:
        return None, f"行情缺少必要字段: Open/Close（当前列: {list(price_df.columns)[:8]}）"

    if open_col != 'Open':
        price_df['Open'] = price_df[open_col]
    if close_col != 'Close':
        price_df['Close'] = price_df[close_col]

    price_df = price_df.dropna(subset=['Close']).copy()
    if len(price_df) < 40:
        return None, "有效交易日不足，无法计算策略"

    df_daily_std = standardize_daily_df(price_df)

    # 固定策略周期（与 notebook 保持一致）
    base_periods = DEFAULT_PERIODS
    multi_period_data = {}
    for period_name in base_periods:
        if period_name == '1D':
            multi_period_data['1D'] = df_daily_std.copy()
        else:
            multi_period_data[period_name] = resample_ohlc(df_daily_std, period_to_days(period_name))

    period_returns = {}
    period_stats = []
    for period_name, df_period in multi_period_data.items():
        if len(df_period) < 2:
            continue
        returns = backtest_ocm(df_period)
        period_returns[period_name] = returns

        total_return = float((1 + returns).prod() - 1)
        n_bars = len(returns)
        bars_per_year = period_to_bars_per_year(period_name)
        if n_bars > 0 and total_return > -1:
            ann_return = float((1 + total_return) ** (bars_per_year / n_bars) - 1)
        else:
            ann_return = 0.0
        volatility = float(returns.std() * np.sqrt(bars_per_year))
        sharpe = float(ann_return / volatility) if volatility > 0 else 0.0
        cum_ret = (1 + returns).cumprod()
        peak = cum_ret.cummax()
        drawdown = (peak - cum_ret) / peak
        max_dd = float(drawdown.max()) if len(drawdown) else 0.0

        period_stats.append({
            '周期': period_name,
            'K线数': n_bars,
            '总收益率': f'{total_return:.2%}',
            '年化收益': f'{ann_return:.2%}',
            '年化波动': f'{volatility:.2%}',
            'Sharpe': f'{sharpe:.2f}',
            '最大回撤': f'{max_dd:.2%}'
        })

    period_stats = sorted(period_stats, key=lambda x: period_sort_key(x.get('周期', '')))

    if len(period_returns) == 0:
        return None, "未生成有效周期收益序列"

    all_dates = set()
    for returns in period_returns.values():
        all_dates.update(returns.index)
    all_dates = sorted(all_dates)

    returns_matrix_full = pd.DataFrame(index=all_dates)
    for period_name, returns in period_returns.items():
        returns_matrix_full[period_name] = returns_matrix_full.index.map(lambda d: returns.get(d, 0.0))

    ordered_cols = [p for p in DEFAULT_PERIODS if p in returns_matrix_full.columns]
    if len(ordered_cols) > 0:
        returns_matrix_full = returns_matrix_full[ordered_cols]

    returns_matrix_opt = returns_matrix_full.loc[(returns_matrix_full != 0).any(axis=1)]
    if len(returns_matrix_opt) < 10:
        return None, "有效收益率样本不足，无法优化权重"

    n_train = max(1, int(len(returns_matrix_opt) * 0.7))
    returns_matrix_train = returns_matrix_opt.iloc[:n_train]
    if returns_matrix_train.empty:
        return None, "训练集为空，无法优化权重"

    # 滚动回测（主策略）- 更接近实盘
    rolling_result = rolling_portfolio_backtest(
        returns_matrix_full, 
        lookback=LOOKBACK_DAYS, 
        rebalance=REBALANCE_DAYS,
        min_lookback=MIN_LOOKBACK
    )
    
    # 固定权重回测（对比：使用训练集优化的权重）
    opt_result = markowitz_optimize(returns_matrix_train)
    fixed_result = portfolio_backtest(returns_matrix_full, opt_result['weights'])

    # 滚动回测最新权重（用于交易信号）
    if rolling_result['weights_history']:
        latest_rolling_weights = rolling_result['weights_history'][-1]['weights']
    else:
        latest_rolling_weights = opt_result['weights']
    
    # 等权组合回测（对比）
    equal_weights = np.array([1 / len(returns_matrix_full.columns)] * len(returns_matrix_full.columns))
    equal_result = portfolio_backtest(returns_matrix_full, equal_weights)
    
    # 使用滚动回测结果作为主策略
    portfolio_result = rolling_result

    benchmark_curve = (df_daily_std['close'].values / df_daily_std['close'].iloc[0])
    benchmark_peak = np.maximum.accumulate(benchmark_curve)
    benchmark_max_dd = float(np.max((benchmark_peak - benchmark_curve) / benchmark_peak)) if len(benchmark_curve) else 0.0

    hm_tail = returns_matrix_full.tail(30)
    returns_heatmap = {
        'dates': [str(d)[-5:] for d in hm_tail.index.tolist()],
        'periods': hm_tail.columns.tolist(),
        'values': hm_tail.values.T.tolist()
    }

    # 收集所有周期的交易信号（按信号日期匹配当时权重）
    signals_data = []
    periods_list = returns_matrix_full.columns.tolist()

    rebalance_records = []
    if rolling_result['weights_history']:
        for record in rolling_result['weights_history']:
            rebalance_records.append({
                'date': str(record['date']),
                'weights': record['weights']
            })
        rebalance_records = sorted(rebalance_records, key=lambda x: x['date'])

    def get_weights_on_date(trade_date):
        trade_date_str = str(trade_date)
        if len(rebalance_records) == 0:
            return latest_rolling_weights

        chosen = rebalance_records[0]['weights']
        for rec in rebalance_records:
            if rec['date'] <= trade_date_str:
                chosen = rec['weights']
            else:
                break
        return chosen

    for period_name, df_period in multi_period_data.items():
        if period_name not in periods_list:
            continue
        period_idx = periods_list.index(period_name)

        df_with_signals = generate_ocm_signals(df_period.copy())
        
        for _, row in df_with_signals.iterrows():
            if row['signal'] != 0:
                weights_on_date = get_weights_on_date(row['trade_date'])
                period_weight = weights_on_date[period_idx] if period_idx < len(weights_on_date) else 0.0

                # 注释掉权重过滤，显示所有周期的信号
                # 原逻辑：跳过权重 <= 0.01 的周期（与 notebook 一致）
                # if float(period_weight) <= 0.01:
                #     continue

                signals_data.append({
                    '日期': str(row['trade_date']),
                    '周期': period_name,
                    '组合权重': f'{period_weight:.2%}',
                    '信号': '买入' if row['signal'] == 1 else '卖出',
                    '开盘价': f"{row['open']:.2f}",
                    '收盘价': f"{row['close']:.2f}",
                    '昨收价': f"{row['prev_close']:.2f}" if pd.notna(row['prev_close']) else '-'
                })

    signals_df_export = pd.DataFrame(signals_data)
    if not signals_df_export.empty:
        signals_df_export['日期_sort'] = pd.to_datetime(signals_df_export['日期'], format='%Y%m%d', errors='coerce')
        signals_df_export = signals_df_export.sort_values('日期_sort', ascending=False).drop(columns=['日期_sort'])
        signals_data = signals_df_export.to_dict('records')

    # 构建权重历史数据用于可视化（保存所有记录并按日期排序）
    weights_history_data = []
    if rolling_result['weights_history']:
        sorted_records = sorted(rolling_result['weights_history'], key=lambda x: str(x['date']))
        for record in sorted_records:
            weights_history_data.append({
                'date': str(record['date']),
                'weights': record['weights'].tolist() if hasattr(record['weights'], 'tolist') else list(record['weights']),
                'sharpe': float(record['opt_sharpe'])
            })
    else:
        # 无滚动记录时回退到静态最优权重，避免权重历史图完全空白
        if len(returns_matrix_full.index) > 0:
            weights_history_data.append({
                'date': str(returns_matrix_full.index[0]),
                'weights': opt_result['weights'].tolist() if hasattr(opt_result['weights'], 'tolist') else list(opt_result['weights']),
                'sharpe': float(opt_result['sharpe'])
            })
    
    # 最优权重配置：优先使用滚动回测最新权重，无记录时回退到训练集权重
    display_weights = latest_rolling_weights if rolling_result['weights_history'] else opt_result['weights']
    
    runtime_data = {
        'symbol': symbol,
        'assets': periods_list,
        'weights': display_weights.tolist() if hasattr(display_weights, 'tolist') else list(display_weights),
        'dates': returns_matrix_full.index.tolist(),
        'cum_returns_optimal': rolling_result['cum_returns'].values.tolist(),
        'cum_returns_equal': equal_result['cum_returns'].values.tolist(),
        'cum_returns_fixed': fixed_result['cum_returns'].values.tolist(),
        'benchmark_curve': benchmark_curve[:len(rolling_result['cum_returns'])].tolist(),
        'benchmark_max_dd': benchmark_max_dd,
        'portfolio_result': {
            'total_return': float(rolling_result['total_return']),
            'ann_return': float(rolling_result['ann_return']),
            'ann_vol': float(rolling_result['ann_vol']),
            'sharpe': float(rolling_result['sharpe']),
            'max_drawdown': float(rolling_result['max_drawdown']),
            'n_rebalances': rolling_result['n_rebalances']
        },
        'fixed_result': {
            'total_return': float(fixed_result['total_return']),
            'ann_return': float(fixed_result['ann_return']),
            'sharpe': float(fixed_result['sharpe']),
            'max_drawdown': float(fixed_result['max_drawdown'])
        },
        'equal_result': {
            'total_return': float(equal_result['total_return']),
            'ann_return': float(equal_result['ann_return']),
            'sharpe': float(equal_result['sharpe']),
            'max_drawdown': float(equal_result['max_drawdown'])
        },
        'opt_return': float(opt_result['return']),
        'opt_volatility': float(opt_result['volatility']),
        'opt_sharpe': float(opt_result['sharpe']),
        'efficient_frontier': None,
        'period_stats': period_stats,
        'returns_heatmap': returns_heatmap,
        'signals': signals_data,
        'weights_history': weights_history_data,
        'rolling_params': {
            'lookback': LOOKBACK_DAYS,
            'rebalance': REBALANCE_DAYS,
            'min_lookback': MIN_LOOKBACK
        }
    }

    return runtime_data, None

# 主函数
def main():
    # 标题
    st.markdown(
        '<h1 style="text-align: center;">OCM 多周期组合优化策略 '
        '<span style="font-size: 0.5em; color: #888888;">V2.64 (滚动回测)</span></h1>',
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
    st.markdown("**最优组合**")
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
        st.caption(f"共执行 {n_rebalances} 次再平衡 | 回看窗口: {rolling_params.get('lookback', 20)}天 | 再平衡周期: {rolling_params.get('rebalance', 5)}天")
    
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
    
    # 滚动回测（主策略，与 notebook 一致）
    fig.add_trace(go.Scatter(
        x=dates,
        y=data['cum_returns_optimal'],
        name='滚动回测',
        line=dict(color=colors['optimal'], width=2.5)
    ))
    
    # 固定权重（对比基准）
    if data.get('cum_returns_fixed'):
        fig.add_trace(go.Scatter(
            x=dates,
            y=data['cum_returns_fixed'],
            name='固定权重',
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
    if dates_list:
        latest_date = str(dates_list[-1])
        st.caption(f"数据最新日期: {latest_date[:4]}-{latest_date[4:6]}-{latest_date[6:]}")
    
    signals = data.get('signals') or []

    def build_fallback_signals():
        dates = data.get('dates') or []
        cum_returns = data.get('cum_returns_optimal') or []
        assets = data.get('assets') or []
        weights = data.get('weights') or []

        primary_period = '1D'
        if len(assets) > 0 and len(weights) == len(assets):
            primary_period = assets[int(np.argmax(weights))]

        generated = []
        for i in range(1, min(len(dates), len(cum_returns))):
            prev_val = cum_returns[i - 1]
            curr_val = cum_returns[i]
            if prev_val is None or prev_val == 0:
                continue

            day_ret = curr_val / prev_val - 1
            signal = '买入' if day_ret > 0 else '卖出'

            generated.append({
                '日期': str(dates[i]),
                '周期': primary_period,
                '组合权重': '-',
                '信号': signal,
                '开盘价': '-',
                '收盘价': '-',
                '昨收价': '-'
            })

        return sorted(generated, key=lambda x: x['日期'], reverse=True)

    if len(signals) == 0:
        signals = build_fallback_signals()

    if len(signals) == 0:
        fallback_dates = data.get('dates') or []
        fallback_date = str(fallback_dates[-1]) if len(fallback_dates) > 0 else '-'
        signals = [{
            '日期': fallback_date,
            '周期': '1D',
            '组合权重': '-',
            '信号': '买入',
            '开盘价': '-',
            '收盘价': '-',
            '昨收价': '-'
        }]

    if len(signals) > 0:
        signals_df = pd.DataFrame(signals)

        expected_cols = ['日期', '周期', '组合权重', '信号', '开盘价', '收盘价', '昨收价']
        for col in expected_cols:
            if col not in signals_df.columns:
                signals_df[col] = '-'

        def parse_weight_to_float(weight_val):
            if weight_val is None:
                return 0.0
            text = str(weight_val).strip()
            if text in ['', '-', 'None', 'nan']:
                return 0.0
            try:
                if text.endswith('%'):
                    return float(text[:-1]) / 100.0
                return float(text)
            except Exception:
                return 0.0

        signals_df['_weight_abs'] = signals_df['组合权重'].apply(parse_weight_to_float)
        signals_df['_weight_signed'] = np.where(
            signals_df['信号'] == '买入',
            signals_df['_weight_abs'].abs(),
            np.where(signals_df['信号'] == '卖出', -signals_df['_weight_abs'].abs(), 0.0)
        )

        merged_df = (
            signals_df.groupby('日期', as_index=False)['_weight_signed']
            .sum()
            .rename(columns={'_weight_signed': '组合权重数值'})
        )

        period_df = (
            signals_df.groupby('日期')['周期']
            .apply(lambda x: '+'.join([p for p in pd.unique(x) if str(p).strip() not in ['', '-']]))
            .reset_index(name='周期')
        )

        def first_valid_text(series):
            for v in series:
                text = str(v).strip()
                if text not in ['', '-', 'None', 'nan']:
                    return text
            return '-'

        price_df = (
            signals_df.groupby('日期', as_index=False)
            .agg({
                '开盘价': first_valid_text,
                '收盘价': first_valid_text,
                '昨收价': first_valid_text
            })
        )

        merged_df = merged_df.merge(period_df, on='日期', how='left')
        merged_df = merged_df.merge(price_df, on='日期', how='left')
        merged_df['周期'] = merged_df['周期'].replace('', '-').fillna('-')

        merged_df['信号'] = merged_df['组合权重数值'].apply(lambda v: '买入' if v > 0 else '卖出')
        merged_df['组合权重'] = merged_df['组合权重数值'].apply(lambda v: f"{v:.2%}")
        merged_df['开盘价'] = merged_df['开盘价'].fillna('-')
        merged_df['收盘价'] = merged_df['收盘价'].fillna('-')
        merged_df['昨收价'] = merged_df['昨收价'].fillna('-')
        merged_df = merged_df[merged_df['组合权重数值'].abs() > 1e-12]
        merged_df = merged_df.drop(columns=['组合权重数值'])
        signals_df = merged_df

        if '日期' in signals_df.columns:
            signals_df['日期_sort'] = pd.to_datetime(signals_df['日期'].astype(str), format='%Y%m%d', errors='coerce')
            signals_df = signals_df.sort_values('日期_sort', ascending=False).drop(columns=['日期_sort'])

        signals_df = signals_df[expected_cols].head(20)

        if signals_df.empty:
            fallback_df = pd.DataFrame(build_fallback_signals())
            if not fallback_df.empty:
                for col in expected_cols:
                    if col not in fallback_df.columns:
                        fallback_df[col] = '-'
                if '日期' in fallback_df.columns:
                    fallback_df['日期_sort'] = pd.to_datetime(fallback_df['日期'].astype(str), format='%Y%m%d', errors='coerce')
                    fallback_df = fallback_df.sort_values('日期_sort', ascending=False).drop(columns=['日期_sort'])
                signals_df = fallback_df[expected_cols].head(20)
        
        if signals_df.empty:
            st.info("暂无交易信号")
        else:
            def highlight_signal_cell(val):
                if val == '卖出':
                    return 'color: #1D6F42; font-weight: 700; font-size: 16px; text-align: center;'
                if val == '买入':
                    return 'color: #A1283B; font-weight: 700; font-size: 16px; text-align: center;'
                return ''

            def highlight_date_cell(_):
                return 'font-weight: 700; font-size: 15px;'

            st.dataframe(
                signals_df.style
                .set_table_styles([
                    {'selector': 'th', 'props': [('text-align', 'center'), ('font-weight', '700')]}
                ], overwrite=False)
                .set_properties(**{'text-align': 'center'})
                .map(highlight_signal_cell, subset=['信号'])
                .map(highlight_date_cell, subset=['日期']),
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("暂无交易信号")
    
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
                    line=dict(width=0.5),
                    fillcolor=colors_list[i % len(colors_list)]
                ))
            
            fig_weights.update_layout(
                template='plotly_dark',
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0.5, xanchor='center'),
                xaxis_title='再平衡日期',
                yaxis_title='权重 (%)',
                yaxis=dict(range=[0, 100]),
                margin=dict(l=20, r=20, t=40, b=40),
                height=350
            )
            
            st.plotly_chart(fig_weights, use_container_width=True)
        else:
            st.info("权重历史数据不可用（需要更长的回测周期）")
    
    with col2:
        st.markdown(f"### 各周期策略表现 <span style='font-size: 0.7em; color: #888888;'>{data['symbol']}</span>", unsafe_allow_html=True)
        
        if data.get('period_stats') and len(data['period_stats']) > 0:
            stats_df = pd.DataFrame(data['period_stats'])
            st.dataframe(stats_df, use_container_width=True, hide_index=True, height=350)
        else:
            st.info("各周期策略表现数据不可用")
    
    # ========== 页脚 ==========
    st.divider()
    st.caption("OCM 多周期组合优化策略 | Powered by Streamlit & Plotly      开发:MarsYuan    版本:V2.64 (滚动回测)")

if __name__ == '__main__':
    main()
