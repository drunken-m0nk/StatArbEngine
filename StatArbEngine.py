"""
Statistical Arbitrage Engine (single-file)
Features:
- Download price data (yfinance) for Indian tickers (use .NS suffix)
- Identify cointegrated pairs via Engle-Granger (statsmodels.tsa.stattools.coint)
- Compute hedge ratio (OLS), spread and z-score
- Generate entry/exit signals based on z-score thresholds
- Risk controls: position sizing, per-trade stop-loss, transaction costs, slippage
- Backtester with P&L and performance metrics
- Streamlit dashboard (run with: streamlit run stat_arb_engine.py)

Requirements:
pip install pandas numpy scipy statsmodels yfinance matplotlib streamlit

Notes:
- This script is designed as a starting point and educational reference. For live trading, connect to a broker API and add robust error handling.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from scipy.stats import zscore
import matplotlib.pyplot as plt
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)

# -----------------------------
# Utility / Data Loader
# -----------------------------
class DataLoader:
    def __init__(self):
        pass

    def fetch_close(self, tickers, start, end):
        logging.info(f"Fetching data for {len(tickers)} tickers from {start} to {end}")
        data = yf.download(tickers, start=start, end=end, progress=False, threads=True)
        if ('Adj Close' in data):
            closes = data['Adj Close'].copy()
        else:
            closes = data['Close'].copy()
        closes.columns = [c.replace('.NS','') for c in closes.columns]
        return closes

# -----------------------------
# Pair Selector
# -----------------------------
class PairSelector:
    def __init__(self, pvalue_threshold=0.05):
        self.pvalue_threshold = pvalue_threshold

    def find_cointegrated_pairs(self, prices):
        tickers = prices.columns
        n = len(tickers)
        pairs = []
        logging.info("Running cointegration tests (this can be slow)")
        for i in range(n):
            for j in range(i+1, n):
                s1 = prices.iloc[:, i].dropna()
                s2 = prices.iloc[:, j].dropna()
                common_idx = s1.index.intersection(s2.index)
                if len(common_idx) < 200:
                    continue
                res = coint(s1.loc[common_idx], s2.loc[common_idx])
                pvalue = res[1]
                tstat = res[0]
                if pvalue < self.pvalue_threshold:
                    pairs.append((tickers[i], tickers[j], pvalue, tstat))
        pairs_sorted = sorted(pairs, key=lambda x: x[2])
        logging.info(f"Found {len(pairs_sorted)} cointegrated pairs")
        return pairs_sorted

# -----------------------------
# Signal Generator
# -----------------------------
class SignalGenerator:
    def __init__(self, lookback=252, entry_z=2.0, exit_z=0.5):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z

    def hedge_ratio(self, x, y):
        x = sm.add_constant(x)
        model = sm.OLS(y, x, missing='drop').fit()
        return model.params[1]

    def generate_signals(self, s1, s2):
        df = pd.concat([s1, s2], axis=1).dropna()
        df.columns = ['s1', 's2']
        hr_series, spreads, zscores = [], [], []
        for end in range(self.lookback, len(df)):
            window = df.iloc[end-self.lookback:end]
            b = self.hedge_ratio(window['s1'], window['s2'])
            spread = df['s2'].iloc[end] - b * df['s1'].iloc[end]
            hist = window['s2'] - b * window['s1']
            mu, sigma = hist.mean(), hist.std()
            z = (spread - mu) / sigma if sigma > 0 else 0.0
            hr_series.append(b)
            spreads.append(spread)
            zscores.append(z)
        idx = df.index[self.lookback:]
        out = pd.DataFrame({'hedge_ratio': hr_series, 'spread': spreads, 'zscore': zscores}, index=idx)
        out['short_entry'] = out['zscore'] > self.entry_z
        out['long_entry'] = out['zscore'] < -self.entry_z
        out['exit'] = out['zscore'].abs() < self.exit_z
        return out

# -----------------------------
# Risk Manager
# -----------------------------
class RiskManager:
    def __init__(self, capital=1000000, max_risk_per_pair=0.01, stop_loss_z=5.0, tc_per_trade=0.0003, slippage=0.0005):
        self.capital = capital
        self.max_risk_per_pair = max_risk_per_pair
        self.stop_loss_z = stop_loss_z
        self.tc = tc_per_trade
        self.slippage = slippage

    def size_position(self, price1, price2, hedge_ratio, spread_sigma):
        alloc = self.capital * self.max_risk_per_pair
        notional_leg = alloc / 2.0
        try:
            y = max(int(notional_leg / price2), 1)
        except Exception:
            y = 0
        if hedge_ratio != 0:
            x = max(int(round(y * price2 / (hedge_ratio * price1))), 1)
        else:
            x = 0
        return x, y

# -----------------------------
# Backtester
# -----------------------------
class Backtester:
    def __init__(self, prices, signals_df, hedge_side=('s1','s2'), risk_manager=None):
        self.prices = prices
        self.signals = signals_df
        self.risk_manager = risk_manager or RiskManager()
        self.results = None

    def run(self):
        position, cash = 0, self.risk_manager.capital
        shares = {'s1':0, 's2':0}
        trades, pnl_series = [], []

        for t in self.signals.index:
            sig = self.signals.loc[t]
            price1, price2 = self.prices.loc[t, 's1'], self.prices.loc[t, 's2']
            if position == 0:
                if sig['long_entry']:
                    b = sig['hedge_ratio']
                    hist = (self.prices['s2'] - b * self.prices['s1']).loc[:t].dropna()
                    spread_sigma = hist.std() if len(hist)>1 else 0.0
                    x, y = self.risk_manager.size_position(price1, price2, b, spread_sigma)
                    tc = (price1 * x + price2 * y) * (self.risk_manager.tc + self.risk_manager.slippage)
                    cash -= price2 * y
                    cash += price1 * x
                    cash -= tc
                    position, shares = 1, {'s1': -x, 's2': y}
                    trades.append({'time': t, 'type':'enter_long', 'x':x, 'y':y})
                elif sig['short_entry']:
                    b = sig['hedge_ratio']
                    hist = (self.prices['s2'] - b * self.prices['s1']).loc[:t].dropna()
                    spread_sigma = hist.std() if len(hist)>1 else 0.0
                    x, y = self.risk_manager.size_position(price1, price2, b, spread_sigma)
                    tc = (price1 * x + price2 * y) * (self.risk_manager.tc + self.risk_manager.slippage)
                    cash += price2 * y
                    cash -= price1 * x
                    cash -= tc
                    position, shares = -1, {'s1': x, 's2': -y}
                    trades.append({'time': t, 'type':'enter_short', 'x':x, 'y':y})
            else:
                if sig['exit']:
                    tc = (abs(shares['s1'])*price1 + abs(shares['s2'])*price2) * (self.risk_manager.tc + self.risk_manager.slippage)
                    cash += -shares['s1'] * price1
                    cash += -shares['s2'] * price2
                    cash -= tc
                    trades.append({'time': t, 'type':'exit', 'x':shares['s1'], 'y':shares['s2']})
                    shares, position = {'s1':0, 's2':0}, 0
            nav = cash + shares['s1'] * price1 + shares['s2'] * price2
            pnl_series.append({'time': t, 'nav': nav})

        pnl_df = pd.DataFrame(pnl_series).set_index('time')
        pnl_df['returns'] = pnl_df['nav'].pct_change().fillna(0)
        self.results = {'trades':trades, 'pnl':pnl_df, 'metrics':self.performance_metrics(pnl_df)}
        return self.results

    def performance_metrics(self, pnl_df):
        returns = pnl_df['returns']
        nav = pnl_df['nav']
        total_return = nav.iloc[-1]/nav.iloc[0] - 1
        ann_ret = ((1+total_return) ** (252/len(nav))) - 1 if len(nav) > 1 else 0
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_ret/ann_vol if ann_vol != 0 else 0
        drawdown = nav / nav.cummax() - 1
        max_dd = drawdown.min()
        metrics = {
            'Final NAV': nav.iloc[-1],
            'Total Return %': total_return*100,
            'Annualized Return %': ann_ret*100,
            'Annualized Volatility %': ann_vol*100,
            'Sharpe Ratio': sharpe,
            'Max Drawdown %': max_dd*100
        }
        return metrics

# -----------------------------
# Streamlit Dashboard
# -----------------------------

st.set_page_config(page_title='StatArb Engine', layout='wide')

@st.cache_data
def load_prices(tickers, start, end):
    dl = DataLoader()
    return dl.fetch_close(tickers, start, end)

st.title('Statistical Arbitrage Engine for Indian Equities')

with st.sidebar:
    st.header('Setup')
    tickers_input = st.text_area('Enter tickers (comma separated, add .NS if you like)', value='RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS')
    start = st.date_input('Start date', value=pd.to_datetime('2022-01-01'))
    end = st.date_input('End date', value=pd.to_datetime(datetime.today().date()))
    pvalue_thresh = st.slider('Cointegration p-value threshold', min_value=0.01, max_value=0.2, value=0.05, step=0.01)
    lookback = st.number_input('Lookback (days) for zscore', value=252, min_value=60, max_value=1000)
    entry_z = st.number_input('Entry z threshold', value=2.0, step=0.1)
    exit_z = st.number_input('Exit z threshold', value=0.5, step=0.1)
    capital = st.number_input('Capital (INR)', value=1000000)
    max_risk_per_pair = st.number_input('Max risk per pair (fraction of capital)', value=0.01, step=0.001)
    tc = st.number_input('Transaction cost per trade (fraction)', value=0.0003)
    slippage = st.number_input('Slippage fraction', value=0.0005)
    run_btn = st.button('Run analysis')

if run_btn:
    tickers = [t.strip() for t in tickers_input.split(',') if t.strip()]
    if len(tickers) < 2:
        st.error('Enter at least two tickers')
    else:
        prices = load_prices(tickers, start, end)
        st.success('Data loaded — showing last 5 rows')
        st.dataframe(prices.tail())

        selector = PairSelector(pvalue_threshold=pvalue_thresh)
        pairs = selector.find_cointegrated_pairs(prices)
        if len(pairs) == 0:
            st.warning('No cointegrated pairs found with given threshold')
        else:
            st.subheader('Cointegrated pairs')
            pairs_df = pd.DataFrame(pairs, columns=['s1','s2','pvalue','tstat'])
            st.dataframe(pairs_df)

            pair_choice = st.selectbox('Select pair to analyze', pairs_df.index.tolist(), format_func=lambda i: f"{pairs_df.loc[i,'s1']} - {pairs_df.loc[i,'s2']} (p={pairs_df.loc[i,'pvalue']:.4f})")
            s1, s2 = pairs_df.loc[pair_choice, 's1'], pairs_df.loc[pair_choice, 's2']
            st.write(f'Analyzing pair: {s1} & {s2}')

            sg = SignalGenerator(lookback=lookback, entry_z=entry_z, exit_z=exit_z)
            sigs = sg.generate_signals(prices[s1], prices[s2])
            st.subheader('Signals overview (last 10 rows)')
            st.dataframe(sigs.tail(10))

            fig1, ax1 = plt.subplots(figsize=(10,4))
            ax1.plot(sigs['spread'].iloc[-500:])
            ax1.set_title('Spread (recent)')
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots(figsize=(10,4))
            ax2.plot(sigs['zscore'].iloc[-500:])
            ax2.axhline(entry_z, linestyle='--')
            ax2.axhline(-entry_z, linestyle='--')
            ax2.axhline(exit_z, linestyle=':')
            ax2.axhline(-exit_z, linestyle=':')
            ax2.set_title('Z-score (recent)')
            st.pyplot(fig2)

            rm = RiskManager(capital=capital, max_risk_per_pair=max_risk_per_pair, tc_per_trade=tc, slippage=slippage)
            p_df = pd.concat([prices[s1], prices[s2]], axis=1).dropna()
            p_df.columns = ['s1','s2']
            p_df = p_df.loc[sigs.index]
            bt = Backtester(p_df, sigs, risk_manager=rm)
            res = bt.run()
            pnl, trades, metrics = res['pnl'], res['trades'], res['metrics']

            st.subheader('Backtest Performance Metrics')
            st.table(pd.DataFrame(metrics, index=[0]).T.rename(columns={0:'Value'}))

            st.subheader('Trade Log')
            st.dataframe(pd.DataFrame(trades))

            fig3, ax3 = plt.subplots(figsize=(10,4))
            ax3.plot(pnl['nav'])
            ax3.set_title('NAV over time')
            st.pyplot(fig3)

            st.info('This demo uses simplified assumptions for shorting and cash flows. For production, integrate with broker APIs and margin calculations.')

if __name__ == "__main__":
    # If running with streamlit, call main()
    try:
        import streamlit as st
        main()
    except:
        # Fallback: Run in CLI mode
        from datetime import datetime
        tickers = ['RELIANCE.NS','TCS.NS','INFY.NS']
        dl = DataLoader()
        prices = dl.fetch_close(tickers, '2022-01-01', datetime.today().strftime('%Y-%m-%d'))
        ps = PairSelector()
        pairs = ps.find_cointegrated_pairs(prices)
        print('Pairs found:', pairs)
        if len(pairs) > 0:
            sg = SignalGenerator()
            sigs = sg.generate_signals(prices[pairs[0][0]], prices[pairs[0][1]])
            bt = Backtester(prices[[pairs[0][0], pairs[0][1]]].rename(columns={pairs[0][0]:'s1', pairs[0][1]:'s2'}), sigs)
            res = bt.run()
            print('Performance Metrics:', res['metrics'])