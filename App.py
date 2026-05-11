import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px
import scipy.optimize as sco
import seaborn as sns
import datetime

st.set_page_config(layout="wide")

st.sidebar.markdown(
    """
    <a href="https://6309-lcy.github.io/Personal_Webpage/" target="_blank"
       style="display:inline-flex;align-items:left;gap:8px;background-color:#1f77b4;
              color:white;padding:10px 18px;border-radius:24px;text-decoration:none;
              font-weight:600;font-size:14px;box-shadow:0 4px 14px rgba(0,0,0,0.25);">
        About the Developer
    </a>
    """,
    unsafe_allow_html=True,
)

# ── Global parameters ────────────────────────────────────────────────────────
intervals        = '1d'
periods_per_year = 252
starts           = '2005-01-01'
ends             = datetime.date.today()
annual_rf        = 0.05          # fixed; no live fetch needed
market_ticker    = '^GSPC'


# ── Cached data-fetch helpers ─────────────────────────────────────────────────
# TTL = 3600 s means data is re-fetched at most once per hour, regardless of
# how many times Streamlit reruns the script.

@st.cache_data(ttl=3600)
def fetch_ticker_history(ticker: str, starts: str, ends: str, intervals: str) -> pd.Series:
    """Download daily Close prices for a SINGLE ticker (cached individually).
    Starting from 2005 gives buffer before the 2008 stress-test period.
    Adding a new stock only triggers ONE new download — existing tickers stay cached."""
    try:
        raw = yf.download(
            ticker,
            start=starts,
            end=str(ends),
            interval=intervals,
            auto_adjust=False,
            progress=False,
        )
        if raw.empty:
            return pd.Series(dtype=float, name=ticker)
        close = raw['Close']
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
        close.name = ticker
        return close.astype(float)
    except Exception:
        return pd.Series(dtype=float, name=ticker)


def fetch_historical_data(tickers: tuple, starts: str, ends: str, intervals: str) -> pd.DataFrame:
    """Merge per-ticker cached Series into one DataFrame.
    Each ticker is cached independently — adding a new stock only downloads that one ticker."""
    if not tickers:
        return pd.DataFrame()
    series = [fetch_ticker_history(t, starts, ends, intervals) for t in tickers]
    df = pd.concat(series, axis=1)
    df.columns = list(tickers)
    return df


@st.cache_data(ttl=3600)
def fetch_market_data(starts: str, ends: str, intervals: str) -> pd.Series:
    """Download S&P 500 Close prices.  Called ONCE per session thanks to caching."""
    raw = yf.download(
        market_ticker,
        start=starts,
        end=str(ends),
        interval=intervals,
        auto_adjust=False,
        progress=False,
    )
    if raw.empty:
        return pd.Series(dtype=float, name=market_ticker)
    close = raw['Close']
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    return close


@st.cache_data(ttl=3600)
def fetch_current_price(ticker: str) -> float:
    """Fetch a single ticker's most-recent Close price via yf.download (avoids .info rate limits)."""
    try:
        raw = yf.download(
            ticker,
            period='5d',          # last 5 trading days is enough
            interval='1d',
            auto_adjust=False,
            progress=False,
        )
        if raw.empty:
            return float('nan')
        close = raw['Close']
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
        last = close.dropna().iloc[-1]
        return float(last)
    except Exception:
        return float('nan')


# ── Compute market returns once (derived from cached data) ────────────────────
# This block runs only when market_data changes (i.e., at most once per hour).
market_data    = fetch_market_data(starts, str(ends), intervals)
market_returns = market_data.pct_change().dropna()
market_mean    = market_returns.mean() * periods_per_year
market_vol     = market_returns.std() * np.sqrt(periods_per_year)


# ── Session-state initialisation ──────────────────────────────────────────────
if 'account_balance' not in st.session_state:
    st.session_state.account_balance = 100_000.0
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}   # {ticker: {shares, buy_price, buy_date}}


# ═════════════════════════════════════════════════════════════════════════════
st.title("Portfolio Management Application")

# ── Section 1: Account balance ────────────────────────────────────────────────
st.header("Set Your Account Balance")
new_balance = st.number_input("Total Account Balance (USD)",
                               value=st.session_state.account_balance, min_value=0.0)
if st.button("Update Balance"):
    st.session_state.account_balance = new_balance
    st.success(f"Balance updated to ${new_balance:,.2f}")

# ── Section 2: Add stocks ─────────────────────────────────────────────────────
st.header("Search and Add Stocks to Portfolio")

# Form so the API call fires ONLY on Enter/Submit, never on every keystroke.
with st.form("search_form", clear_on_submit=False):
    search_query = st.text_input("Enter any valid Stock Ticker (e.g., TSLA, AAPL, 9988.HK)")
    submitted = st.form_submit_button("Search")

if submitted and search_query:
    ticker_upper = search_query.upper().strip()
    st.session_state['searched_ticker'] = ticker_upper
    st.session_state['searched_price']  = fetch_current_price(ticker_upper)

# Show results — persists across reruns via session state
if st.session_state.get('searched_ticker'):
    ticker_upper  = st.session_state['searched_ticker']
    current_price = st.session_state.get('searched_price', float('nan'))

    if not np.isnan(current_price):
        st.write(f"**{ticker_upper}** — Current Price: **${current_price:,.2f}**")
        amount_to_invest = st.number_input(
            f"Amount to Invest in {ticker_upper} (USD)",
            min_value=0.0,
            max_value=float(st.session_state.account_balance),
            key="invest_amount",
        )
        if amount_to_invest > 0:
            max_shares = int(amount_to_invest / current_price)
            st.write(f"Max Shares You Can Buy: {max_shares}")
            shares_to_buy = st.number_input(
                "Shares to Buy", min_value=0, max_value=max_shares, key="shares_input"
            )
            if st.button(f"Add {ticker_upper} to Portfolio"):
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    st.session_state.account_balance -= cost
                    st.session_state.portfolio[ticker_upper] = {
                        'shares':    shares_to_buy,
                        'buy_price': current_price,
                        'buy_date':  datetime.datetime.now(),
                    }
                    st.session_state.pop('searched_ticker', None)
                    st.session_state.pop('searched_price',  None)
                    st.success(
                        f"Added {shares_to_buy} shares of {ticker_upper} "
                        f"at ${current_price:,.2f}. Cost: ${cost:,.2f}"
                    )
                    st.rerun()
    else:
        st.error(
            f"Could not fetch a price for '{ticker_upper}'. "
            "Please check the ticker symbol and try again."
        )
        st.session_state.pop('searched_ticker', None)
        st.session_state.pop('searched_price',  None)

# ── Derived portfolio lists (recalculated each rerun from session state) ───────
tickers    = list(st.session_state.portfolio.keys())
num_assets = len(tickers)

# Fetch historical data for portfolio tickers in ONE call (cached by tuple)
historical_data = fetch_historical_data(tuple(tickers), starts, str(ends), intervals) if tickers else pd.DataFrame()
returns_df      = historical_data.pct_change().dropna() if not historical_data.empty else pd.DataFrame()

# ── Section 3: Portfolio display ──────────────────────────────────────────────
if st.session_state.portfolio:
    # Current prices — one cached call per ticker
    current_prices_map = {t: fetch_current_price(t) for t in tickers}

    portfolio_df = pd.DataFrame.from_dict(st.session_state.portfolio, orient='index')
    portfolio_df['Current Price'] = [current_prices_map[t] for t in tickers]
    portfolio_df['Current Value'] = portfolio_df['shares'] * portfolio_df['Current Price']
    portfolio_df['Return ($)']    = (portfolio_df['Current Value']
                                     - portfolio_df['shares'] * portfolio_df['buy_price'])
    portfolio_df['Return (%)']    = (portfolio_df['Return ($)']
                                     / (portfolio_df['shares'] * portfolio_df['buy_price'])) * 100

    if not returns_df.empty:
        avg_annual = returns_df.mean() * periods_per_year * 100
        portfolio_df['Average Annual Return (%)'] = [avg_annual.get(t, np.nan) for t in tickers]
    else:
        portfolio_df['Average Annual Return (%)'] = np.nan

    current_values = portfolio_df['Current Value'].values
    total_value    = current_values.sum()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Portfolio Allocation Pie Chart")
        if total_value > 0:
            weights = current_values / total_value
            fig_pie = go.Figure(data=[go.Pie(labels=tickers, values=weights, hole=.3)])
            fig_pie.update_layout(title="Portfolio Weights by Current Value")
            st.plotly_chart(fig_pie)
        else:
            st.warning("No valid current prices for pie chart.")
    with col2:
        st.subheader("Portfolio Details")
        st.write(f"Account Balance (Cash): ${st.session_state.account_balance:,.2f}")
        st.write(f"Total Portfolio Value (Invested): ${total_value:,.2f}")
        st.dataframe(portfolio_df.style.format({
            'buy_price':                    '${:,.2f}',
            'Current Price':                '${:,.2f}',
            'Current Value':                '${:,.2f}',
            'Return ($)':                   '${:,.2f}',
            'Return (%)':                   '{:,.2f}%',
            'Average Annual Return (%)':    '{:,.2f}%',
        }), height=400)
        csv = portfolio_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Portfolio CSV", csv, "portfolio.csv", "text/csv")

# ── Section 3.5: Sell shares ──────────────────────────────────────────────────
if st.session_state.portfolio:
    st.header("Sell Shares from Portfolio")
    sell_ticker = st.selectbox("Select Stock to Sell", options=tickers)
    if sell_ticker:
        sell_price    = fetch_current_price(sell_ticker)
        owned_shares  = st.session_state.portfolio[sell_ticker]['shares']
        st.write(f"You own {owned_shares} shares of {sell_ticker}. "
                 f"Current Price: ${sell_price:,.2f}")
        shares_to_sell = st.number_input("Shares to Sell", min_value=0, max_value=owned_shares)
        if st.button(f"Sell {shares_to_sell} Shares of {sell_ticker}"):
            if shares_to_sell > 0 and not np.isnan(sell_price):
                proceeds = shares_to_sell * sell_price
                st.session_state.account_balance += proceeds
                st.session_state.portfolio[sell_ticker]['shares'] -= shares_to_sell
                if st.session_state.portfolio[sell_ticker]['shares'] <= 0:
                    del st.session_state.portfolio[sell_ticker]
                st.success(
                    f"Sold {shares_to_sell} shares of {sell_ticker} at ${sell_price:,.2f}. "
                    f"Proceeds: ${proceeds:,.2f}. New Balance: ${st.session_state.account_balance:,.2f}"
                )
            else:
                st.error("Invalid sale: check shares or price.")

# ── Helper: compute betas against cached market returns ───────────────────────
def compute_betas(returns_df, tickers, market_returns):
    betas = []
    for t in tickers:
        try:
            s = returns_df[t].reindex(market_returns.index).ffill().dropna()
            m = market_returns.reindex(s.index).squeeze()
            aligned = pd.concat([s, m], axis=1).dropna()
            if len(aligned) > 1:
                cov  = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
                beta = cov / aligned.iloc[:, 1].var()
                betas.append(beta)
            else:
                betas.append(np.nan)
        except Exception:
            betas.append(np.nan)
    return betas


# ── Section 3.1: Portfolio summary ────────────────────────────────────────────
if st.session_state.portfolio and not returns_df.empty and len(returns_df) > 252:
    st.subheader("Your Portfolio Summary: Return and Risk")

    weights_arr  = np.array([current_prices_map[t] * st.session_state.portfolio[t]['shares']
                              for t in tickers])
    total_val    = weights_arr.sum()
    weights_arr  = weights_arr / total_val if total_val > 0 else np.ones(num_assets) / num_assets

    cov_matrix   = returns_df.cov() * periods_per_year
    port_vol     = np.sqrt(np.dot(weights_arr, np.dot(cov_matrix, weights_arr)))
    port_mean    = np.dot(returns_df.mean() * periods_per_year, weights_arr)
    port_hist_r  = returns_df.dot(weights_arr)
    port_var     = np.percentile(port_hist_r, 5) * np.sqrt(periods_per_year)
    tail         = port_hist_r[port_hist_r <= np.percentile(port_hist_r, 5)]
    port_cvar    = tail.mean() * np.sqrt(periods_per_year) if len(tail) else np.nan
    port_sharpe  = (port_mean - annual_rf) / port_vol if port_vol > 0 else np.nan

    betas        = compute_betas(returns_df, tickers, market_returns)
    valid_betas  = [b for b in betas if not np.isnan(b)]
    port_beta    = np.dot(weights_arr, [b if not np.isnan(b) else 0 for b in betas])

    st.markdown(f"**Annualized Expected Return:** {port_mean:.2%}")
    st.markdown(f"**Annualized Volatility (Risk):** {port_vol:.2%}")
    st.markdown(f"**Portfolio Beta:** {port_beta:.2f}" if not np.isnan(port_beta) else "**Portfolio Beta:** N/A")
    st.markdown(f"**Sharpe Ratio:** {port_sharpe:.2f}"  if not np.isnan(port_sharpe) else "**Sharpe Ratio:** N/A")
    st.markdown(f"**95% Annualized VaR:** {port_var:.2%}")
    st.markdown(f"**95% Annualized CVaR:** {port_cvar:.2%}" if not np.isnan(port_cvar) else "**95% Annualized CVaR:** N/A")

    if port_vol > 0.20:
        st.warning("High Risk Alert: Annualized volatility exceeds 20%. Consider diversifying.")
    if not np.isnan(port_sharpe) and port_sharpe < 0.5:
        st.warning("Low Efficiency Alert: Sharpe ratio below 0.5. Optimize further.")

# ── Section 4: Performance graphs ─────────────────────────────────────────────
if st.session_state.portfolio:
    st.header("Performance Graphs (Optional)")
    if st.checkbox("Show Historical Cumulative Returns (Each Stock and Portfolio)"):
        st.info("Backtests current weights historically for risk insight — not actual performance.")
        if not historical_data.empty:
            hist_returns_joint = historical_data.pct_change().dropna(how='any')
            if not hist_returns_joint.empty:
                fig_merged = plt.figure(figsize=(15, 7), dpi=100)
                colors = plt.cm.tab10(np.linspace(0, 1, num_assets))
                for i, ticker in enumerate(tickers):
                    prices = historical_data[ticker].dropna()
                    if len(prices) > 1:
                        cum = (1 + prices.pct_change().dropna()).cumprod() - 1
                        plt.plot(cum.index, cum * 100, color=colors[i], label=ticker)
                if total_value > 0:
                    w = [current_prices_map[t] * st.session_state.portfolio[t]['shares'] / total_value
                         for t in tickers]
                    port_cum = (1 + (hist_returns_joint * w).sum(axis=1)).cumprod() - 1
                    plt.plot(port_cum.index, port_cum * 100, 'g--', linewidth=2.5,
                             label='Portfolio (Weighted)')
                plt.xlabel("Time", fontsize=14)
                plt.ylabel("Cumulative Return (%)", fontsize=14)
                plt.legend(fontsize=12, loc='upper left')
                plt.grid(True)
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                plt.gca().xaxis.set_major_locator(mdates.YearLocator())
                plt.tight_layout(pad=2.0)
                st.pyplot(fig_merged)
                plt.close(fig_merged)

# ── Section 5: Risk analysis ──────────────────────────────────────────────────
if st.session_state.portfolio:
    st.header("Risk Analysis")
    if not returns_df.empty and len(returns_df) > 252:
        weights_arr = np.array([current_prices_map[t] * st.session_state.portfolio[t]['shares']
                                 for t in tickers])
        total_val   = weights_arr.sum()
        if total_val > 0:
            weights_arr = weights_arr / total_val
            betas = compute_betas(returns_df, tickers, market_returns)

            # Individual metrics
            st.subheader("Return and Risk Metrics for Each Stock")
            risk_data = []
            for i, ticker in enumerate(tickers):
                sr = returns_df[ticker].dropna().values
                if len(sr) > 1:
                    ann_ret  = np.mean(sr) * periods_per_year * 100
                    vol      = np.std(sr) * np.sqrt(periods_per_year) * 100
                    var95    = np.percentile(sr, 5) * np.sqrt(periods_per_year) * 100
                    cvar95   = sr[sr <= np.percentile(sr, 5)].mean() * np.sqrt(periods_per_year) * 100
                    beta     = betas[i]
                    risk_data.append({
                        'Ticker':                    ticker,
                        'Annualized Return (%)':     f"{ann_ret:.2f}%",
                        'Annualized Volatility (%)': f"{vol:.2f}%",
                        'Beta':                      f"{beta:.2f}" if not np.isnan(beta) else "N/A",
                        '95% Annualized VaR (%)':    f"{var95:.2f}%",
                        '95% Annualized CVaR (%)':   f"{cvar95:.2f}%",
                    })
                else:
                    risk_data.append({'Ticker': ticker, **{k: "N/A" for k in
                        ['Annualized Return (%)', 'Annualized Volatility (%)',
                         'Beta', '95% Annualized VaR (%)', '95% Annualized CVaR (%)']}})
            st.dataframe(pd.DataFrame(risk_data))

            # Correlation heatmap
            st.subheader("Correlations Between Stocks (Heatmap)")
            corr_matrix = returns_df.corr()
            fig_corr = px.imshow(corr_matrix.values, x=tickers, y=tickers,
                                  text_auto='.2f', color_continuous_scale='RdBu_r')
            fig_corr.update_layout(title="Correlation Matrix (Red=High, Blue=Low/Negative)")
            st.plotly_chart(fig_corr)

            # Portfolio risk metrics
            st.subheader("Portfolio Risk Metrics")
            cov_ann     = returns_df.cov() * periods_per_year
            port_vol    = np.sqrt(np.dot(weights_arr, np.dot(cov_ann, weights_arr)))
            port_mean   = np.dot(returns_df.mean() * periods_per_year, weights_arr)
            port_hr     = returns_df.dot(weights_arr)
            port_var_h  = np.percentile(port_hr, 5) * np.sqrt(periods_per_year)
            tail        = port_hr[port_hr <= np.percentile(port_hr, 5)]
            port_cvar_h = tail.mean() * np.sqrt(periods_per_year) if len(tail) else np.nan
            z           = stats.norm.ppf(0.95)
            port_vol_d  = np.sqrt(np.dot(weights_arr, np.dot(returns_df.cov(), weights_arr)))
            port_mean_d = returns_df.dot(weights_arr).mean()
            port_var_p  = (port_mean_d - z * port_vol_d) * np.sqrt(periods_per_year)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Annualized Portfolio Volatility:** {port_vol:.2%}")
                st.markdown(f"**Annualized 95% Parametric VaR:** {port_var_p:.2%}")
                st.markdown(f"**Annualized 95% Historical VaR:** {port_var_h:.2%}")
                st.markdown(f"**Annualized 95% CVaR:** {port_cvar_h:.2%}"
                            if not np.isnan(port_cvar_h) else "**CVaR:** N/A")
            with col2:
                st.markdown(f"**Annualized Portfolio Return:** {port_mean*100:.2f}%")
                for rd in risk_data:
                    st.markdown(f"**{rd['Ticker']} Annualized Return:** {rd['Annualized Return (%)']}")

            # Volatility bar chart
            st.subheader("Risk Visualization")
            stock_vols = [np.std(returns_df[t].dropna()) * np.sqrt(periods_per_year) for t in tickers]
            fig_risk = plt.figure(figsize=(15, 7), dpi=100)
            plt.bar(tickers + ['Portfolio'], stock_vols + [port_vol], color='orange')
            plt.xlabel("Assets", fontsize=14)
            plt.ylabel("Annualized Volatility", fontsize=14)
            plt.title("Annualized Volatility Comparison")
            plt.grid(True)
            st.pyplot(fig_risk)
            plt.close(fig_risk)

            # VaR band
            port_cum = (1 + port_hr).cumprod() - 1
            fig_band = plt.figure(figsize=(15, 7), dpi=100)
            plt.plot(returns_df.index, port_cum * 100, 'b', label='Historical Portfolio Return')
            upper = np.full(len(port_cum), 100 * (1 - port_var_h))
            lower = np.full(len(port_cum), 100 * (1 + port_var_h))
            plt.fill_between(returns_df.index, lower, upper, color='red', alpha=0.2,
                             label='Annualized VaR Risk Band')
            plt.xlabel("Time", fontsize=14)
            plt.ylabel("Cumulative Return (%)", fontsize=14)
            plt.legend(fontsize=14)
            plt.grid(True)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
            plt.tight_layout(pad=2.0)
            st.pyplot(fig_band)
            plt.close(fig_band)
        else:
            st.warning("Total portfolio value is zero — cannot compute weights.")
    else:
        st.warning("Insufficient historical data (need at least 1 year).")
else:
    st.info("Add stocks to portfolio to see risk analysis.")

# ── Scenario analysis ─────────────────────────────────────────────────────────
if st.session_state.portfolio:
    st.header("Scenario Analysis (What-If)")
    market_drop = st.number_input("Hypothetical Market Drop (%)", value=-10.0, step=0.5)
    if 'port_beta' in locals() and not np.isnan(port_beta):
        port_drop = port_beta * (market_drop / 100)
        st.markdown(
            f"Estimated Portfolio Loss: {port_drop:.2%} "
            f"(Based on beta; actual may vary due to idiosyncratic risks.)"
        )
    elif not returns_df.empty and len(returns_df) > 252:
        weights_arr  = np.array([current_prices_map[t] * st.session_state.portfolio[t]['shares']
                                  for t in tickers])
        total_val    = weights_arr.sum()
        weights_arr  = weights_arr / total_val if total_val > 0 else np.ones(num_assets) / num_assets
        betas_sc     = compute_betas(returns_df, tickers, market_returns)
        port_beta_sc = np.dot(weights_arr, [b if not np.isnan(b) else 0 for b in betas_sc])
        port_drop    = port_beta_sc * (market_drop / 100)
        st.markdown(f"Estimated Portfolio Loss: {port_drop:.2%}")
    else:
        st.warning("Cannot compute scenario (beta unavailable).")

# ── Stress testing ────────────────────────────────────────────────────────────
if st.session_state.portfolio:
    st.header("Stress Testing (Historical Events)")
    stress_periods = {
        '2008 Financial Crisis': ('2008-01-01', '2008-12-31'),
        'COVID-19 Crash':        ('2020-02-01', '2020-03-31'),
    }
    selected_stress = st.selectbox("Select Stress Event", list(stress_periods.keys()))
    if selected_stress:
        start_s, end_s = stress_periods[selected_stress]
        # Slice from already-cached historical_data (no new API call)
        if not historical_data.empty:
            stress_slice = historical_data.loc[start_s:end_s].dropna(how='all')
            valid_tickers_s = [t for t in tickers
                                if t in stress_slice.columns
                                and stress_slice[t].dropna().shape[0] > 1]
            if valid_tickers_s:
                stress_returns = stress_slice[valid_tickers_s].pct_change().dropna(how='any')
                weights_arr = np.array([current_prices_map[t] * st.session_state.portfolio[t]['shares']
                                         for t in valid_tickers_s])
                w_sum = weights_arr.sum()
                if w_sum > 0 and not stress_returns.empty:
                    valid_weights = weights_arr / w_sum
                    stress_port_return = np.dot(stress_returns.mean(), valid_weights) * len(stress_returns)
                    st.markdown(
                        f"Estimated Portfolio Return During **{selected_stress}**: "
                        f"{stress_port_return:.2%} (applied current weights to historical returns)"
                    )
                else:
                    st.warning("Insufficient data for stress test.")
            else:
                st.warning(
                    f"No stocks in your portfolio had data during {selected_stress} "
                    f"(e.g., listed after that period)."
                )
        else:
            st.warning("No historical data loaded yet.")

# ── Monte Carlo simulation ─────────────────────────────────────────────────────
if st.session_state.portfolio and st.checkbox("Run Monte Carlo Simulation (Future Projections)"):
    st.header("Monte Carlo Simulation (Predict Future Risks)")
    if not returns_df.empty:
        weights_arr = np.array([current_prices_map[t] * st.session_state.portfolio[t]['shares']
                                 for t in tickers])
        total_val   = weights_arr.sum()
        weights_arr = weights_arr / total_val if total_val > 0 else np.ones(num_assets) / num_assets

        num_sim     = 1000
        sim_periods = 252
        mean_daily  = returns_df.mean().values
        cov_daily   = returns_df.cov().values
        sim_returns = np.random.multivariate_normal(mean_daily, cov_daily, (num_sim, sim_periods))
        sim_port    = np.dot(sim_returns, weights_arr)
        sim_cum     = np.cumprod(1 + sim_port, axis=1) - 1
        final_r     = sim_cum[:, -1]
        sim_var     = np.percentile(final_r, 5)
        fig_hist    = px.histogram(final_r, nbins=50,
                                   title="Distribution of 1-Year Future Returns (1 000 Simulations)")
        st.plotly_chart(fig_hist)
        st.markdown(
            f"**95% VaR from Simulation (1-Year Loss Potential):** {sim_var:.2%} "
            f"(assumes historical patterns continue)"
        )
    else:
        st.warning("Cannot run simulation (data unavailable).")

# ── Section 6: Portfolio optimisation ─────────────────────────────────────────
if st.session_state.portfolio and num_assets >= 2 and not returns_df.empty:
    st.header("Portfolio Optimization")

    weights_arr = np.array([current_prices_map[t] * st.session_state.portfolio[t]['shares']
                             for t in tickers])
    total_val   = weights_arr.sum()
    weights_arr = weights_arr / total_val if total_val > 0 else np.ones(num_assets) / num_assets

    mean_returns = returns_df.mean().values * periods_per_year
    cov_matrix   = returns_df.cov().values  * periods_per_year

    def port_perf(w):
        return np.dot(w, mean_returns), np.sqrt(np.dot(w, np.dot(cov_matrix, w)))

    def neg_sharpe(w):
        r, v = port_perf(w)
        return -(r - annual_rf) / v if v > 0 else np.inf

    def min_vol_fn(w):
        return port_perf(w)[1]

    cons      = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
    bnds      = tuple((0, 1) for _ in range(num_assets))
    init      = np.ones(num_assets) / num_assets

    opt_sharpe   = sco.minimize(neg_sharpe,  init, method='SLSQP', bounds=bnds, constraints=cons)
    opt_min_risk = sco.minimize(min_vol_fn,  init, method='SLSQP', bounds=bnds, constraints=cons)

    if opt_sharpe.success:
        ow_s    = opt_sharpe.x
        or_s, ov_s = port_perf(ow_s)
        osr_s   = -opt_sharpe.fun
    else:
        ow_s = or_s = ov_s = osr_s = None

    if opt_min_risk.success:
        ow_r    = opt_min_risk.x
        or_r, ov_r = port_perf(ow_r)
    else:
        ow_r = or_r = ov_r = None

    st.subheader("Optimized Allocations")
    if ow_s is not None:
        st.write("**Max Sharpe Ratio Portfolio (Balanced Return/Risk):**")
        st.dataframe(pd.DataFrame({'Ticker': tickers,
                                   'Optimal Weight': [f"{w:.2%}" for w in ow_s]}))
        st.markdown(f"Expected Annual Return: {or_s:.4%} | Annual Risk: {ov_s:.4%} | Sharpe: {osr_s:.2f}")
    if ow_r is not None:
        st.write("**Minimum Risk Portfolio (Lowest Volatility):**")
        st.dataframe(pd.DataFrame({'Ticker': tickers,
                                   'Optimal Weight': [f"{w:.2%}" for w in ow_r]}))
        st.markdown(f"Expected Annual Return: {or_r:.4%} | Annual Risk: {ov_r:.4%}")

    # Efficient frontier
    if ow_s is not None and ow_r is not None:
        st.subheader("Efficient Frontier with CAL and Risk-Free Rate")
        target_rets  = np.linspace(or_r, mean_returns.max(), 50)
        frontier_vols = []
        for tr in target_rets:
            c = cons + ({'type': 'eq', 'fun': lambda w, tr=tr: port_perf(w)[0] - tr},)
            res = sco.minimize(min_vol_fn, init, method='SLSQP', bounds=bnds, constraints=c)
            frontier_vols.append(res.fun if res.success else np.nan)

        stock_vols_diag = np.sqrt(np.diag(cov_matrix))
        cal_x = np.linspace(0, ov_s * 1.5, 50)
        cal_y = annual_rf + cal_x * osr_s

        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(x=frontier_vols, y=target_rets,
                                    mode='lines', name='Efficient Frontier'))
        fig_ef.add_trace(go.Scatter(x=[ov_s], y=[or_s], mode='markers', name='Max Sharpe'))
        fig_ef.add_trace(go.Scatter(x=[ov_r], y=[or_r], mode='markers', name='Min Risk'))
        fig_ef.add_trace(go.Scatter(x=[0], y=[annual_rf], mode='markers',
                                    name='Risk-Free Rate', marker=dict(color='green', size=10)))
        fig_ef.add_trace(go.Scatter(x=cal_x, y=cal_y, mode='lines',
                                    name='CAL', line=dict(dash='dash')))
        fig_ef.add_trace(go.Scatter(x=stock_vols_diag, y=mean_returns,
                                    mode='markers+text', text=tickers, textposition='top center',
                                    name='Individual Stocks', marker=dict(color='purple', size=10)))
        fig_ef.update_layout(xaxis_title='Annual Volatility',
                             yaxis_title='Expected Annual Return',
                             title='Efficient Frontier with CAL')
        st.plotly_chart(fig_ef)

    # SML — uses the already-fetched (cached) market_returns / market_mean
    betas = compute_betas(returns_df, tickers, market_returns)
    valid_idx   = [i for i, b in enumerate(betas) if not np.isnan(b)]
    valid_betas = [betas[i] for i in valid_idx]
    valid_tks   = [tickers[i] for i in valid_idx]
    valid_rets  = [mean_returns[i] for i in valid_idx]

    if valid_betas:
        st.subheader("Security Market Line (SML)")
        x_sml = [0, max(valid_betas) + 0.5]
        y_sml = [annual_rf, annual_rf + x_sml[1] * (market_mean - annual_rf)]
        fig_sml = go.Figure()
        fig_sml.add_trace(go.Scatter(x=valid_betas, y=valid_rets,
                                     mode='markers+text', text=valid_tks, name='Stocks'))
        fig_sml.add_trace(go.Scatter(x=x_sml, y=y_sml, mode='lines', name='SML'))
        fig_sml.update_layout(xaxis_title='Beta', yaxis_title='Expected Annual Return',
                              title='Security Market Line')
        st.plotly_chart(fig_sml)

    # Sharpe comparison
    port_vol_cur  = np.sqrt(np.dot(weights_arr, np.dot(cov_matrix, weights_arr)))
    port_mean_cur = np.dot(weights_arr, mean_returns)
    current_sharpe = (port_mean_cur - annual_rf) / port_vol_cur if port_vol_cur > 0 else np.nan
    market_sharpe  = (market_mean - annual_rf) / market_vol if market_vol > 0 else np.nan
    st.markdown(f"**Current Portfolio Sharpe Ratio:** {current_sharpe:.2f}"
                if not np.isnan(current_sharpe) else "**Current Portfolio Sharpe Ratio:** N/A")
    if ow_s is not None:
        st.markdown(f"**Optimized Max Sharpe Ratio:** {osr_s:.2f}")
    st.markdown(f"**Market Sharpe Ratio:** {market_sharpe:.2f}"
                if not np.isnan(market_sharpe) else "**Market Sharpe Ratio:** N/A")

# ── Section 7: Custom complete portfolio (MPT) ────────────────────────────────
if st.session_state.portfolio and num_assets >= 2 and not returns_df.empty:
    st.header("Custom Complete Portfolio (Modern Portfolio Theory)")
    st.markdown("""
    This section applies Modern Portfolio Theory (MPT). You set a target return or risk level,
    and the app allocates between the risk-free asset (5% annual) and the tangency portfolio
    (max Sharpe ratio risky portfolio). All metrics are annualized.
    """)

    allow_short = st.checkbox("Allow Short Selling (and Leverage/Borrowing)", value=False)
    input_type  = st.selectbox("Select Input Type", ["Expected Return", "Risk Level (Volatility)"])
    if input_type == "Expected Return":
        target = st.number_input("Desired Annual Expected Return (e.g., 0.12 for 12%)",
                                  value=0.12, step=0.01)
    else:
        target = st.number_input("Desired Annual Volatility (e.g., 0.15 for 15%)",
                                  value=0.15, min_value=0.0, step=0.01)

    mean_returns_mpt = returns_df.mean().values * periods_per_year
    cov_matrix_mpt   = returns_df.cov().values  * periods_per_year

    def port_perf_mpt(w):
        return np.dot(w, mean_returns_mpt), np.sqrt(np.dot(w, np.dot(cov_matrix_mpt, w)))

    def neg_sharpe_mpt(w):
        r, v = port_perf_mpt(w)
        return -(r - annual_rf) / v if v > 0 else np.inf

    def min_vol_mpt(w):
        return port_perf_mpt(w)[1]

    cons_mpt = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
    bnds_mpt = None if allow_short else tuple((0, 1) for _ in range(num_assets))
    init_mpt = np.ones(num_assets) / num_assets

    opt_t = sco.minimize(neg_sharpe_mpt, init_mpt, method='SLSQP',
                         bounds=bnds_mpt, constraints=cons_mpt)
    if not opt_t.success:
        st.error("Tangency portfolio optimisation failed.")
    else:
        opt_w   = opt_t.x
        opt_ret, opt_vol = port_perf_mpt(opt_w)
        opt_sr  = (opt_ret - annual_rf) / opt_vol if opt_vol > 0 else np.nan

        opt_mr  = sco.minimize(min_vol_mpt, init_mpt, method='SLSQP',
                               bounds=bnds_mpt, constraints=cons_mpt)
        min_r_w = opt_mr.x if opt_mr.success else None
        min_r, min_v = port_perf_mpt(min_r_w) if min_r_w is not None else (np.nan, np.nan)

        max_ret_idx = int(np.argmax(mean_returns_mpt))
        max_ret_w   = np.zeros(num_assets); max_ret_w[max_ret_idx] = 1
        max_r, max_v = port_perf_mpt(max_ret_w)

        if np.isnan(opt_ret) or opt_vol == 0 or opt_ret <= annual_rf:
            st.warning("Invalid tangency portfolio. Cannot compute complete portfolio.")
        else:
            complete_weights = None
            if input_type == "Expected Return":
                alpha = (target - annual_rf) / (opt_ret - annual_rf)
                if not allow_short and not (0 <= alpha <= 1):
                    st.warning(f"Cannot achieve {target:.4%} without short/leverage.")
                    st.info(f"Max complete (100% tangency): Return {opt_ret:.4%}, Vol {opt_vol:.4%}")
                    st.info(f"Min risk risky: Return {min_r:.4%}, Vol {min_v:.4%}")
                    st.info(f"All risk-free: Return {annual_rf:.4%}, Vol 0.00%")
                else:
                    complete_ret = target
                    complete_vol = abs(alpha) * opt_vol
                    complete_weights = {'^TNX (Risk-Free)': 1 - alpha}
                    for i, t in enumerate(tickers):
                        complete_weights[t] = alpha * opt_w[i]
            else:
                alpha = target / opt_vol if opt_vol > 0 else 0
                if not allow_short and alpha > 1:
                    st.warning(f"Cannot achieve vol {target:.4%} without leverage.")
                    st.info(f"Max complete (100% tangency): Vol {opt_vol:.4%}, Return {opt_ret:.4%}")
                    st.info(f"Min risk risky: Vol {min_v:.4%}, Return {min_r:.4%}")
                    st.info(f"All risk-free: Vol 0.00%, Return {annual_rf:.4%}")
                else:
                    complete_vol = target
                    complete_ret = annual_rf + alpha * (opt_ret - annual_rf)
                    complete_weights = {'^TNX (Risk-Free)': 1 - alpha}
                    for i, t in enumerate(tickers):
                        complete_weights[t] = alpha * opt_w[i]

            if complete_weights is not None:
                st.subheader("Complete Portfolio Details")
                cdf = pd.DataFrame(list(complete_weights.items()), columns=['Asset', 'Weight'])
                cdf['Weight'] = cdf['Weight'].apply(lambda w: f"{w:.2%}")
                st.dataframe(cdf)
                st.markdown(f"**Annual Expected Return:** {complete_ret:.4%}")
                st.markdown(f"**Annual Volatility:** {complete_vol:.4%}")
                c_sharpe = (complete_ret - annual_rf) / complete_vol if complete_vol > 0 else np.nan
                st.markdown(f"**Sharpe Ratio:** {c_sharpe:.2f}"
                            if not np.isnan(c_sharpe) else "**Sharpe Ratio:** N/A (zero risk)")
                st.markdown(
                    "**Note:** Negative weights = short positions; "
                    "risk-free weight < 0 = leveraged (borrowed)."
                )
                # CAL chart
                cal_x = np.linspace(0, max(complete_vol, opt_vol) * 1.5, 50)
                cal_y = annual_rf + (opt_ret - annual_rf) / opt_vol * cal_x if opt_vol > 0 else np.full_like(cal_x, annual_rf)
                fig_cal = go.Figure()
                fig_cal.add_trace(go.Scatter(x=[0], y=[annual_rf], mode='markers',
                                             name='Risk-Free', marker=dict(color='green', size=10)))
                fig_cal.add_trace(go.Scatter(x=cal_x, y=cal_y, mode='lines', name='CAL'))
                fig_cal.add_trace(go.Scatter(x=[opt_vol], y=[opt_ret], mode='markers',
                                             name='Tangency Portfolio'))
                fig_cal.add_trace(go.Scatter(x=[complete_vol], y=[complete_ret], mode='markers',
                                             name='Your Complete Portfolio',
                                             marker=dict(color='red', size=12)))
                fig_cal.update_layout(xaxis_title='Annual Volatility',
                                      yaxis_title='Expected Annual Return',
                                      title='Your Portfolio on the CAL')
                st.plotly_chart(fig_cal)

# ── Refresh button ─────────────────────────────────────────────────────────────
if st.button("Refresh Data (Update Prices & Graphs)"):
    st.cache_data.clear()   # force all cached fetches to re-run
    st.rerun()
