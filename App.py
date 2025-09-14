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

# Define global parameters
intervals = '1d'  # Changed to daily for better predictions/betas (resample to monthly where needed)
periods_per_year = 252  # Trading days for annualization (more accurate)
rf_ticker = '^TNX'  # 10-year Treasury yield (risk-free proxy); fallback to BND if fails
starts = '2015-01-01'  # Start date for historical data
ends = datetime.date.today()  # End date as current day
num_assets = 0  # Will be set based on number of stocks in portfolio

# Cache historical data with Close (daily)
@st.cache_data(ttl=300)
def fetch_historical_data(tickers, starts, ends, intervals):
    if not tickers:
        return pd.DataFrame()
    data = yf.download(tickers, start=starts, end=ends, interval=intervals)
    close = data.xs('Close', level=0, axis=1)
    return close[tickers]  # Order as in tickers

# Fetch risk-free rate (try Treasury, fallback BND mean monthly return)
@st.cache_data(ttl=300)
def fetch_rf_data(starts, ends, intervals):
    try:
        rf_data = yf.download(rf_ticker, start=starts, end=ends, interval=intervals)['Close'] / 100  
        rf_mean = rf_data.mean()  # Annual rf (yield is annual)
    except:
        rf_data = yf.download('BND', start=starts, end=ends, interval=intervals)['Close']
        rf_returns = rf_data.pct_change().dropna()
        rf_mean = rf_returns.mean() * periods_per_year  # Annualized
        if isinstance(rf_mean, pd.Series):
            rf_mean = rf_mean.iloc[0]
    return rf_mean

# Session state initialization
if 'account_balance' not in st.session_state:
    st.session_state.account_balance = 100000.0  # Default USD
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}  # {ticker: {'shares': int, 'buy_price': float, 'buy_date': datetime}}

st.title("Portfolio Management Application")

# Section 1: Set Account Balance
st.header("Set Your Account Balance")
new_balance = st.number_input("Total Account Balance (USD)", value=st.session_state.account_balance, min_value=0.0)
if st.button("Update Balance"):
    st.session_state.account_balance = new_balance
    st.success(f"Balance updated to ${new_balance:,.2f}")

# Section 2: Search and Add Stocks
st.header("Search and Add Stocks to Portfolio")
search_query = st.text_input("Search Stock Ticker (e.g., TSLA, AAPL)")
if search_query:
    try:
        stock_info = yf.Ticker(search_query).info
        current_price = stock_info.get('currentPrice', None)
        if current_price:
            st.write(f"Current Price for {search_query}: ${current_price:,.2f}")
            amount_to_invest = st.number_input(f"Amount to Invest in {search_query} (USD)", min_value=0.0, max_value=st.session_state.account_balance)
            if amount_to_invest > 0:
                max_shares = int(amount_to_invest / current_price)
                st.write(f"Max Shares You Can Buy: {max_shares}")
                shares_to_buy = st.number_input("Shares to Buy", min_value=0, max_value=max_shares)
                if st.button(f"Add {search_query} to Portfolio"):
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_price
                        st.session_state.account_balance -= cost
                        st.session_state.portfolio[search_query] = {
                            'shares': shares_to_buy,
                            'buy_price': current_price,
                            'buy_date': datetime.datetime.now()
                        }
                        st.success(f"Added {shares_to_buy} shares of {search_query} at ${current_price:,.2f} each. Cost: ${cost:,.2f}")
    except Exception as e:
        st.error(f"Error fetching data for {search_query}: {str(e)}")


tickers = list(st.session_state.portfolio.keys())
num_assets = len(tickers)

annual_rf = fetch_rf_data(starts, ends, intervals)  # Annualized

# Fetch market data once for betas, SML, etc.
market_ticker = '^GSPC'
market_data = yf.download(market_ticker, start=starts, end=ends, interval=intervals)['Close']
market_returns = market_data.pct_change().dropna()

# Section 3: Display Portfolio with Pie Chart and Details
if st.session_state.portfolio:
    # Create portfolio_df
    portfolio_df = pd.DataFrame.from_dict(st.session_state.portfolio, orient='index')
    portfolio_df['Current Price'] = [yf.Ticker(ticker).info.get('currentPrice', np.nan) for ticker in tickers]
    portfolio_df['Current Value'] = portfolio_df['shares'] * portfolio_df['Current Price']
    portfolio_df['Return ($)'] = portfolio_df['Current Value'] - (portfolio_df['shares'] * portfolio_df['buy_price'])
    portfolio_df['Return (%)'] = (portfolio_df['Return ($)'] / (portfolio_df['shares'] * portfolio_df['buy_price'])) * 100

    # Add average annual return
    historical_data = fetch_historical_data(tickers, starts, ends, intervals)
    returns_df = historical_data.pct_change().dropna()  # Daily returns
    avg_annual = returns_df.mean() * periods_per_year * 100  # Annualized
    portfolio_df['Average Annual Return (%)'] = [avg_annual.get(t, np.nan) for t in tickers]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Portfolio Allocation Pie Chart")
        current_values = portfolio_df['Current Value'].values
        total_value = current_values.sum()
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
            'buy_price': '${:,.2f}',
            'Current Price': '${:,.2f}',
            'Current Value': '${:,.2f}',
            'Return ($)': '${:,.2f}',
            'Return (%)': '{:,.2f}%',
            'Average Annual Return (%)': '{:,.2f}%'
        }), height=800)  

        # Add export
        csv = portfolio_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Portfolio CSV", csv, "portfolio.csv", "text/csv")

# Section 3.5: Sell Shares
if st.session_state.portfolio:
    st.header("Sell Shares from Portfolio")
    sell_ticker = st.selectbox("Select Stock to Sell", options=tickers)
    if sell_ticker:
        current_price = yf.Ticker(sell_ticker).info.get('currentPrice', np.nan)
        owned_shares = st.session_state.portfolio[sell_ticker]['shares']
        st.write(f"You own {owned_shares} shares of {sell_ticker}. Current Price: ${current_price:,.2f}")
        shares_to_sell = st.number_input("Shares to Sell", min_value=0, max_value=owned_shares)
        if st.button(f"Sell {shares_to_sell} Shares of {sell_ticker}"):
            if shares_to_sell > 0 and not np.isnan(current_price):
                proceeds = shares_to_sell * current_price
                st.session_state.account_balance += proceeds
                st.session_state.portfolio[sell_ticker]['shares'] -= shares_to_sell
                if st.session_state.portfolio[sell_ticker]['shares'] <= 0:
                    del st.session_state.portfolio[sell_ticker]
                st.success(f"Sold {shares_to_sell} shares of {sell_ticker} at ${current_price:,.2f} each. Proceeds: ${proceeds:,.2f}. New Balance: ${st.session_state.account_balance:,.2f}")
            else:
                st.error("Invalid sale: Check shares or price.")

# Section 3.1: Portfolio Summary (Return and Risk)
if st.session_state.portfolio:
    st.subheader("Your Portfolio Summary: Return and Risk")
    total_invested = sum(st.session_state.portfolio[t]['shares'] * st.session_state.portfolio[t]['buy_price'] for t in tickers)
    total_current = portfolio_df['Current Value'].sum()
    port_return = (total_current - total_invested) / total_invested if total_invested > 0 else 0
    st.markdown(f"**Since-Purchase Return:** {port_return:.2%} (Total Gain/Loss: ${total_current - total_invested:,.2f})")

    #risk metrics (using daily data)
    historical_data = fetch_historical_data(tickers, starts, ends, intervals)
    if not historical_data.empty and len(historical_data) > 252:  # At least 1 year data
        returns_df = historical_data.pct_change().dropna()
        if not returns_df.empty:
            current_prices = portfolio_df['Current Price'].values
            current_values = portfolio_df['Current Value'].values
            total_value = sum(current_values)
            if total_value > 0:
                weights = np.array(current_values / total_value)
                cov_matrix = returns_df.cov() * periods_per_year  # Annualized cov
                port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # Annualized 
                port_mean = np.dot(returns_df.mean() * periods_per_year, weights)  # Annualized 
                port_hist_returns = returns_df.dot(weights)
                port_var_hist = np.percentile(port_hist_returns, 5) * np.sqrt(periods_per_year)  # Approx 
                port_cvar_hist = port_hist_returns[port_hist_returns <= np.percentile(port_hist_returns, 5)].mean() * np.sqrt(periods_per_year) if len(port_hist_returns) > 0 else np.nan
                port_sharpe = (port_mean - annual_rf.item()) / port_vol if port_vol > 0 else np.nan

                # betas
                betas = []
                for t in tickers:
                    stock_returns = returns_df[t].reindex(market_returns.index).ffill().dropna()
                    aligned_market = market_returns.reindex(stock_returns.index).squeeze()  
                    if len(stock_returns) > 1:
                        beta = np.cov(stock_returns, aligned_market)[0,1] / np.var(aligned_market)
                        betas.append(beta)
                    else:
                        betas.append(np.nan)
                port_beta = np.dot(weights, [b if not np.isnan(b) else 0 for b in betas]) if betas else np.nan

                st.markdown(f"**Annualized Expected Return:** {port_mean:.2%}")
                st.markdown(f"**Annualized Volatility (Risk):** {port_vol:.2%}")
                st.markdown(f"**Portfolio Beta (Systematic Risk):** {port_beta:.2f}" if not np.isnan(port_beta) else "**Portfolio Beta:** N/A")
                st.markdown(f"**Sharpe Ratio (Return per Risk):** {port_sharpe:.2f}" if not np.isnan(port_sharpe) else "**Sharpe Ratio:** N/A")
                st.markdown(f"**95% Annualized VaR (Potential Loss, Approx):** {port_var_hist:.2%}")
                st.markdown(f"**95% Annualized CVaR (Worst-Case Avg Loss, Approx):** {port_cvar_hist:.2%}")

                # Alerts
                if port_vol > 0.20:
                    st.warning("High Risk Alert: Annualized volatility exceeds 20% (too risky for conservative investors). Consider diversifying.")
                if port_sharpe < 0.5 and not np.isnan(port_sharpe):
                    st.warning("Low Efficiency Alert: Sharpe ratio below 0.5 (poor return for the risk). Optimize further.")
            else:
                st.warning("Unable to compute risks (no valid values).")
        else:
            st.warning("No historical returns for risk computation.")
    else:
        st.warning("Insufficient historical data (need at least 1 year for reliable metrics).")

# Section 4: Optional Performance Graphs
if st.session_state.portfolio:
    st.header("Performance Graphs (Optional)")
    if st.checkbox("Show Historical Cumulative Returns (Each Stock and Portfolio)"):
        st.info("This backtests your current weights historically for risk insights—not actual performance.")
        if not historical_data.empty:
            hist_returns = historical_data.pct_change().dropna()
            if not hist_returns.empty:
                fig_merged = plt.figure(figsize=(15, 7), dpi=100)
                colors = plt.cm.tab10(np.linspace(0, 1, num_assets))
                for i, ticker in enumerate(tickers):
                    stock_cum_returns = (1 + hist_returns[ticker]).cumprod() - 1
                    plt.plot(historical_data.index[1:], stock_cum_returns * 100, color=colors[i], label=ticker)
                current_values = portfolio_df['Current Value'].values
                total_current_value = sum(current_values)
                if total_current_value > 0:
                    weights = [current_values[i] / total_current_value for i in range(len(current_values))]
                    port_hist_daily_returns = (hist_returns * weights).sum(axis=1)
                    port_hist_cum_returns = (1 + port_hist_daily_returns).cumprod() - 1
                    plt.plot(historical_data.index[1:], port_hist_cum_returns * 100, 'g--', linewidth=2.5, label='Portfolio (Weighted)')
                plt.xlabel("Time", fontsize=14)
                plt.ylabel("Cumulative Return (%)", fontsize=14)
                plt.legend(fontsize=14, loc='upper left')
                plt.grid(True)
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                plt.gca().xaxis.set_major_locator(mdates.YearLocator())
                plt.tight_layout(pad=2.0)
                st.pyplot(fig_merged)
                plt.close(fig_merged)
            else:
                st.warning("No historical return data available.")
        else:
            st.warning("Insufficient historical data for plotting.")
    if st.checkbox("Show Portfolio Return Since Purchase (Primary View)", value=True):
        try:
            earliest_buy = min(info['buy_date'] for info in st.session_state.portfolio.values())
            port_data_raw = yf.download(tickers, start=earliest_buy, end=ends, interval=intervals)
            port_data = port_data_raw.xs('Close', level=0, axis=1)[tickers]  # Order as tickers
            if not port_data.empty and len(port_data) > 1:
                weights = portfolio_df['Current Value'].values / portfolio_df['Current Value'].sum()
                port_returns = port_data.pct_change().dropna()
                if not port_returns.empty:
                    port_daily_returns = (port_returns * weights).sum(axis=1)
                    port_cum_returns = (1 + port_daily_returns).cumprod() - 1
                    if not port_cum_returns.empty:
                        fig_port = plt.figure(figsize=(15, 7), dpi=100)
                        plt.plot(port_data.index[:len(port_cum_returns)], port_cum_returns * 100, 'b', label='Portfolio Cumulative Return')
                        plt.xlabel("Time", fontsize=14)
                        plt.ylabel("Cumulative Return (%)", fontsize=14)
                        plt.legend(fontsize=14)
                        plt.grid(True)
                        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
                        plt.tight_layout(pad=2.0)
                        st.pyplot(fig_port)
                        plt.close(fig_port)
                    else:
                        st.warning("Cumulative returns empty.")
                else:
                    st.warning("No return data since purchase.")
            else:
                st.warning("Insufficient data since purchase.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Section 5: Risk Analysis
if st.session_state.portfolio:
    st.header("Risk Analysis")
    try:
        historical_data = fetch_historical_data(tickers, starts, ends, intervals)
        if not historical_data.empty and len(historical_data) > 252:
            returns_df = historical_data.pct_change().dropna()
            if not returns_df.empty:
                current_prices = [yf.Ticker(t).info.get('currentPrice', np.nan) for t in tickers]
                current_values = [st.session_state.portfolio[t]['shares'] * current_prices[i] if not np.isnan(current_prices[i]) else 0 for i, t in enumerate(tickers)]
                total_value = sum(current_values)
                if total_value > 0:
                    weights = np.array([v / total_value for v in current_values])

                    # Portfolio mean for use
                    port_mean = np.dot(returns_df.mean() * periods_per_year, weights)

                    # Individual Stock Metrics 
                    st.subheader("Return and Risk Metrics for Each Stock")
                    risk_data = []
                    for i, ticker in enumerate(tickers):
                        stock_returns = returns_df[ticker].dropna().values
                        if len(stock_returns) > 1:
                            annual_return = np.mean(stock_returns) * periods_per_year * 100
                            var_95 = np.percentile(stock_returns, 5) * np.sqrt(periods_per_year) * 100
                            cvar_95 = stock_returns[stock_returns <= np.percentile(stock_returns, 5)].mean() * np.sqrt(periods_per_year) * 100 if len(stock_returns) > 0 else np.nan
                            vol = np.std(stock_returns) * np.sqrt(periods_per_year) * 100
                            beta = betas[i] if 'betas' in locals() else np.nan
                            risk_data.append({
                                'Ticker': ticker,
                                'Annualized Return (%)': f"{annual_return:.2f}%",
                                'Annualized Volatility (%)': f"{vol:.2f}%",
                                'Beta': f"{beta:.2f}" if not np.isnan(beta) else "N/A",
                                '95% Annualized VaR (%)': f"{var_95:.2f}%" if not np.isnan(var_95) else "N/A",
                                '95% Annualized CVaR (%)': f"{cvar_95:.2f}%" if not np.isnan(cvar_95) else "N/A"
                            })
                        else:
                            risk_data.append({
                                'Ticker': ticker,
                                'Annualized Return (%)': "N/A",
                                'Annualized Volatility (%)': "N/A",
                                'Beta': "N/A",
                                '95% Annualized VaR (%)': "N/A",
                                '95% Annualized CVaR (%)': "N/A"
                            })
                    risk_df = pd.DataFrame(risk_data)
                    st.dataframe(risk_df, height=800)  # Wider row height

                    # Add Correlations Heatmap
                    st.subheader("Correlations Between Stocks (Heatmap)")
                    corr_matrix = returns_df.corr()
                    fig_corr = px.imshow(corr_matrix.values, x=tickers, y=tickers, text_auto='.2f', color_continuous_scale='RdBu_r')
                    fig_corr.update_layout(title="Correlation Matrix (Red=High, Blue=Low/Negative)")
                    st.plotly_chart(fig_corr)
                   
                    # Portfolio Risk Metrics
                    st.subheader("Portfolio Risk Metrics")
                    cov_matrix_monthly = returns_df.cov()
                    cov_matrix = cov_matrix_monthly * periods_per_year  # Annualized covariance
                    port_vol_monthly = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_monthly, weights)))
                    port_vol = port_vol_monthly * np.sqrt(periods_per_year)
                    port_mean_monthly = np.mean(returns_df.dot(weights))
                    port_mean = port_mean_monthly * periods_per_year
                    z = stats.norm.ppf(0.95)
                    port_var_param_monthly = port_mean_monthly - z * port_vol_monthly
                    port_var_param = port_var_param_monthly * np.sqrt(periods_per_year)
                    port_hist_returns_monthly = returns_df.dot(weights)
                    port_var_hist_monthly = np.percentile(port_hist_returns_monthly, 5)
                    port_var_hist = port_var_hist_monthly * np.sqrt(periods_per_year)
                    port_cvar_hist_monthly = port_hist_returns_monthly[port_hist_returns_monthly <= port_var_hist_monthly].mean() if len(port_hist_returns_monthly[port_hist_returns_monthly <= port_var_hist_monthly]) > 0 else np.nan
                    port_cvar_hist = port_cvar_hist_monthly * np.sqrt(periods_per_year)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Annualized Portfolio Volatility:** {port_vol:.2%}")
                        st.markdown(f"**Annualized Portfolio 95% Parametric VaR:** {port_var_param:.2%}")
                        st.markdown(f"**Annualized Portfolio 95% Historical VaR:** {port_var_hist:.2%}")
                        st.markdown(f"**Annualized Portfolio 95% CVaR:** {port_cvar_hist:.2%}")
                    with col2:
                        st.markdown(f"**Annualized Portfolio Return:** {port_mean*100:.2f}%")
                        for rd in risk_data:
                            ticker = rd['Ticker']
                            ret = rd['Annualized Return (%)']
                            st.markdown(f"**{ticker} Annualized Return:** {ret}")

                    # Risk Visualization
                    st.subheader("Risk Visualization")
                    fig_risk = plt.figure(figsize=(15, 7), dpi=100)
                    stock_vols_monthly = [np.std(returns_df[t].dropna()) for t in tickers]
                    stock_vols = [v * np.sqrt(periods_per_year) for v in stock_vols_monthly]
                    plt.bar(tickers + ['Portfolio'], stock_vols + [port_vol], color='orange')
                    plt.xlabel("Assets", fontsize=14)
                    plt.ylabel("Annualized Volatility (%)", fontsize=14)
                    plt.title("Annualized Volatility Comparison")
                    plt.grid(True)
                    st.pyplot(fig_risk)
                    plt.close(fig_risk)

                    # VaR Band (adjusted for annualized)
                    port_hist_returns = port_hist_returns_monthly * np.sqrt(periods_per_year)
                    port_cum_returns = (1 + port_hist_returns_monthly).cumprod() - 1  # Cumulative is total, no need to annualize
                    if not port_cum_returns.empty:
                        fig_risk_band = plt.figure(figsize=(15, 7), dpi=100)
                        plt.plot(historical_data.index[1:], port_cum_returns * 100, 'b', label='Historical Portfolio Return')
                        upper = np.full(len(port_cum_returns), 100 * (1 + port_var_hist))
                        lower = np.full(len(port_cum_returns), 100 * (1 + port_var_hist * -1))
                        plt.fill_between(historical_data.index[1:], lower, upper, color='red', alpha=0.2, label='Annualized VaR Risk Band')
                        plt.xlabel("Time", fontsize=14)
                        plt.ylabel("Cumulative Return (%)", fontsize=14)
                        plt.legend(fontsize=14)
                        plt.grid(True)
                        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
                        plt.tight_layout(pad=2.0)
                        st.pyplot(fig_risk_band)
                        plt.close(fig_risk_band)
                    else:
                        st.warning("No cumulative returns for VaR band (insufficient data).")
                else:
                    st.warning("Total portfolio value is zero (no valid prices)—cannot compute weights.")
            else:
                st.warning("No returns data for risk calculations (e.g., too few historical points).")
        else:
            st.warning("Insufficient historical data fetched (e.g., invalid range or API issue).")
    except Exception as e:
        st.error(f"Risk section error: {str(e)}. Check tickers, dates, or network.")
else:
    st.info("Add stocks to portfolio to see risk analysis.")

# Section: Scenario Analysis
if st.session_state.portfolio:
    st.header("Scenario Analysis (What-If)")
    market_drop = st.number_input("Hypothetical Market Drop (%)", value=-10.0, step=0.5)
    if 'port_beta' in locals() and not np.isnan(port_beta):
        port_drop = port_beta * (market_drop / 100)
        st.markdown(f"Estimated Portfolio Loss: {port_drop:.2%} (Based on beta; actual may vary due to idiosyncratic risks). (Plain English: Idiosyncratic risks are stock-specific risks not tied to the market, like company scandals.)")
    else:
        st.warning("Cannot compute scenario (beta unavailable).")

# Section: Stress Testing
if st.session_state.portfolio:
    st.header("Stress Testing (Historical Events)")
    stress_periods = {
        '2008 Financial Crisis': ('2008-01-01', '2008-12-31'),
        'COVID-19 Crash': ('2020-02-01', '2020-03-31')
    }
    selected_stress = st.selectbox("Select Stress Event", list(stress_periods.keys()))
    if selected_stress:
        start_s, end_s = stress_periods[selected_stress]
        stress_data = yf.download(tickers, start=start_s, end=end_s, interval=intervals)['Close']
        stress_returns = stress_data.pct_change().dropna()
        if not stress_returns.empty and 'weights' in locals():
            stress_port_return = np.dot(stress_returns.mean(), weights) * len(stress_returns)  # Total return over period
            st.markdown(f"Estimated Portfolio Return During {selected_stress}: {stress_port_return:.2%} (Applied current weights to historical returns). (Plain English: This simulates how your portfolio would have performed during past crises using today's allocation.)")
        else:
            st.warning("Insufficient data for stress test.")

# Section: Monte Carlo Simulation
if st.session_state.portfolio and st.checkbox("Run Monte Carlo Simulation (Future Projections)"):
    st.header("Monte Carlo Simulation (Predict Future Risks)")
    num_sim = 1000
    sim_periods = 252  # 1 year daily
    if 'returns_df' in locals() and 'weights' in locals() and not returns_df.empty:
        mean_daily = returns_df.mean().values
        cov_daily = returns_df.cov().values
        sim_returns = np.random.multivariate_normal(mean_daily, cov_daily, (num_sim, sim_periods))
        sim_port_returns = np.dot(sim_returns, weights)
        sim_cum_returns = np.cumprod(1 + sim_port_returns, axis=1) - 1
        final_returns = sim_cum_returns[:, -1]
        sim_var = np.percentile(final_returns, 5)
        fig_hist = px.histogram(final_returns, nbins=50, title="Distribution of 1-Year Future Returns (1000 Simulations)")
        st.plotly_chart(fig_hist)
        st.markdown(f"95% VaR from Simulation (1-Year Loss Potential): {sim_var:.2%} (Probability-based; assumes historical patterns continue). (Plain English: Monte Carlo uses random sampling to model thousands of possible futures, showing the range of outcomes like a 'worst 5% case' loss.)")
    else:
        st.warning("Cannot run simulation (data unavailable).")

# Section 6: Portfolio Optimization and Plots
if st.session_state.portfolio and len(tickers) >= 2:
    st.header("Portfolio Optimization")
    historical_data = fetch_historical_data(tickers, starts, ends, intervals)
    returns_df = historical_data.pct_change().dropna()

    # Calculate weights for portfolio metrics
    current_prices = [yf.Ticker(t).info.get('currentPrice', np.nan) for t in tickers]
    current_values = [st.session_state.portfolio[t]['shares'] * current_prices[i] if not np.isnan(current_prices[i]) else 0 for i, t in enumerate(tickers)]
    total_value = sum(current_values)
    if total_value > 0:
        weights = np.array([v / total_value for v in current_values])
    else:
        weights = np.array([1.0 / num_assets] * num_assets)  # Equal weights fallback

    mean_returns_monthly = returns_df.mean().values
    mean_returns = mean_returns_monthly * periods_per_year  # Annualized mean returns
    cov_matrix_monthly = returns_df.cov().values
    cov_matrix = cov_matrix_monthly * periods_per_year  # Annualized covariance

    # Portfolio mean and vol for Sharpe
    port_vol_monthly = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_monthly, weights)))
    port_vol = port_vol_monthly * np.sqrt(periods_per_year)
    port_mean_monthly = np.mean(returns_df.dot(weights))
    port_mean = port_mean_monthly * periods_per_year

    def port_perf(weights):
        ret = np.dot(weights, mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return ret, vol

    def neg_sharpe(weights):
        ret, vol = port_perf(weights)
        return - (ret - annual_rf) / vol if vol > 0 else np.inf  # Use annualized RF

    def min_vol(weights):
        return port_perf(weights)[1]

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
    bnds = tuple((0, 1) for _ in range(num_assets))
    init_guess = np.array([1.0 / num_assets] * num_assets)

    # Max Sharpe
    opt_sharpe = sco.minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bnds, constraints=cons)
    opt_weights_sharpe = opt_sharpe.x
    opt_ret_sharpe, opt_vol_sharpe = port_perf(opt_weights_sharpe)
    opt_sharpe_ratio = -opt_sharpe.fun if opt_sharpe.success else np.nan

    # Min Risk
    opt_min_risk = sco.minimize(min_vol, init_guess, method='SLSQP', bounds=bnds, constraints=cons)
    opt_weights_min_risk = opt_min_risk.x
    opt_ret_min_risk, opt_vol_min_risk = port_perf(opt_weights_min_risk)

    st.subheader("Optimized Allocations")
    st.write("**Max Sharpe Ratio Portfolio (Balanced Return/Risk):**")
    st.dataframe(pd.DataFrame({'Ticker': tickers, 'Optimal Weight': [f"{w:.2%}" for w in opt_weights_sharpe]}), height=800)
    st.markdown(f"Expected Annual Return: {opt_ret_sharpe:.4%}, Annual Risk (Vol): {opt_vol_sharpe:.4%}, Sharpe Ratio: {opt_sharpe_ratio:.2f}" if not np.isnan(opt_sharpe_ratio) else "Optimization failed for Max Sharpe.")
    st.write("**Minimum Risk Portfolio (Lowest Volatility):**")
    st.dataframe(pd.DataFrame({'Ticker': tickers, 'Optimal Weight': [f"{w:.2%}" for w in opt_weights_min_risk]}), height=800)
    st.markdown(f"Expected Annual Return: {opt_ret_min_risk:.4%}, Annual Risk (Vol): {opt_vol_min_risk:.4%}")

    # Efficient Frontier
    st.subheader("Efficient Frontier with CAL and Risk-Free Rate")
    target_rets = np.linspace(opt_ret_min_risk, mean_returns.max(), 50)
    frontier_vols = []
    for tr in target_rets:
        cons_tr = cons + ({'type': 'eq', 'fun': lambda w: port_perf(w)[0] - tr},)
        res = sco.minimize(min_vol, init_guess, method='SLSQP', bounds=bnds, constraints=cons_tr)
        if res.success:
            frontier_vols.append(res.fun)
        else:
            frontier_vols.append(np.nan)
    fig_combined = go.Figure()
    fig_combined.add_trace(go.Scatter(x=frontier_vols, y=target_rets, mode='lines', name='Efficient Frontier'))
    fig_combined.add_trace(go.Scatter(x=[opt_vol_sharpe], y=[opt_ret_sharpe], mode='markers', name='Max Sharpe (Tangency)'))
    fig_combined.add_trace(go.Scatter(x=[opt_vol_min_risk], y=[opt_ret_min_risk], mode='markers', name='Min Risk'))
    fig_combined.add_trace(go.Scatter(x=[0], y=[annual_rf.item()], mode='markers', name='Risk-Free Rate', marker=dict(color='green', size=10)))
    cal_x = np.linspace(0, opt_vol_sharpe * 1.5, 50)
    cal_y = annual_rf.item() + cal_x * opt_sharpe_ratio
    fig_combined.add_trace(go.Scatter(x=cal_x, y=cal_y, mode='lines', name='CAL', line=dict(dash='dash')))

    # Add individual stocks
    stock_vols = np.sqrt(np.diag(cov_matrix))
    fig_combined.add_trace(go.Scatter(
        x=stock_vols,
        y=mean_returns,
        mode='markers+text',
        name='Individual Stocks',
        text=tickers,
        textposition='top center',
        marker=dict(color='purple', size=10)
    ))

    fig_combined.update_layout(xaxis_title='Annual Risk (Volatility)', yaxis_title='Expected Annual Return', title='Efficient Frontier with CAL and Risk-Free Rate')
    st.plotly_chart(fig_combined)
    # Market data for SML (annualized)
    market_data = yf.download('^GSPC', start=starts, end=ends, interval=intervals)['Close']
    market_returns_monthly = market_data.pct_change().dropna()
    market_mean_monthly = market_returns_monthly.mean()
    if isinstance(market_mean_monthly, pd.Series):
        market_mean_monthly = market_mean_monthly.iloc[0]
    market_mean = market_mean_monthly * periods_per_year
    market_vol_monthly = market_returns_monthly.std()
    if isinstance(market_vol_monthly, pd.Series):
        market_vol_monthly = market_vol_monthly.iloc[0]
    market_vol = market_vol_monthly * np.sqrt(periods_per_year)

    # Betas (unchanged, as beta is scale-invariant)
    betas = []
    for t in tickers:
        stock_returns_monthly = returns_df[t].dropna()
        stock_returns_monthly = stock_returns_monthly.reindex(market_returns_monthly.index).ffill().dropna()
        aligned_market = market_returns_monthly.reindex(stock_returns_monthly.index)
        aligned = pd.concat([stock_returns_monthly, aligned_market], axis=1).dropna()
        if len(aligned) > 1:
            cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
            beta = cov / (aligned.iloc[:, 1].var())
            betas.append(beta)
        else:
            betas.append(np.nan)
    valid_betas = [b for b in betas if not np.isnan(b)]
    if len(valid_betas) > 0:
        betas_for_dot = [b if not np.isnan(b) else np.mean(valid_betas) for b in betas]
        port_beta = np.dot(weights, betas_for_dot)
    else:
        port_beta = np.nan
        st.warning("Unable to compute betas due to data misalignment.")

    # SML
    st.subheader("Security Market Line (SML)")
    if len(valid_betas) > 0:
        valid_indices = [i for i, b in enumerate(betas) if not np.isnan(b)]
        valid_mean_returns = [mean_returns[i] for i in valid_indices]  # Already annualized
        valid_tickers = [tickers[i] for i in valid_indices]
        expected_rets = [annual_rf + b * (market_mean - annual_rf) for b in valid_betas]
        fig_sml = go.Figure()
        fig_sml.add_trace(go.Scatter(x=valid_betas, y=valid_mean_returns, mode='markers+text', text=valid_tickers, name='Stocks'))
        max_beta = max(valid_betas + [1])
        x_sml = [0, max_beta + 0.5]
        y_sml = [annual_rf, annual_rf + x_sml[1] * (market_mean - annual_rf)]
        fig_sml.add_trace(go.Scatter(x=x_sml, y=y_sml, mode='lines', name='SML'))
        fig_sml.update_layout(xaxis_title='Beta', yaxis_title='Expected Annual Return', title='SML (Expected Return vs. Beta)')
        st.plotly_chart(fig_sml)
    else:
        st.warning("No valid betas to plot SML. Check data alignment or ticker availability.")

    # Sharpe Ratios (annualized)
    if port_vol > 0 and not np.isnan(port_mean):
        current_sharpe = (port_mean - annual_rf.item()) / port_vol
    else:
        current_sharpe = np.nan
    if market_vol > 0 and not np.isnan(market_mean):
        market_sharpe = (market_mean - annual_rf.item()) / market_vol
    else:
        market_sharpe = np.nan
    st.markdown(f"**Current Portfolio Sharpe Ratio:** {current_sharpe:.2f}" if not np.isnan(current_sharpe) else "**Current Portfolio Sharpe Ratio:** N/A (Insufficient data)")
    st.markdown(f"**Optimized Max Sharpe Ratio:** {opt_sharpe_ratio:.2f}" if not np.isnan(opt_sharpe_ratio) else "**Optimized Max Sharpe Ratio:** N/A")
    st.markdown(f"**Market Sharpe Ratio:** {market_sharpe:.2f}" if not np.isnan(market_sharpe) else "**Market Sharpe Ratio:** N/A (Insufficient market data)")
# 7 Section: Custom Complete Portfolio based on MPT
if st.session_state.portfolio and len(tickers) >= 2:
    st.header("Custom Complete Portfolio (Modern Portfolio Theory)")
    st.markdown("""
    This section applies Modern Portfolio Theory (MPT) (a framework for constructing portfolios to maximize expected return for a given risk level or minimize risk for a given return level, developed by Harry Markowitz).
    You can input your desired expected return or risk level for the complete portfolio (combination of risky assets and risk-free US bond).
    The risky portfolio is the tangency portfolio (max Sharpe ratio portfolio, the optimal risky portfolio where the capital allocation line touches the efficient frontier).
    The complete portfolio allocates between the risk-free asset (BND, US bond ETF, with average monthly return as risk-free rate) and the tangency portfolio.
    If short selling is allowed, the efficient frontier extends, and allocation can involve short positions (negative weights) or leverage (allocation >100% to risky, negative to risk-free).
    All metrics are annualized (e.g., expected return is average annual return, volatility is annual standard deviation of returns).
    If the desired level cannot be achieved without short/leverage (when not allowed), it shows boundary portfolios: maximum complete (fully in tangency), minimum risk risky (not including risk-free), and all risk-free.
    """)

    allow_short = st.checkbox("Allow Short Selling (and Leverage/Borrowing)", value=False)
    input_type = st.selectbox("Select Input Type", ["Expected Return", "Risk Level (Volatility)"])
    if input_type == "Expected Return":
        target = st.number_input("Desired Annual Expected Return (decimal, e.g., 0.12 for 12%)", value=0.12, step=0.01)
    else:
        target = st.number_input("Desired Annual Volatility (Risk Level, decimal, e.g., 0.15 for 15%)", value=0.15, min_value=0.0, step=0.01)

    # Recompute with annualized
    historical_data = fetch_historical_data(tickers, starts, ends, intervals)
    returns_df = historical_data.pct_change().dropna()
    mean_returns_monthly = returns_df.mean().values
    mean_returns = mean_returns_monthly * periods_per_year
    cov_matrix_monthly = returns_df.cov().values
    cov_matrix = cov_matrix_monthly * periods_per_year

    def port_perf(weights):
        ret = np.dot(weights, mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return ret, vol

    def neg_sharpe(weights):
        ret, vol = port_perf(weights)
        return - (ret - annual_rf) / vol if vol > 0 else np.inf

    def min_vol(weights):
        return port_perf(weights)[1]

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
    if allow_short:
        bnds = None
    else:
        bnds = tuple((0, 1) for _ in range(num_assets))
    init_guess = np.array([1.0 / num_assets] * num_assets) if num_assets > 0 else np.array([])

    opt_sharpe = sco.minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bnds, constraints=cons)
    if not opt_sharpe.success:
        st.error("Optimization for tangency portfolio failed. Check data or add more assets.")
    else:
        opt_weights = opt_sharpe.x
        opt_ret, opt_vol = port_perf(opt_weights)
        opt_sharpe_ratio = (opt_ret - annual_rf) / opt_vol if opt_vol > 0 else np.nan

        # Min risk risky
        opt_min_risk = sco.minimize(min_vol, init_guess, method='SLSQP', bounds=bnds, constraints=cons)
        min_risk_weights = opt_min_risk.x if opt_min_risk.success else None
        min_ret, min_vol = port_perf(min_risk_weights) if min_risk_weights is not None else (np.nan, np.nan)

        # Max ret risky (single asset with highest mean return)
        if len(mean_returns) > 0:
            max_ret_idx = np.argmax(mean_returns)
            max_ret_weights = np.zeros(num_assets)
            max_ret_weights[max_ret_idx] = 1
            max_ret, max_vol = port_perf(max_ret_weights)
        else:
            max_ret, max_vol = np.nan, np.nan

        # Compute complete portfolio
        if np.isnan(opt_ret) or opt_vol == 0 or opt_ret <= annual_rf.item():
            st.warning("Invalid tangency portfolio (e.g., return <= risk-free or zero vol). Cannot compute complete portfolio.")
        else:
            if input_type == "Expected Return":
                if target == annual_rf.item():
                    alpha = 0.0
                    complete_ret = annual_rf.item()
                    complete_vol = 0.0
                    complete_weights = {rf_ticker: 1.0}
                    for t in tickers:
                        complete_weights[t] = 0.0
                else:
                    alpha = (target - annual_rf.item()) / (opt_ret - annual_rf.item())
                    if not allow_short and (alpha < 0 or alpha > 1):
                        st.warning(f"Cannot achieve desired annual return {target:.4%} without short selling or leverage.")
                        if alpha < 0:
                            st.info(f"Minimum complete portfolio (all in risk-free): Annual Expected Return {annual_rf.item():.4%}, Volatility 0.00% (weights: 100% in {rf_ticker})")
                            st.info(f"Optimal minimum risk risky portfolio (not complete): Annual Expected Return {min_ret:.4%}, Volatility {min_vol:.4%} (weights: {dict(zip(tickers, [f'{w:.2%}' for w in min_risk_weights]))})")
                        else:
                            st.info(f"Maximum complete portfolio (all in tangency): Annual Expected Return {opt_ret:.4%}, Volatility {opt_vol:.4%} (weights: {dict(zip(tickers, [f'{w:.2%}' for w in opt_weights]))}, 0% in {rf_ticker})")
                            st.info(f"Optimal minimum risk risky portfolio (not complete): Annual Expected Return {min_ret:.4%}, Volatility {min_vol:.4%} (weights: {dict(zip(tickers, [f'{w:.2%}' for w in min_risk_weights]))})")
                        st.info(f"Maximum return risky portfolio (not complete, highest return asset): Annual Expected Return {max_ret:.4%}, Volatility {max_vol:.4%} (weights: 100% in {tickers[max_ret_idx]})")
                    else:
                        complete_ret = target
                        complete_vol = abs(alpha) * opt_vol
                        complete_weights = {rf_ticker: 1 - alpha}
                        for i, t in enumerate(tickers):
                            complete_weights[t] = alpha * opt_weights[i]
            else:  # Risk Level
                if target == 0:
                    alpha = 0.0
                    complete_ret = annual_rf.item()
                    complete_vol = 0.0
                    complete_weights = {rf_ticker: 1.0}
                    for t in tickers:
                        complete_weights[t] = 0.0
                else:
                    alpha = target / opt_vol  # Positive branch
                    if alpha < 0:
                        alpha = -alpha
                    if not allow_short and (alpha > 1):
                        st.warning(f"Cannot achieve desired annual volatility {target:.4%} without leverage.")
                        st.info(f"Maximum complete portfolio (all in tangency): Volatility {opt_vol:.4%}, Annual Expected Return {opt_ret:.4%} (weights: {dict(zip(tickers, [f'{w:.2%}' for w in opt_weights]))}, 0% in {rf_ticker})")
                        st.info(f"Optimal minimum risk risky portfolio (not complete): Volatility {min_vol:.4%}, Annual Expected Return {min_ret:.4%} (weights: {dict(zip(tickers, [f'{w:.2%}' for w in min_risk_weights]))})")
                        st.info(f"Minimum complete portfolio (all in risk-free): Volatility 0.00%, Annual Expected Return {annual_rf.item():.4%} (weights: 100% in {rf_ticker})")
                    else:
                        complete_vol = target
                        complete_ret = annual_rf.item() + alpha * (opt_ret - annual_rf.item())
                        complete_weights = {rf_ticker: 1 - alpha}
                        for i, t in enumerate(tickers):
                            complete_weights[t] = alpha * opt_weights[i]

            # Display results
            if 'complete_weights' in locals():
                st.subheader("Complete Portfolio Details")
                complete_df = pd.DataFrame(list(complete_weights.items()), columns=['Asset', 'Weight'])
                complete_df['Weight'] = complete_df['Weight'].apply(lambda w: f"{w:.2%}")
                st.dataframe(complete_df, height=800)  # Wider row height
                st.markdown(f"**Annual Expected Return:** {complete_ret:.4%}")
                st.markdown(f"**Annual Volatility (Risk):** {complete_vol:.4%}")
                complete_sharpe = (complete_ret - annual_rf.item()) / complete_vol if complete_vol > 0 else np.nan
                st.markdown(f"**Sharpe Ratio:** {complete_sharpe:.2f}" if not np.isnan(complete_sharpe) else "**Sharpe Ratio:** N/A (zero risk)")
                st.markdown(r"**Note:** Weights may sum to 100% (or not exactly due to rounding). Negative weights indicate short positions (selling borrowed assets to potentially profit from price declines). Leverage occurs if risk-free weight <0 (borrowing at risk-free rate to invest more in risky assets). (Plain English: Short selling is betting on price drops; leverage is using borrowed money to amplify investments.)")
                # Graph: Position on CAL
                fig_cal = go.Figure()
                fig_cal.add_trace(go.Scatter(x=[0], y=[annual_rf.item()], mode='markers', name='Risk-Free', marker=dict(color='green', size=10)))
                cal_x = np.linspace(0, max(complete_vol, opt_vol) * 1.5, 50)
                cal_y = annual_rf.item() + (opt_ret - annual_rf.item()) / opt_vol * cal_x if opt_vol > 0 else np.full_like(cal_x, annual_rf.item())
                fig_cal.add_trace(go.Scatter(x=cal_x, y=cal_y, mode='lines', name='Capital Allocation Line (CAL)'))
                fig_cal.add_trace(go.Scatter(x=[opt_vol], y=[opt_ret], mode='markers', name='Tangency Portfolio'))
                fig_cal.add_trace(go.Scatter(x=[complete_vol], y=[complete_ret], mode='markers', name='Your Complete Portfolio', marker=dict(color='red', size=12)))
                fig_cal.update_layout(xaxis_title='Annual Volatility', yaxis_title='Expected Annual Return', title='Your Portfolio on the CAL (Capital Allocation Line)')
                st.plotly_chart(fig_cal)

# Refresh Button
if st.button("Refresh Data (Update Prices & Graphs)"):

    st.rerun()
