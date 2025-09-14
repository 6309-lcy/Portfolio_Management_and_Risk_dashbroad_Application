## Overview
This Portfolio Management and Risk Dashbroad Application is a comprehensive web-based tool for managing stock portfolios, analyzing performance, assessing risks, optimizing allocations, and simulating scenarios. Built with Streamlit for the user interface and yfinance for financial data retrieval, it supports daily data intervals and historical analysis from 2015 onward. The application incorporates Modern Portfolio Theory (MPT) to compute metrics such as Sharpe ratio, beta, Value at Risk (VaR), and Conditional Value at Risk (CVaR).
This project is designed for educational and personal use. It assumes a default starting balance of $100,000 USD and fetches data from Yahoo Finance. **Note that this is not financial advice; consult a professional for investment decisions.**

---
## Features
- **Performance Tracking**: View current values, returns, and allocation visualizations via pie charts.
- **Portfolio Management**: Add, sell, and track stocks with real-time price updates and cash balance management.
- **Risk and Return Metrics**: Annualized calculations for expected return, volatility, beta, Sharpe ratio, VaR, and CVaR at individual stock and portfolio levels.
- **Visualization**: Cumulative return graphs, volatility comparisons, correlation heatmaps, efficient frontier, Security Market Line (SML), and Capital Allocation Line (CAL).
- **Stress Testing**: Simulate portfolio performance during historical events (e.g., 2008 Financial Crisis, COVID-19 Crash).
- **Monte Carlo Simulation**: Generate 1000 one-year projections with histograms and VaR estimates.
- **Scenario Analysis**: Estimate impacts from hypothetical market drops using beta.
- **Optimization**: Compute maximum Sharpe ratio and minimum volatility portfolios.
- **Custom MPT Portfolio**: Allocate between tangency portfolio and risk-free asset, with support for short selling and leverage.
- **Export**: Download portfolio data as CSV.
- **Refresh**: Update prices and graphs with a single button.

---
## Limitations
- Depends on Yahoo Finance, which may experience delays or downtime.
- Historical patterns assumed to persist; excludes dividends and taxes.
- Optimization runtime increases with asset count.
- For production, deploy via Streamlit Community Cloud or similar.
---
## Functions to be added
- **Hedging suggestion** Based on the portfolio, auto grab related derivatives and measure the hedge ratio to quantify result.
- **API connection** Connect with broker account for real time management.
- **Machine Learning Integration** Integrate with multi-model agents for better investment/ portfolio management. (e.g. sentimental analysis)

---
## Future development
- Integrate with others models to build up a portfolio management system
- Deploy this system for 2025 bloomberg trading challenge
- Study and develop further for my research topic in "Event-based Automatic Portfolio Investment Strategies using Multi-modal Large Language Models"
