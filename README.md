# Mean Reversion Strategy: SPY-QQQ

This project explores a simple statistical mean-reversion trading strategy using the price spread between SPY (S&P 500 ETF) and QQQ (Nasdaq-100 ETF).

The goal is to analyze whether deviations in the SPY-QQQ spread tend to revert toward a historical average and whether this behavior can be exploited using a systematic trading rule.

---

## Data
- Daily adjusted close prices for:
    - SPY (S&P 500 ETF)
    - QQQ (Nasdaq-100 ETF)
- Source: Yahoo Finance
- Time period: 2015 - 2024

---

## Methodology
1. Computed the daily price spread between SPY and QQQ
2. Calculated rolling mean and standard deviation of the spread
3. Converted spread deviations into z-scores
4. Generated trading signals based on z-score thresholds:
    - Long spread when z-score <= -1
    - Short spread when z-score >= 1
    - Exit positions when z-score crosses back through 0
5. Backtested a long/short spread strategy
6. Evaluated performance using:
    - CAGR
    - Annualized volatility
    - Sharpe ratio
    - Maximum drawdown
7. Incorporated transaction costs (5 basis points per round trip)
8. Performed parameter tuning over different lookback windows and entry thresholds

---


## Key Findings
- The mean-reversion strategy produces smoother returns than buy-and-hold SPY
- Risk-adjusted performance improves under certain parameter combinations
- The strategy underperforms during strong trending bull markets
- Transaction costs materially reduce overall performance

---

## Limitations
- No leverage or capital constraints
- No shorting or borrow costs
- No regime detection or out-of-sample testing
- Strategy performance is sensitive to parameter selection

---

## Tools Used
- Python
- pandas, numpy
- matplotlib
- yfinance

---

## Disclaimer
This project is for educational purposes only and does not constitute investment advice.