Mean reversion is the concept of stocks (in this case) returning back towards their historical average (the "mean).
As in, if a stock starts to peak up or crash down, mean reversion says it will eventually return back towards that average, unless something fundamental has changed.

Why does mean reversion happen:
- markets overreact to news
- investor psychology causes prices to swing too far
- eventually fundamentals (earnings, growth, etc.) may pull prices back toward normal levels

Applications:
- pairs trading (done with 2 stocks that usually move together, knowing that no matter how far one stock goes, the other will eventually catch up)
- statistical arbitrage (similar to pair trading but instead of a "pair" of stocks, you are looking at a bunch of stocks)
- boilinger bands (technical analysis) (bands get wider when prices are volatile, and get narrower when prices are calm) (help traders see when a stok might be too high or too low compared to its normal behavior)
- value investing (buying a really good stock when the price is low)

```python
# importing necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# make sure plots show inside notebook
%matplotlib inline
```


```python
# 2 of the most-wdiely traded ETF's in the U.S:
# - SPY (SPDR S&P 500 ETF Trust): tracks the perfromance of the S&P 500 Index
# - QQQ (Invesco QQQ Trust): tracks the performance of the Nasdaq-100 Index)

# define tickers (setting up the 2 ETF's) (exhange-traded funds) 
tickers = ['SPY', 'QQQ']

# download full data
data = yf.download(tickers, start='2015-01-01', end='2024-06-04', auto_adjust=True)

# extract just 'Adj Close' using pandas .loc with MultiIndex columns
adj_close = data['Close']
```

    [*********************100%***********************]  2 of 2 completed



```python
# make graph comparing SPY and QQQ
adj_close.plot(figsize=(12,6), title='SPY vs QQQ Adjusted Close Prices')
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.grid(True)
plt.show()
```


    
![png](output_3_0.png)
    



```python
#1. compute the spread (the difference between the SPY line and the QQQ line)
spread = adj_close['SPY'] - adj_close['QQQ']

#2. calculate the the mean (average) and the std for the past lookback days
lookback = 60
rolling_mean = spread.rolling(window=lookback).mean()
rolling_std = spread.rolling(window=lookback).std()
    
#3. compute z-score
zscore = (spread - rolling_mean) / rolling_std

# quick check:L show head of each series
pd.DataFrame({
    'SPY-QQQ Spread': spread, 
    'Rolling Mean': rolling_mean, 
    'Rolling Std': rolling_std,
    'Z-Score': zscore
}).dropna().head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SPY-QQQ Spread</th>
      <th>Rolling Mean</th>
      <th>Rolling Std</th>
      <th>Z-Score</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-03-30</th>
      <td>76.363701</td>
      <td>75.900390</td>
      <td>0.853314</td>
      <td>0.542955</td>
    </tr>
    <tr>
      <th>2015-03-31</th>
      <td>75.878990</td>
      <td>75.882507</td>
      <td>0.842073</td>
      <td>-0.004177</td>
    </tr>
    <tr>
      <th>2015-04-01</th>
      <td>75.774200</td>
      <td>75.891416</td>
      <td>0.837974</td>
      <td>-0.139880</td>
    </tr>
    <tr>
      <th>2015-04-02</th>
      <td>76.331978</td>
      <td>75.915201</td>
      <td>0.829710</td>
      <td>0.502316</td>
    </tr>
    <tr>
      <th>2015-04-06</th>
      <td>76.704979</td>
      <td>75.930308</td>
      <td>0.835780</td>
      <td>0.926883</td>
    </tr>
  </tbody>
</table>
</div>




```python
#* now visualize the table above into 3 graphs

plt.figure(figsize=(12,5))
plt.plot(spread.index, zscore, label='Z-Score (60d window)')
plt.axhline( 1.0, color='grey', linestyle='--') # upper threshold
plt.axhline( 0.0, color='black', linestyle='-') # mean
plt.axhline(-1.0, color='grey', linestyle='--') # lower threshold
plt.legend()
plt.title('Spread Z-Score Between SPY and QQQ')
plt.show()
```


    
![png](output_5_0.png)
    


now we'll generate trading signals and backtest by using a classic mean-reversion rule:
- long the spread (long SPY, short QQQ) when z <= -1
- exit (flat) when z >= 0
- short the spread (short SPY, long QQQ) when z >= 1
- exit when z <= 0


```python
#1. calculate daily returns
ret_spy = adj_close['SPY'].pct_change()
ret_qqq = adj_close['QQQ'].pct_change()

#2. build signal series: 1 = long spread, -1 = short spread, 0 = flat
signal = pd.Series(0, index=zscore.index)
signal[zscore <= -1] = 1  # enter long
signal[zscore >= 1] = -1  # enter short

#3. apply a simple "exit to zero" rule whenever z crosses back through 0, go to flat
signal[(zscore.shift() < 0) & (zscore >= 0)] = 0
signal[(zscore.shift() > 0) & (zscore <= 0)] = 0

#4. forward-fill positions so we hold until exit
positions = signal.where(signal != 0).ffill().fillna(0)

#5. strategy returns:
# if position = 1: ret_spy - ret_qqq
# if position = -1: -(ret_spy - ret_qqq)
strat_ret = positions.shift() * (ret_spy - ret_qqq)

#6. compute cumulative returns
cum_strat = (1 + strat_ret.dropna()).cumprod()

#7. plot performance vs buy-and-old SPY
plt.figure(figsize=(12,5))
plt.plot(cum_strat, label='Strategy')
plt.plot((1+ret_spy.dropna()).cumprod(), label='Buy & Hold SPY', alpha=0.7)
plt.title('Mean Reversion Strategy vs Buy & Hold SPY')
plt.ylabel('Cumulative Return')
plt.legend()
plt.show()
```


    
![png](output_7_0.png)
    

Underperformance vs SPY: over this period, the strategy captures some sideways moves and small mean reversals, but misses the big bull runs in 2017-2021 and suffers drawdowns during strong trends.

Low Volatility: this strategy is much smoother, which may be attractive if you care more about steadiness than raw returnsso far we have:
- pulled the adjusted prices

- computed the SPY-QQQ spread (the difference betwee the 2 ETF's)

- plotted the normalized -zscore with +-1 thresholds (1 above and 1 below to account for deviation) (how far is today's spread from the average; if the difference is super big, the z-score is a large number, if the difference is very small, the z-score is very negative)

- made simple rules/signals:
 * when the z-score is very low, we pretend to buy orange and sell thinking orange is cheaper than blue
 * when the z-score goes back to zero, we pretend to sell both orange and blue since now we aren't making any money because there are no differences and there stands a risk of going into the negative
 * when the z-score is very high, we pretend to buy blue and sell orange thinking blue is cheaper than orange

- calculated a 60-day rolling mean/std which just means calculating the mean and std over the 60 days (which showed us on average what the difference usually is, and how much does the difference bounce around)

```python
def compute_performance(returns, freq=252):
    r = returns.dropna()
    # compound annual growth rate
    total_days = (r.index[-1] - r.index[0]).days
    cagr = (1 + r).prod() ** (365.0/total_days) - 1

    # annualized volatility
    vol = r.std() * np.sqrt(freq)

    # sharpe (0% RF)
    sharpe = cagr / vol if vol else np.nan

    # max drawdown
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    mdd = ((cum-peak) / peak).min()
    return{
    'CAGR': cagr,
    'Volatility': vol,
    'Sharpe': sharpe,
    'Max Drawdown': mdd

    }

# compute and print
strat_metrics = compute_performance(strat_ret)
bh_metrics = compute_performance(ret_spy)

print("Strategy Performance: ", strat_metrics)
print("\nBuy & Hold SPY: ", bh_metrics)
```

    Strategy Performance:  {'CAGR': np.float64(0.02515768379009331), 'Volatility': np.float64(0.08470025219301863), 'Sharpe': np.float64(0.2970201757222975), 'Max Drawdown': -0.20918280800014824}
    
    Buy & Hold SPY:  {'CAGR': np.float64(0.1248855190060636), 'Volatility': np.float64(0.1784504559091447), 'Sharpe': np.float64(0.699833006140635), 'Max Drawdown': -0.33717280153771284}



```python
# run this when testing what time frame you're running your paper trading

# print("Date range in `data`:", data.index.min(), "to", data.index.max())
# adj_close = data['Close']
# print("Date range in `adj_close`:", adj_close.index.min(), "to", adj_close.index.max())
```
now we'll deal with model transaction costs, because in the real world, every trade costs something-commissions, slippage, fees. Let's subtract 5 basis points (0.05%) per round-trip trade and see how performance changes

```python
#1. set a per-trade cost (5 bps round-trip)
cost_per_trade = 0.0005

#2. count trades as changes in position
trades = positions.diff().abs()

#3. compute transaction-cost series
tc = trades * cost_per_trade

#4. net strategy returns after cpsts
strat_ret_tc = strat_ret - tc

#5. recompute metrics
tc_metrics = compute_performance(strat_ret_tc)

print("Strategy w/ 5bp Costs:", tc_metrics)
```

    Strategy w/ 5bp Costs: {'CAGR': np.float64(0.020023136713496426), 'Volatility': np.float64(0.08442829730361781), 'Sharpe': np.float64(0.2371614417555999), 'Max Drawdown': -0.21310233973088075}

What we did so far:

we turned the z-score rules into signals where (1=long spread, 0=flat, -1=short spread), and then filled in each day's position so the robot stays in until the next exist signal

* long spread: we bought something (e.g. we "own" SPY and sold QQQ), betting the spread will go up
* short spread: it has sold something it dosen't own (e.g. we "own" QQQ and have sold SPY), betting the spread will go down
* flat (not "in"): it dosen't hold anything, so it's just waiting

everyday we looked at the daily returns of each ETF:
- when the robot was "in" (long or short), it earned/lost exactly the difference in those daily returns
-  we added up all those little gains and losses over many years to see how much money we'd have if we started with $1

we compared the 2 lines (blue and orange):
- blue line: the money made by the robot
- orange line: if we bought the stock and held onto it

we used 4 rules to measure the effectivness of the robot:
- CAGR: average yearly growth
- Volatility: how much the fund goes up and down
- Sharpe Ratio: how much we making per risk -> (average annual return - risk - free rate)/annualized volatillity
    * Sharpe = CAGR/VolatilityNow we'll do some parameter tuning to find the sweet spot between the lookback window and the entry/exit thresholds. We will do this by wrapping our backtest into a function, looping through a grid of these values, and produce a table showing Sharpe, CAGR, and a drawdown for each combo. 

We are looking for the best combination of the lookback window and the entry threshold which gives us the highest risk-adjusted return (the highest Sharpe ratio)

```python
# grabs SPY/QQQ prices and returns a simple DataFrame of their adjusted closes
def download_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)
    return df['Close']

# given those prices and the arguments passed through the function, this cimputes the spread, 
# rolling mean/std, builds a -+1 signals (with proper .loc[...] so pandas doesn't warn, forward-fills so 
# you hold the fund until you cross back through zero to prevent loss.

# also takes the positions Series and the price DataFrame and computs daily strategy returns vs SPY returns
# for any return series, calculates CAGR, annualized volatility, Sharpe, and max drawdown
def generate_positions(adj_close, lookback, entry_z):
    spread = adj_close['SPY'] - adj_close['QQQ']
    m = spread.rolling(lookback).mean()
    s = spread.rolling(lookback).std()
    zsc = (spread - m) / s

    sig = pd.Series(0, index=zsc.index)
    # use .loc to avoid the deprecation warning
    sig.loc[zsc <= -entry_z] = 1
    sig.loc[zsc >=  entry_z] = -1
    sig.loc[(zsc.shift() < 0) & (zsc >= 0)] = 0
    sig.loc[(zsc.shift() > 0) & (zsc <= 0)] = 0

    # forward-fill and fill initial NaNs
    return sig.where(sig != 0).ffill().fillna(0)

def backtest(adj_close, positions):
    ret_spy = adj_close['SPY'].pct_change()
    ret_qqq = adj_close['QQQ'].pct_change()
    strat = positions.shift() * (ret_spy - ret_qqq)
    return strat.dropna(), ret_spy.dropna()

def compute_metrics(strat_ret, bh_ret, freq=252):
    def perf(r):
        r = r.dropna()
        days = (r.index[-1] - r.index[0]).days
        cagr = (1 + r).prod() ** (365.0/days) - 1
        vol = r.std() * np.sqrt(freq)
        sharpe = cagr/vol if vol else np.nan
        cum = (1 + r).cumprod()
        dd = (cum - cum.cummax())/cum.cummax()
        mdd = dd.min()
        return cagr, vol, sharpe, mdd
    return perf(strat_ret), perf(bh_ret)
```


```python
# here we run a parameter grid search where its looped over a small grid of lookbacks days and entry 
# thresholds. For each (lookback, entry_z) pair we generated the trading positions, backtested the stregy 
# vs buy-&-hold SPY, computed performance metrics for each, appended those results into a list of dicts
# and converted that list into a DataFrame called grid which is then sorted by the strat_Sharpe column we
# added, so we can immediately see which parameter combinations gave the highest risk-adjusted return

# download once
adj = download_data(['SPY','QQQ'], '2015-01-01','2024-06-01')

results = []
for lookback in [20, 40, 60, 120]:
    for entry_z in [0.5, 1.0, 1.5]:
        pos = generate_positions(adj, lookback, entry_z)
        strat_ret, bh_ret = backtest(adj, pos)
        (c_s, v_s, sr_s, dd_s), (c_b, v_b, sr_b, dd_b) = compute_metrics(strat_ret, bh_ret)

        results.append({
            'lookback': lookback,
            'entry_z': entry_z,
            'strat_CAGR': c_s,
            'strat_Sharpe': sr_s,
            'strat_Drawdown': dd_s,
            'bh_CAGR': c_b,
            'bh_Sharpe': sr_b,
            'bh_Drawdown': dd_b
        })

grid = pd.DataFrame(results)
grid.sort_values('strat_Sharpe', ascending=False).head(10)
```

    [*********************100%***********************]  2 of 2 completed





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lookback</th>
      <th>entry_z</th>
      <th>strat_CAGR</th>
      <th>strat_Sharpe</th>
      <th>strat_Drawdown</th>
      <th>bh_CAGR</th>
      <th>bh_Sharpe</th>
      <th>bh_Drawdown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>60</td>
      <td>1.5</td>
      <td>0.041168</td>
      <td>0.487433</td>
      <td>-0.151839</td>
      <td>0.124904</td>
      <td>0.699787</td>
      <td>-0.337173</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40</td>
      <td>1.0</td>
      <td>0.032255</td>
      <td>0.380141</td>
      <td>-0.255615</td>
      <td>0.124904</td>
      <td>0.699787</td>
      <td>-0.337173</td>
    </tr>
    <tr>
      <th>5</th>
      <td>40</td>
      <td>1.5</td>
      <td>0.028481</td>
      <td>0.335643</td>
      <td>-0.203050</td>
      <td>0.124904</td>
      <td>0.699787</td>
      <td>-0.337173</td>
    </tr>
    <tr>
      <th>7</th>
      <td>60</td>
      <td>1.0</td>
      <td>0.025677</td>
      <td>0.303136</td>
      <td>-0.209183</td>
      <td>0.124904</td>
      <td>0.699787</td>
      <td>-0.337173</td>
    </tr>
    <tr>
      <th>6</th>
      <td>60</td>
      <td>0.5</td>
      <td>0.025019</td>
      <td>0.295266</td>
      <td>-0.262431</td>
      <td>0.124904</td>
      <td>0.699787</td>
      <td>-0.337173</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>1.5</td>
      <td>0.023790</td>
      <td>0.279885</td>
      <td>-0.253745</td>
      <td>0.124904</td>
      <td>0.699787</td>
      <td>-0.337173</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>1.0</td>
      <td>0.022083</td>
      <td>0.259796</td>
      <td>-0.292360</td>
      <td>0.124904</td>
      <td>0.699787</td>
      <td>-0.337173</td>
    </tr>
    <tr>
      <th>0</th>
      <td>20</td>
      <td>0.5</td>
      <td>0.021269</td>
      <td>0.250223</td>
      <td>-0.260896</td>
      <td>0.124904</td>
      <td>0.699787</td>
      <td>-0.337173</td>
    </tr>
    <tr>
      <th>10</th>
      <td>120</td>
      <td>1.0</td>
      <td>0.017770</td>
      <td>0.210537</td>
      <td>-0.202288</td>
      <td>0.124904</td>
      <td>0.699787</td>
      <td>-0.337173</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40</td>
      <td>0.5</td>
      <td>0.017183</td>
      <td>0.202465</td>
      <td>-0.246822</td>
      <td>0.124904</td>
      <td>0.699787</td>
      <td>-0.337173</td>
    </tr>
  </tbody>
</table>
</div>




```python
grid.sort_values('strat_Sharpe', ascending=False).head(10).style.format({
    'strat_CAGR': '{:.2%}',
    'strat_Sharpe': '{:.2f}',
    'strat_Drawdown': '{:.1%}'
})
```




<style type="text/css">
</style>
<table id="T_7a121">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_7a121_level0_col0" class="col_heading level0 col0" >lookback</th>
      <th id="T_7a121_level0_col1" class="col_heading level0 col1" >entry_z</th>
      <th id="T_7a121_level0_col2" class="col_heading level0 col2" >strat_CAGR</th>
      <th id="T_7a121_level0_col3" class="col_heading level0 col3" >strat_Sharpe</th>
      <th id="T_7a121_level0_col4" class="col_heading level0 col4" >strat_Drawdown</th>
      <th id="T_7a121_level0_col5" class="col_heading level0 col5" >bh_CAGR</th>
      <th id="T_7a121_level0_col6" class="col_heading level0 col6" >bh_Sharpe</th>
      <th id="T_7a121_level0_col7" class="col_heading level0 col7" >bh_Drawdown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_7a121_level0_row0" class="row_heading level0 row0" >8</th>
      <td id="T_7a121_row0_col0" class="data row0 col0" >60</td>
      <td id="T_7a121_row0_col1" class="data row0 col1" >1.500000</td>
      <td id="T_7a121_row0_col2" class="data row0 col2" >4.12%</td>
      <td id="T_7a121_row0_col3" class="data row0 col3" >0.49</td>
      <td id="T_7a121_row0_col4" class="data row0 col4" >-15.2%</td>
      <td id="T_7a121_row0_col5" class="data row0 col5" >0.124904</td>
      <td id="T_7a121_row0_col6" class="data row0 col6" >0.699787</td>
      <td id="T_7a121_row0_col7" class="data row0 col7" >-0.337173</td>
    </tr>
    <tr>
      <th id="T_7a121_level0_row1" class="row_heading level0 row1" >4</th>
      <td id="T_7a121_row1_col0" class="data row1 col0" >40</td>
      <td id="T_7a121_row1_col1" class="data row1 col1" >1.000000</td>
      <td id="T_7a121_row1_col2" class="data row1 col2" >3.23%</td>
      <td id="T_7a121_row1_col3" class="data row1 col3" >0.38</td>
      <td id="T_7a121_row1_col4" class="data row1 col4" >-25.6%</td>
      <td id="T_7a121_row1_col5" class="data row1 col5" >0.124904</td>
      <td id="T_7a121_row1_col6" class="data row1 col6" >0.699787</td>
      <td id="T_7a121_row1_col7" class="data row1 col7" >-0.337173</td>
    </tr>
    <tr>
      <th id="T_7a121_level0_row2" class="row_heading level0 row2" >5</th>
      <td id="T_7a121_row2_col0" class="data row2 col0" >40</td>
      <td id="T_7a121_row2_col1" class="data row2 col1" >1.500000</td>
      <td id="T_7a121_row2_col2" class="data row2 col2" >2.85%</td>
      <td id="T_7a121_row2_col3" class="data row2 col3" >0.34</td>
      <td id="T_7a121_row2_col4" class="data row2 col4" >-20.3%</td>
      <td id="T_7a121_row2_col5" class="data row2 col5" >0.124904</td>
      <td id="T_7a121_row2_col6" class="data row2 col6" >0.699787</td>
      <td id="T_7a121_row2_col7" class="data row2 col7" >-0.337173</td>
    </tr>
    <tr>
      <th id="T_7a121_level0_row3" class="row_heading level0 row3" >7</th>
      <td id="T_7a121_row3_col0" class="data row3 col0" >60</td>
      <td id="T_7a121_row3_col1" class="data row3 col1" >1.000000</td>
      <td id="T_7a121_row3_col2" class="data row3 col2" >2.57%</td>
      <td id="T_7a121_row3_col3" class="data row3 col3" >0.30</td>
      <td id="T_7a121_row3_col4" class="data row3 col4" >-20.9%</td>
      <td id="T_7a121_row3_col5" class="data row3 col5" >0.124904</td>
      <td id="T_7a121_row3_col6" class="data row3 col6" >0.699787</td>
      <td id="T_7a121_row3_col7" class="data row3 col7" >-0.337173</td>
    </tr>
    <tr>
      <th id="T_7a121_level0_row4" class="row_heading level0 row4" >6</th>
      <td id="T_7a121_row4_col0" class="data row4 col0" >60</td>
      <td id="T_7a121_row4_col1" class="data row4 col1" >0.500000</td>
      <td id="T_7a121_row4_col2" class="data row4 col2" >2.50%</td>
      <td id="T_7a121_row4_col3" class="data row4 col3" >0.30</td>
      <td id="T_7a121_row4_col4" class="data row4 col4" >-26.2%</td>
      <td id="T_7a121_row4_col5" class="data row4 col5" >0.124904</td>
      <td id="T_7a121_row4_col6" class="data row4 col6" >0.699787</td>
      <td id="T_7a121_row4_col7" class="data row4 col7" >-0.337173</td>
    </tr>
    <tr>
      <th id="T_7a121_level0_row5" class="row_heading level0 row5" >2</th>
      <td id="T_7a121_row5_col0" class="data row5 col0" >20</td>
      <td id="T_7a121_row5_col1" class="data row5 col1" >1.500000</td>
      <td id="T_7a121_row5_col2" class="data row5 col2" >2.38%</td>
      <td id="T_7a121_row5_col3" class="data row5 col3" >0.28</td>
      <td id="T_7a121_row5_col4" class="data row5 col4" >-25.4%</td>
      <td id="T_7a121_row5_col5" class="data row5 col5" >0.124904</td>
      <td id="T_7a121_row5_col6" class="data row5 col6" >0.699787</td>
      <td id="T_7a121_row5_col7" class="data row5 col7" >-0.337173</td>
    </tr>
    <tr>
      <th id="T_7a121_level0_row6" class="row_heading level0 row6" >1</th>
      <td id="T_7a121_row6_col0" class="data row6 col0" >20</td>
      <td id="T_7a121_row6_col1" class="data row6 col1" >1.000000</td>
      <td id="T_7a121_row6_col2" class="data row6 col2" >2.21%</td>
      <td id="T_7a121_row6_col3" class="data row6 col3" >0.26</td>
      <td id="T_7a121_row6_col4" class="data row6 col4" >-29.2%</td>
      <td id="T_7a121_row6_col5" class="data row6 col5" >0.124904</td>
      <td id="T_7a121_row6_col6" class="data row6 col6" >0.699787</td>
      <td id="T_7a121_row6_col7" class="data row6 col7" >-0.337173</td>
    </tr>
    <tr>
      <th id="T_7a121_level0_row7" class="row_heading level0 row7" >0</th>
      <td id="T_7a121_row7_col0" class="data row7 col0" >20</td>
      <td id="T_7a121_row7_col1" class="data row7 col1" >0.500000</td>
      <td id="T_7a121_row7_col2" class="data row7 col2" >2.13%</td>
      <td id="T_7a121_row7_col3" class="data row7 col3" >0.25</td>
      <td id="T_7a121_row7_col4" class="data row7 col4" >-26.1%</td>
      <td id="T_7a121_row7_col5" class="data row7 col5" >0.124904</td>
      <td id="T_7a121_row7_col6" class="data row7 col6" >0.699787</td>
      <td id="T_7a121_row7_col7" class="data row7 col7" >-0.337173</td>
    </tr>
    <tr>
      <th id="T_7a121_level0_row8" class="row_heading level0 row8" >10</th>
      <td id="T_7a121_row8_col0" class="data row8 col0" >120</td>
      <td id="T_7a121_row8_col1" class="data row8 col1" >1.000000</td>
      <td id="T_7a121_row8_col2" class="data row8 col2" >1.78%</td>
      <td id="T_7a121_row8_col3" class="data row8 col3" >0.21</td>
      <td id="T_7a121_row8_col4" class="data row8 col4" >-20.2%</td>
      <td id="T_7a121_row8_col5" class="data row8 col5" >0.124904</td>
      <td id="T_7a121_row8_col6" class="data row8 col6" >0.699787</td>
      <td id="T_7a121_row8_col7" class="data row8 col7" >-0.337173</td>
    </tr>
    <tr>
      <th id="T_7a121_level0_row9" class="row_heading level0 row9" >3</th>
      <td id="T_7a121_row9_col0" class="data row9 col0" >40</td>
      <td id="T_7a121_row9_col1" class="data row9 col1" >0.500000</td>
      <td id="T_7a121_row9_col2" class="data row9 col2" >1.72%</td>
      <td id="T_7a121_row9_col3" class="data row9 col3" >0.20</td>
      <td id="T_7a121_row9_col4" class="data row9 col4" >-24.7%</td>
      <td id="T_7a121_row9_col5" class="data row9 col5" >0.124904</td>
      <td id="T_7a121_row9_col6" class="data row9 col6" >0.699787</td>
      <td id="T_7a121_row9_col7" class="data row9 col7" >-0.337173</td>
    </tr>
  </tbody>
</table>





```python
grid.head(10).style.format({
    'strat_CAGR': '{:.2%}',
    'strat_Sharpe': '{:.2f}',
    'strat_Drawdown': '{:.1%}',
    'bh_CAGR': '{:.2%}',
    'bh_Sharpe': '{:.2f}',
    'bh_Drawdown': '{:.1%}'
})
```




<style type="text/css">
</style>
<table id="T_dd296">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_dd296_level0_col0" class="col_heading level0 col0" >lookback</th>
      <th id="T_dd296_level0_col1" class="col_heading level0 col1" >entry_z</th>
      <th id="T_dd296_level0_col2" class="col_heading level0 col2" >strat_CAGR</th>
      <th id="T_dd296_level0_col3" class="col_heading level0 col3" >strat_Sharpe</th>
      <th id="T_dd296_level0_col4" class="col_heading level0 col4" >strat_Drawdown</th>
      <th id="T_dd296_level0_col5" class="col_heading level0 col5" >bh_CAGR</th>
      <th id="T_dd296_level0_col6" class="col_heading level0 col6" >bh_Sharpe</th>
      <th id="T_dd296_level0_col7" class="col_heading level0 col7" >bh_Drawdown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_dd296_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_dd296_row0_col0" class="data row0 col0" >20</td>
      <td id="T_dd296_row0_col1" class="data row0 col1" >0.500000</td>
      <td id="T_dd296_row0_col2" class="data row0 col2" >2.13%</td>
      <td id="T_dd296_row0_col3" class="data row0 col3" >0.25</td>
      <td id="T_dd296_row0_col4" class="data row0 col4" >-26.1%</td>
      <td id="T_dd296_row0_col5" class="data row0 col5" >12.49%</td>
      <td id="T_dd296_row0_col6" class="data row0 col6" >0.70</td>
      <td id="T_dd296_row0_col7" class="data row0 col7" >-33.7%</td>
    </tr>
    <tr>
      <th id="T_dd296_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_dd296_row1_col0" class="data row1 col0" >20</td>
      <td id="T_dd296_row1_col1" class="data row1 col1" >1.000000</td>
      <td id="T_dd296_row1_col2" class="data row1 col2" >2.21%</td>
      <td id="T_dd296_row1_col3" class="data row1 col3" >0.26</td>
      <td id="T_dd296_row1_col4" class="data row1 col4" >-29.2%</td>
      <td id="T_dd296_row1_col5" class="data row1 col5" >12.49%</td>
      <td id="T_dd296_row1_col6" class="data row1 col6" >0.70</td>
      <td id="T_dd296_row1_col7" class="data row1 col7" >-33.7%</td>
    </tr>
    <tr>
      <th id="T_dd296_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_dd296_row2_col0" class="data row2 col0" >20</td>
      <td id="T_dd296_row2_col1" class="data row2 col1" >1.500000</td>
      <td id="T_dd296_row2_col2" class="data row2 col2" >2.38%</td>
      <td id="T_dd296_row2_col3" class="data row2 col3" >0.28</td>
      <td id="T_dd296_row2_col4" class="data row2 col4" >-25.4%</td>
      <td id="T_dd296_row2_col5" class="data row2 col5" >12.49%</td>
      <td id="T_dd296_row2_col6" class="data row2 col6" >0.70</td>
      <td id="T_dd296_row2_col7" class="data row2 col7" >-33.7%</td>
    </tr>
    <tr>
      <th id="T_dd296_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_dd296_row3_col0" class="data row3 col0" >40</td>
      <td id="T_dd296_row3_col1" class="data row3 col1" >0.500000</td>
      <td id="T_dd296_row3_col2" class="data row3 col2" >1.72%</td>
      <td id="T_dd296_row3_col3" class="data row3 col3" >0.20</td>
      <td id="T_dd296_row3_col4" class="data row3 col4" >-24.7%</td>
      <td id="T_dd296_row3_col5" class="data row3 col5" >12.49%</td>
      <td id="T_dd296_row3_col6" class="data row3 col6" >0.70</td>
      <td id="T_dd296_row3_col7" class="data row3 col7" >-33.7%</td>
    </tr>
    <tr>
      <th id="T_dd296_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_dd296_row4_col0" class="data row4 col0" >40</td>
      <td id="T_dd296_row4_col1" class="data row4 col1" >1.000000</td>
      <td id="T_dd296_row4_col2" class="data row4 col2" >3.23%</td>
      <td id="T_dd296_row4_col3" class="data row4 col3" >0.38</td>
      <td id="T_dd296_row4_col4" class="data row4 col4" >-25.6%</td>
      <td id="T_dd296_row4_col5" class="data row4 col5" >12.49%</td>
      <td id="T_dd296_row4_col6" class="data row4 col6" >0.70</td>
      <td id="T_dd296_row4_col7" class="data row4 col7" >-33.7%</td>
    </tr>
    <tr>
      <th id="T_dd296_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_dd296_row5_col0" class="data row5 col0" >40</td>
      <td id="T_dd296_row5_col1" class="data row5 col1" >1.500000</td>
      <td id="T_dd296_row5_col2" class="data row5 col2" >2.85%</td>
      <td id="T_dd296_row5_col3" class="data row5 col3" >0.34</td>
      <td id="T_dd296_row5_col4" class="data row5 col4" >-20.3%</td>
      <td id="T_dd296_row5_col5" class="data row5 col5" >12.49%</td>
      <td id="T_dd296_row5_col6" class="data row5 col6" >0.70</td>
      <td id="T_dd296_row5_col7" class="data row5 col7" >-33.7%</td>
    </tr>
    <tr>
      <th id="T_dd296_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_dd296_row6_col0" class="data row6 col0" >60</td>
      <td id="T_dd296_row6_col1" class="data row6 col1" >0.500000</td>
      <td id="T_dd296_row6_col2" class="data row6 col2" >2.50%</td>
      <td id="T_dd296_row6_col3" class="data row6 col3" >0.30</td>
      <td id="T_dd296_row6_col4" class="data row6 col4" >-26.2%</td>
      <td id="T_dd296_row6_col5" class="data row6 col5" >12.49%</td>
      <td id="T_dd296_row6_col6" class="data row6 col6" >0.70</td>
      <td id="T_dd296_row6_col7" class="data row6 col7" >-33.7%</td>
    </tr>
    <tr>
      <th id="T_dd296_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_dd296_row7_col0" class="data row7 col0" >60</td>
      <td id="T_dd296_row7_col1" class="data row7 col1" >1.000000</td>
      <td id="T_dd296_row7_col2" class="data row7 col2" >2.57%</td>
      <td id="T_dd296_row7_col3" class="data row7 col3" >0.30</td>
      <td id="T_dd296_row7_col4" class="data row7 col4" >-20.9%</td>
      <td id="T_dd296_row7_col5" class="data row7 col5" >12.49%</td>
      <td id="T_dd296_row7_col6" class="data row7 col6" >0.70</td>
      <td id="T_dd296_row7_col7" class="data row7 col7" >-33.7%</td>
    </tr>
    <tr>
      <th id="T_dd296_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_dd296_row8_col0" class="data row8 col0" >60</td>
      <td id="T_dd296_row8_col1" class="data row8 col1" >1.500000</td>
      <td id="T_dd296_row8_col2" class="data row8 col2" >4.12%</td>
      <td id="T_dd296_row8_col3" class="data row8 col3" >0.49</td>
      <td id="T_dd296_row8_col4" class="data row8 col4" >-15.2%</td>
      <td id="T_dd296_row8_col5" class="data row8 col5" >12.49%</td>
      <td id="T_dd296_row8_col6" class="data row8 col6" >0.70</td>
      <td id="T_dd296_row8_col7" class="data row8 col7" >-33.7%</td>
    </tr>
    <tr>
      <th id="T_dd296_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_dd296_row9_col0" class="data row9 col0" >120</td>
      <td id="T_dd296_row9_col1" class="data row9 col1" >0.500000</td>
      <td id="T_dd296_row9_col2" class="data row9 col2" >-0.34%</td>
      <td id="T_dd296_row9_col3" class="data row9 col3" >-0.04</td>
      <td id="T_dd296_row9_col4" class="data row9 col4" >-24.0%</td>
      <td id="T_dd296_row9_col5" class="data row9 col5" >12.49%</td>
      <td id="T_dd296_row9_col6" class="data row9 col6" >0.70</td>
      <td id="T_dd296_row9_col7" class="data row9 col7" >-33.7%</td>
    </tr>
  </tbody>
</table>





```python
# this will give us the highest Sharpe
best = grid.loc[grid['strat_Sharpe'].idxmax()]
best
```




    lookback          60.000000
    entry_z            1.500000
    strat_CAGR         0.041168
    strat_Sharpe       0.487433
    strat_Drawdown    -0.151839
    bh_CAGR            0.124904
    bh_Sharpe          0.699787
    bh_Drawdown       -0.337173
    Name: 8, dtype: float64




```python
# this will give us the top 3 candidates which will also meet our drawdown tolerance ensuring its not too 
# high (a high drawdown will cause us to lose more money at the lowest point in our investing)

good = grid[(grid['strat_Sharpe'] >= 0.30) & (grid['strat_Drawdown'] >= -0.25)]
good.sort_values('strat_Sharpe', ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lookback</th>
      <th>entry_z</th>
      <th>strat_CAGR</th>
      <th>strat_Sharpe</th>
      <th>strat_Drawdown</th>
      <th>bh_CAGR</th>
      <th>bh_Sharpe</th>
      <th>bh_Drawdown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>60</td>
      <td>1.5</td>
      <td>0.041168</td>
      <td>0.487433</td>
      <td>-0.151839</td>
      <td>0.124904</td>
      <td>0.699787</td>
      <td>-0.337173</td>
    </tr>
    <tr>
      <th>5</th>
      <td>40</td>
      <td>1.5</td>
      <td>0.028481</td>
      <td>0.335643</td>
      <td>-0.203050</td>
      <td>0.124904</td>
      <td>0.699787</td>
      <td>-0.337173</td>
    </tr>
    <tr>
      <th>7</th>
      <td>60</td>
      <td>1.0</td>
      <td>0.025677</td>
      <td>0.303136</td>
      <td>-0.209183</td>
      <td>0.124904</td>
      <td>0.699787</td>
      <td>-0.337173</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
