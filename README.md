# Statistical-Arbitrage-in-Indian-Equities

Implementation of Avellaneda and Lee statistical arbitrage strategy in Indian equity market using NIFTY 100 constituents. For detailed explanation of the strategy and results, please refer to the accompanying PDF report.

## Project Structure

```
├── data_preprocessing.py     # Data downloading and preprocessing functions
├── pca_factors.py            # PCA Decomposition and analysis
├── eigenportfolio_utils.py   # Eigenportfolio construction and visualization
├── ou_params.py              # Ornstein-Uhlembeck parameter estimation
├── backtest.py               # Backtesting implementation
├── visualization.ipynb       # Visualization notebook for analysis
├── StatArb_in_Indian_Equities.pdf  # Detailed report and analysis
├── plots	              # Plots
└── requirements.txt
```

## Files and Functions

### data_preprocessing.py
Core functions for data handling and preparation:
- `download_stock_data(ticker_list, start_date, end_date)`: Downloads historical stock data from Yahoo Finance, handling data cleaning and missing values
- `calculate_daily_returns(price_data)`: Computes daily percentage returns from price data
- `calculate_volume_weighted_returns(volume_data, daily_returns, lookback_period=10)`: Calculates volume-weighted returns using trading volume data
- `save_to_pkl(result, file_name, folder_name="data")`: Saves results to pickle files
- `load_from_pkl(file_name, folder_name="data")`: Loads data from pickle files

### pca_factors.py
PCA decompostion and analysis functions:
- `perform_pca(data, window=252, n_components=None, variance_threshold=None)`: Performs rolling window PCA on return data
- `plot_explained_variance(pca_results, date=None, n_components=None)`: Visualizes explained variance distribution
- `analyze_pca_components_over_time(pca_results, start_date=None, end_date=None, variance_threshold=None, n_components=None)`: Analyzes PCA components' evolution over time

### eigenportfolio_utils.py
Functions for eigenportfolio construction and analysis:
- `construct_eigenportfolios(pca_results, returns_data)`: Constructs eigenportfolios
- `plot_eigenportfolio_weights(eigenportfolio, stock_tickers)`: Visualizes eigenportfolio weights
- `compute_eigenportfolio_returns(eigenportfolio_results, returns_data)`: Calculates eigenportfolio returns
- `plot_eigenportfolio_returns(eigenportfolio_returns_results, nifty_returns)`: Plots cumulative returns comparison

### ou_params.py
Ornstein-Uhlembeck process analysis:
- `calculate_ou_parameters(returns_data, eigenportfolio_returns, tickers)`: Estimates Ornstein-Uhlenbeck parameters through two-step regression process
- `create_adf_heatmap(ou_parameters, window_size=60, alpha=0.05)`: Creates heatmap of ADF test p-values
- `analyze_and_plot_taus(taus, fast_threshold=8.4)`: Analyzes characteristic time-scale to meanreversion data
- `plot_s_scores(s_score_data, start_date=None, end_date=None)`: Plots s-scores time series

### backtest.py
Backtesting implementation:
- `backtesting(returns_data, ou_parameters, eigenportfolios)`: Main backtesting function implementing the trading strategy
- `calculate_sharpe_ratios(backtest_results)`: Calculates Sharpe ratios for strategy evaluation
- `plot_cumulative_return(backtest_results, label=None)`: Plots cumulative returns

### visualization.ipynb
Jupyter notebook containing visualizations and analysis of results from the above scripts.

## Requirements

Main dependencies:
```
numpy>=2.2.0
pandas>=2.2.3
scikit-learn>=1.6.0
scipy>=1.14.1
statsmodels>=0.14.4
yfinance>=0.2.50
matplotlib>=3.10.0
seaborn>=0.13.2
tqdm>=4.67.1
jupyter>=1.0.0
```

For complete list of dependencies, see `requirements.txt`

## Usage

Run the scripts in the following order:
1. `data_preprocessing.py`
2. `pca_factors.py`
3. `eigenportfolio_utils.py`
4. `ou_params.py`
5. `backtest.py`


## References

1. Avellaneda, M., & Lee, J. (2008). Statistical Arbitrage in the U.S. Equities Market. Quantitative Finance, 10(8), 761–782.
2. Krauss, C. (2017). Statistical Arbitrage Pairs Trading Strategies: Review and Outlook. Journal of Economic Surveys, 31(2), 513–545.
3. Di Nosse, D. M. (2022). Application of score-driven models in statistical arbitrage. (Master's thesis). Department of Physics, University of Pisa, Pisa, Italy.
4. Soo, C., Lian, Z., Yang, H., & Lou, J. (2017). Statistical Arbitrage. MS&E448 Project, Stanford University.
5. Tsay, R. S. (2010). Analysis of Financial Time Series (3rd ed.). Wiley Series in Probability and Statistics.
6. Alexander, C. (2001). Market Models: A Guide to Financial Data Analysis. Wiley.
