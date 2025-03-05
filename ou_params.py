from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from data_preprocessing import save_to_pkl, load_from_pkl

def calculate_ou_parameters(returns_data: pd.DataFrame,
                          eigenportfolio_returns: dict,
                          tickers: list,
                          lookback_period: int = 60,
                          mean_reversion_threshold: float = 252/30,
                          save_prefix: str = None):
    """
    Estimates Ornstein-Uhlembeck parameters through a two-step regression process:
    1. Regresses stock returns against eigenportfolio returns to obtain residuals:
       R_{it} = Σ_{j=1}^m β_{ijt} F_{jt} + X_{it}
    2. Estimates OU parameters through  one-lag regression.
    
    Parameters
    ----------
    returns_data : pandas.DataFrame
        Daily stock returns with datetime index
    eigenportfolio_returns : dict
        Dictionary containing:
        - returns: array of shape (n_valid_dates, max_factors) for F_{jt}
        - dates: array of dates for the returns
    tickers : list
        List of stock tickers to analyze
    lookback_period : int, default=60
        Rolling window length for parameter estimation
    mean_reversion_threshold : float, default=252/30
        Minimum κ_i threshold for mean reversion speed
    save_prefix : str, optional
        Prefix for saving output files
        
    Returns
    -------
    dict
        Dictionary containing:
        - s_scores: DataFrame of standardized deviations from equilibrium
        - beta_tensor: Array of shape (n_dates, n_stocks, n_factors) for β_{ijt}
        - residuals: DataFrame of ε_{it} values
        - taus: DataFrame of mean reversion times τ_i = 1/κ_i
        - dates: Array of dates corresponding to all results
    """
    eigen_dates = pd.to_datetime(eigenportfolio_returns['dates'])
    common_dates = returns_data.index.intersection(eigen_dates)
    aligned_returns = returns_data.loc[common_dates]
    eigen_mask = np.isin(eigen_dates, common_dates)
    aligned_eigen = eigenportfolio_returns['returns'][eigen_mask]

    trading_days = len(common_dates[lookback_period - 1:])
    n_factors = aligned_eigen.shape[1]

    if not all(ticker in aligned_returns.columns for ticker in tickers):
        raise ValueError("Not all provided tickers are present in returns data")
    
    s_scores = pd.DataFrame(index=common_dates[lookback_period - 1:], columns=tickers)
    taus = pd.DataFrame(index=common_dates[lookback_period - 1:], columns=tickers)
    residuals = pd.DataFrame(index=common_dates[lookback_period - 1:], columns=tickers)

    beta_tensor = np.zeros((trading_days, len(tickers), n_factors))
    
    for t_idx, t in enumerate(common_dates[lookback_period - 1:]):
        window_slice = slice(t_idx, t_idx + lookback_period)
        window_returns = aligned_returns.iloc[window_slice]
        window_eigen = aligned_eigen[window_slice]

        ou_parameters = pd.DataFrame(index=tickers,
                                   columns=['a', 'b', 'Var(zeta)', 'kappa', 'm',
                                          'sigma', 'sigma_eq', 'residual'])
        
        for stock_idx, stock in enumerate(tickers):
            model1 = LinearRegression().fit(window_eigen, window_returns[stock])
            beta_coefficients = model1.coef_
            residual_returns = window_returns[stock].values - model1.predict(window_eigen)

            beta_tensor[t_idx, stock_idx, :] = beta_coefficients
            residuals.loc[t, stock] = residual_returns[-1]
            
            Xk = residual_returns.cumsum()
            X_ou = Xk[:-1].reshape(-1, 1)
            y_ou = Xk[1:]
            model2 = LinearRegression().fit(X_ou, y_ou)
            
            a = model2.intercept_
            b = model2.coef_[0]
            zeta = y_ou - model2.predict(X_ou)

            kappa = -np.log(b) * 252
            m = a / (1 - b)
            sigma = np.sqrt(np.var(zeta) * 2 * kappa / (1 - b**2))
            sigma_eq = np.sqrt(np.var(zeta) / (1 - b**2))
            
            if kappa > mean_reversion_threshold:
                ou_parameters.loc[stock] = [a, b, np.var(zeta), kappa, m,
                                         sigma, sigma_eq, residual_returns[-1]]
        
        if not ou_parameters.empty:
            ou_parameters['m_bar'] = (ou_parameters['a'] / (1 - ou_parameters['b']) -
                                    ou_parameters['a'].mean() / (1 - ou_parameters['b'].mean()))
            ou_parameters['s'] = -ou_parameters['m_bar'] / ou_parameters['sigma_eq']
            s_scores.loc[t] = ou_parameters['s']
            
            ou_parameters['tau'] = 1 / ou_parameters['kappa'] * 100
            taus.loc[t] = ou_parameters['tau']
    
    results = {
        's_scores': s_scores,
        'beta_tensor': beta_tensor,
        'residuals': residuals,
        'taus': taus,
        'dates': common_dates[lookback_period - 1:]
    }
    
    if save_prefix:
        try:
            save_to_pkl(results, f'ou_parameters_{save_prefix}.pkl')
        except Exception as e:
            print(f"Error saving ou parameters: {e}")    
            
    return results


def create_adf_heatmap(ou_parameters, window_size=60, alpha=0.05):
    """
    Create a simple heatmap of ADF test p-values across time and stocks,
    with alternating stock labels and reformatted date display.
    """
    residuals_df = ou_parameters['residuals']
    dates = residuals_df.index
    
    total_periods = len(residuals_df)
    n_windows = total_periods - window_size + 1
    stock_names = residuals_df.columns
    
    p_values = np.zeros((n_windows, len(stock_names)))
    
    print("Computing ADF tests...")
    for t in tqdm(range(n_windows)):
        window_data = residuals_df.iloc[t:t+window_size]
        for s in range(len(stock_names)):
            try:
                result = adfuller(window_data.iloc[:, s])
                p_values[t, s] = result[1]
            except:
                p_values[t, s] = np.nan
    

    plt.close('all')
    plt.figure(figsize=(14, 10))  

    y_labels = [label if i % 2 == 0 else '' for i, label in enumerate(stock_names)]
    
    ax = sns.heatmap(p_values.T,
                     cmap='YlOrRd_r',
                     vmin=0,
                     vmax=0.05,
                     cbar_kws={'label': 'p-value'},
                     yticklabels=y_labels)
    

    window_dates = dates[window_size-1:]
    tick_positions = range(0, len(window_dates), 126)  # 126 trading days ≈ 6 months
    tick_labels = [window_dates[i].strftime('%b %Y') for i in tick_positions]  # Changed date format
    plt.xticks(tick_positions, tick_labels, rotation=90)  # Changed rotation to 90

    rejection_rate = (np.sum(p_values < alpha) / np.prod(p_values.shape)) * 100
    
    stats_text = (f'Rejected $H_0$: {rejection_rate:.1f}%\n'
              f'Failed to Reject $H_0$: {100-rejection_rate:.1f}%')
    plt.text(0.5, -0.2,
         stats_text,
         transform=ax.transAxes,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
         horizontalalignment='center',
         verticalalignment='top',
         fontsize=12,
         usetex=False) 
    plt.subplots_adjust(bottom=0.25)

    plt.xlabel('Time')
    plt.ylabel('Stocks')
    plt.title('ADF Test P-values\n(Dark = Stationary)')
    
    plt.tight_layout()
    fig = plt.gcf() 
    return fig


def analyze_and_plot_taus(taus, fast_threshold=8.4):
    """
    Analyze the characteristic time-scale (τ) data and plot its distribution.

    Returns:
    --------
        Descriptive statistics of the characteristic time-scale (τ).
    """
    tau_series = taus.stack()  # Convert to a single Series
    descriptive_stats = {
        "Maximum": tau_series.max(),
        "75th Percentile": tau_series.quantile(0.75),
        "Median": tau_series.median(),
        "25th Percentile": tau_series.quantile(0.25),
        "Minimum": tau_series.min()
    }

    fast_days_percentage = (tau_series < fast_threshold).mean() * 100

    print("Descriptive Statistics on the Mean-Reversion Time (τ):")
    for key, value in descriptive_stats.items():
        print(f"{key}: {value:.2f} days")
    print(f"Fast Days (< {fast_threshold} days): {fast_days_percentage:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.margins(x=0)
    plt.grid(axis='y')
    sns.histplot(tau_series.dropna(), bins=100, kde=True)

    stats_text = (
        f"Maximum: {descriptive_stats['Maximum']:.2f} days\n"
        f"75th Percentile: {descriptive_stats['75th Percentile']:.2f} days\n"
        f"Median: {descriptive_stats['Median']:.2f} days\n"
        f"25th Percentile: {descriptive_stats['25th Percentile']:.2f} days\n"
        f"Minimum: {descriptive_stats['Minimum']:.2f} days\n"
        f"Fast Days: {fast_days_percentage:.2f}%"
    )

    plt.xlabel('Characteristic Time to Mean-Reversion (τ)')
    plt.ylabel('Counts')
    plt.title('Empirical Distribution of Characteristic Time to Mean-Reversion (τ)')

    plt.legend([stats_text], loc='upper right')
    plt.show()

def plot_s_scores(s_score_data, start_date=None, end_date=None):
    """
    Plots S-Scores time series data with  an optional date range.
    
    """
    if start_date or end_date:
        s_score_data = s_score_data.loc[start_date:end_date]
        
    plt.figure(figsize=(12, 6))
    s_score_data.plot(color='black')
    plt.xticks(rotation=90)
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=3))
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
    
    for level in range(-3, 4):
        plt.axhline(y=level, color='gray', linestyle='--', linewidth=0.5)
    
    plt.xlabel('Time')
    plt.ylabel('S-Scores')
    plt.margins(x=0.001)
    plt.tight_layout()
    plt.show()


def main():
    daily_returns = load_from_pkl('daily_returns.pkl')
    volume_weighted_returns = load_from_pkl('volume_weighted_returns.pkl')
    tickers = load_from_pkl('tickers.pkl')
    eigenportfolio_returns_dr1=load_from_pkl('eigenportfolio_returns_dr1.pkl')
    eigenportfolio_returns_dr15=load_from_pkl('eigenportfolio_returns_dr15.pkl')
    eigenportfolio_returns_dr55=load_from_pkl('eigenportfolio_returns_dr55.pkl')
    eigenportfolio_returns_dr75=load_from_pkl('eigenportfolio_returns_dr75.pkl')
    eigenportfolio_returns_vw15=load_from_pkl('eigenportfolio_returns_vw15.pkl')
    eigenportfolio_returns_vw55=load_from_pkl('eigenportfolio_returns_vw55.pkl')
    eigenportfolio_returns_vw75=load_from_pkl('eigenportfolio_returns_vw75.pkl')

    # Calculate Ornstein-Uhlenbeck parameters for daily returns with 1 component
    ou_parameters_dr1 = calculate_ou_parameters(daily_returns, eigenportfolio_returns_dr1, tickers, save_prefix="dr1")

    # Calculate Ornstein-Uhlenbeck parameters for daily returns with 15 principal components
    ou_parameters_dr15 = calculate_ou_parameters(daily_returns, eigenportfolio_returns_dr15, tickers, save_prefix="dr15")

    # Calculate Ornstein-Uhlenbeck parameters for daily returns with 55% explained variance
    ou_parameters_dr55 = calculate_ou_parameters(daily_returns, eigenportfolio_returns_dr55, tickers, save_prefix="dr55")

     # Calculate Ornstein-Uhlenbeck parameters for daily returns with 75% explained variance
    ou_parameters_dr75 = calculate_ou_parameters(daily_returns, eigenportfolio_returns_dr75, tickers, save_prefix="dr75")

    # Calculate Ornstein-Uhlenbeck parameters for volume-weighted returns with 15 principal components
    ou_parameters_vw15 = calculate_ou_parameters(volume_weighted_returns, eigenportfolio_returns_vw15, tickers, save_prefix="vw15")

    # Calculate Ornstein-Uhlenbeck parameters for volume-weighted returns with 55% explained variance
    ou_parameters_vw55 = calculate_ou_parameters(volume_weighted_returns, eigenportfolio_returns_vw55, tickers, save_prefix="vw55")

    # Calculate Ornstein-Uhlenbeck parameters for volume-weighted returns with 55% explained variance
    ou_parameters_vw75 = calculate_ou_parameters(volume_weighted_returns, eigenportfolio_returns_vw75, tickers, save_prefix="vw75")

    print("\nOrnstein-Uhlenbeck parameters computed and saved successfully.")


if __name__ == "__main__":
    main()

    