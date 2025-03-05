import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_preprocessing import save_to_pkl, load_from_pkl

def backtesting(returns_data: pd.DataFrame,
           ou_parameters: dict,
           eigenportfolios: dict,
           s_bo: float = 1.2,
           s_so: float = 1.2,
           s_bc: float = 0.75 ,
           s_sc: float= 0.5,
           leverage: float = 1.0,
           epsilon: float = 0.0005,
           save_prefix: Optional[str] = None) -> dict:
    '''
    Statistical arbitrage backtesting strategy based on Avellaneda-Lee (2008).
    
    Parameters
    ----------
    returns_data : pd.DataFrame
        DataFrame of daily stock returns (dates × tickers)
    ou_parameters : dict
        Dictionary containing:
        - s_scores: DataFrame with s-scores
        - beta_tensor: Array of shape (n_dates, n_stocks, n_factors)
        - dates: Array of reference dates
    eigenportfolios : dict
        Dictionary containing:
        - eigenweights: Array of shape (n_dates, n_stocks, max_components)
        - dates: Array of dates
    s_bo, s_so, s_bc, s_sc : float
        Thresholds for buy to open, short to open, close long, close short
    epsilon : float
        Slippage parameter
        
    Returns
    -------
    dict
        Dictionary containing:
        - dates: Array of backtesting dates
        - pnl: Array of PnL values
        - positions: Array of position percentages (long, short, closed)
        - fees: Array of transaction fees
    '''
    s_scores = ou_parameters['s_scores']
    beta_tensor = ou_parameters['beta_tensor']  # shape: (n_dates, n_stocks, n_factors)
    reference_dates = ou_parameters['dates']
    eigenweights = eigenportfolios['eigenweights']  # shape: (n_dates, n_stocks, max_components)
    
    assert beta_tensor.shape[2] == eigenweights.shape[2], "Number of factors must match number of components"

    returns_aligned = returns_data.loc[reference_dates]

    n_days = len(reference_dates)
    n_stocks = returns_aligned.shape[1]
    
    PnL = np.zeros(n_days)
    PnL[0] = 1.00
    
    lambda_t = leverage/n_stocks #positions size
    
    daily_PnL = np.zeros(shape=(n_days, n_stocks))
    state = np.array(['c' for _ in range(n_stocks)])
    day_counter_long = np.zeros(shape=n_stocks, dtype=int)
    day_counter_short = np.zeros(shape=n_stocks, dtype=int)
    perc_positions = np.zeros(shape=(n_days, 3))
    invest_amount = np.zeros(shape=(n_days + 1))
    fees = np.zeros(shape=n_days)
    
    for day in tqdm(range(n_days - 1)):
        counter_no_trades = 0
        
        for stock_idx, stock in enumerate(returns_aligned.columns):
            if pd.isna(s_scores.iloc[day, stock_idx]):
                counter_no_trades += 1
                continue
                
            score = s_scores.iloc[day, stock_idx]
            
            factor_exposure = np.dot(beta_tensor[day, stock_idx, :], 
                                   eigenweights[day, stock_idx, :])
            
            # backtesting logic
            if score < -s_bo and (state[stock_idx] == 'c'):
                # Open long position
                state[stock_idx] = 'l'
                k = PnL[day] * lambda_t / (1 + factor_exposure)
                
                stock_return = returns_aligned.iloc[day + 1, stock_idx]
                factor_return = np.dot(
                    beta_tensor[day, stock_idx, :],
                    np.dot(eigenweights[day, :, :].T,
                          returns_aligned.iloc[day + 1].values)
                )
                daily_PnL[day, stock_idx] = k * (stock_return - factor_return)
                invest_amount[day + 1] = factor_exposure
                
            elif (day > 0) and (score < -s_sc) and (state[stock_idx] == 'l'):
                # Maintain long position
                day_counter_long[stock_idx] += 1
                prev_day = day - day_counter_long[stock_idx]
                
                prev_exposure = np.dot(beta_tensor[prev_day, stock_idx, :],
                                     eigenweights[prev_day, stock_idx, :])
                k = PnL[prev_day] * lambda_t / (1 + prev_exposure)
                
                stock_return = returns_aligned.iloc[day + 1, stock_idx]
                factor_return = np.dot(
                    beta_tensor[prev_day, stock_idx, :],
                    np.dot(eigenweights[prev_day, :, :].T,
                          returns_aligned.iloc[day + 1].values)
                )
                
                daily_PnL[day, stock_idx] = k * (stock_return - factor_return)
                
            elif score > s_so and (state[stock_idx] == 'c'):
                # Open short position
                state[stock_idx] = 's'
                k = PnL[day] * lambda_t / (1 + factor_exposure)
                
                stock_return = returns_aligned.iloc[day + 1, stock_idx]
                factor_return = np.dot(
                    beta_tensor[day, stock_idx, :],
                    np.dot(eigenweights[day, :, :].T,
                          returns_aligned.iloc[day + 1].values)
                )
                
                daily_PnL[day, stock_idx] = k * (-stock_return + factor_return)
                invest_amount[day + 1] = factor_exposure
                
            elif (day > 0) and (score > s_bc) and (state[stock_idx] == 's'):
                # Maintain short position
                day_counter_short[stock_idx] += 1
                prev_day = day - day_counter_short[stock_idx]
                
                prev_exposure = np.dot(beta_tensor[prev_day, stock_idx, :],
                                     eigenweights[prev_day, stock_idx, :])
                k = PnL[prev_day] * lambda_t/ (1 + prev_exposure)
                
                stock_return = returns_aligned.iloc[day + 1, stock_idx]
                factor_return = np.dot(
                    beta_tensor[prev_day, stock_idx, :],
                    np.dot(eigenweights[prev_day, :, :].T,
                          returns_aligned.iloc[day + 1].values)
                )
                
                daily_PnL[day, stock_idx] = k * (-stock_return + factor_return)
                
            elif (day > 0) and (score > -s_sc) and (state[stock_idx] == 'l'):
                # Close long position
                day_counter_long[stock_idx] = 0
                state[stock_idx] = 'c'
                daily_PnL[day, stock_idx] = 0.0
                
            elif (day > 0) and (score < s_bc) and (state[stock_idx] == 's'):
                # Close short position
                day_counter_short[stock_idx] = 0
                state[stock_idx] = 'c'
                daily_PnL[day, stock_idx] = 0.0
                
            else:
                counter_no_trades += 1
                continue
        
        perc_positions[day, 0] = np.count_nonzero(state == 'l') / n_stocks  # long
        perc_positions[day, 1] = np.count_nonzero(state == 's') / n_stocks  # short
        perc_positions[day, 2] = np.count_nonzero(state == 'c') / n_stocks  # closed
        
        fees[day] = np.abs(invest_amount[day + 1] - invest_amount[day]).sum() * epsilon
        PnL[day + 1] = PnL[day] + daily_PnL[day, :].sum() - fees[day]
    
    results = {
        'dates': reference_dates,
        'pnl': PnL,
        'positions': perc_positions,
        'fees': fees
    }

    if save_prefix:
        try:
            save_to_pkl(results, f'backtest_{save_prefix}.pkl')
        except Exception as e:
            print(f"Error saving backtesting results: {e}")
    
    return results

def calculate_sharpe_ratios(backtest_results):
    """
    Calculate annual and since-inception Sharpe ratios for a trading strategy compared to Benchmark Nifty 100 index.

    """
    strategy_returns = pd.Series(
        np.diff(backtest_results['pnl']) / backtest_results['pnl'][:-1],
        index=pd.to_datetime(backtest_results['dates'][1:])
    )

    nifty_returns = load_from_pkl('nifty_returns.pkl')
    nifty_series = nifty_returns['^CNX100']

    common_dates = strategy_returns.index.intersection(nifty_series.index)
    excess_returns = strategy_returns[common_dates] - nifty_series[common_dates]

    annual_sharpe = {}
    for year in range(common_dates[0].year, common_dates[-1].year + 1):
        mask = (excess_returns.index.year == year)
        if mask.any():
            annual_ret = excess_returns[mask]
            annual_sharpe[year] = np.sqrt(252) * (annual_ret.mean() / annual_ret.std())

    since_inception_sharpe = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    
    return {
        'annual_sharpe_ratios': annual_sharpe,
        'since_inception_sharpe': since_inception_sharpe,
        'excess_returns': excess_returns
    }

def plot_cumulative_return(backtest_results, label=None):
   """Plots cumulative returns from backtest """
   strategy_returns = pd.Series(
       np.diff(backtest_results['pnl']) / backtest_results['pnl'][:-1],
       index=pd.to_datetime(backtest_results['dates'][1:])
   )
   cum_returns = (1 + strategy_returns).cumprod() - 1
   plt.plot(cum_returns.index, cum_returns.values, label=label or 'Strategy')
   return cum_returns

def main():
    daily_returns = load_from_pkl('daily_returns.pkl')
    volume_weighted_returns = load_from_pkl('volume_weighted_returns.pkl')
    tickers = load_from_pkl('tickers.pkl')
    eigenportfolios_dr1=load_from_pkl('eigenportfolios_dr1.pkl')
    eigenportfolios_dr15=load_from_pkl('eigenportfolios_dr15.pkl')
    eigenportfolios_dr55=load_from_pkl('eigenportfolios_dr55.pkl')
    eigenportfolios_dr75=load_from_pkl('eigenportfolios_dr75.pkl')
    eigenportfolios_vw15=load_from_pkl('eigenportfolios_vw15.pkl')
    eigenportfolios_vw55=load_from_pkl('eigenportfolios_vw55.pkl')
    eigenportfolios_vw75=load_from_pkl('eigenportfolios_vw75.pkl')
    ou_parameters_dr1=load_from_pkl('ou_parameters_dr1.pkl')
    ou_parameters_dr15=load_from_pkl('ou_parameters_dr15.pkl')
    ou_parameters_dr55=load_from_pkl('ou_parameters_dr55.pkl')
    ou_parameters_dr75=load_from_pkl('ou_parameters_dr75.pkl')
    ou_parameters_vw15=load_from_pkl('ou_parameters_vw15.pkl')
    ou_parameters_vw55=load_from_pkl('ou_parameters_vw55.pkl')
    ou_parameters_vw75=load_from_pkl('ou_parameters_vw75.pkl')


    backtest_dr1 = backtesting(
            returns_data=daily_returns,
            ou_parameters=ou_parameters_dr1,
            eigenportfolios=eigenportfolios_dr1,
            save_prefix= "dr1"
        )

    backtest_dr15 = backtesting(
            returns_data=daily_returns,
            ou_parameters=ou_parameters_dr15,
            eigenportfolios=eigenportfolios_dr15,
            save_prefix= "dr15"
        )

    backtest_dr55 = backtesting(
            returns_data=daily_returns,
            ou_parameters=ou_parameters_dr55,
            eigenportfolios=eigenportfolios_dr55,
            save_prefix= "dr55"
        )

    backtest_dr75 = backtesting(
            returns_data=daily_returns,
            ou_parameters=ou_parameters_dr75,
            eigenportfolios=eigenportfolios_dr75,
            save_prefix= "dr75"
        )
    
    backtest_vw15 = backtesting(
            returns_data=volume_weighted_returns,
            ou_parameters=ou_parameters_vw15,
            eigenportfolios=eigenportfolios_vw15,
            save_prefix= "vw15"
        )

    backtest_vw55 = backtesting(
            returns_data=volume_weighted_returns,
            ou_parameters=ou_parameters_vw55,
            eigenportfolios=eigenportfolios_vw55,
            save_prefix= "vw55"
        )

    backtest_vw75 = backtesting(
            returns_data=volume_weighted_returns,
            ou_parameters=ou_parameters_vw75,
            eigenportfolios=eigenportfolios_vw75,
            save_prefix= "vw75"
        )
    
    # 2x leverage

    backtest_dr1_2x = backtesting(
            returns_data=daily_returns,
            ou_parameters=ou_parameters_dr1,
            eigenportfolios=eigenportfolios_dr1,
            leverage =2.0,
            save_prefix= "dr1_2x"
        )

    backtest_dr15_2x = backtesting(
            returns_data=daily_returns,
            ou_parameters=ou_parameters_dr15,
            eigenportfolios=eigenportfolios_dr15,
            leverage =2.0,
            save_prefix= "dr15_2x"
        )

    backtest_dr55_2x = backtesting(
            returns_data=daily_returns,
            ou_parameters=ou_parameters_dr55,
            eigenportfolios=eigenportfolios_dr55,
            leverage =2.0,
            save_prefix= "dr55_2x"
        )

    backtest_dr75_2x = backtesting(
            returns_data=daily_returns,
            ou_parameters=ou_parameters_dr75,
            eigenportfolios=eigenportfolios_dr75,
            leverage =2.0,
            save_prefix= "dr75_2x"
        )
    
    backtest_vw15_2x = backtesting(
            returns_data=volume_weighted_returns,
            ou_parameters=ou_parameters_vw15,
            eigenportfolios=eigenportfolios_vw15,
            leverage =2.0,
            save_prefix= "vw15_2x"
        )

    backtest_vw55_2x = backtesting(
            returns_data=volume_weighted_returns,
            ou_parameters=ou_parameters_vw55,
            eigenportfolios=eigenportfolios_vw55,
            leverage =2.0,
            save_prefix= "vw55_2x"
        )

    backtest_vw75_2x = backtesting(
            returns_data=volume_weighted_returns,
            ou_parameters=ou_parameters_vw75,
            eigenportfolios=eigenportfolios_vw75,
            leverage =2.0,
            save_prefix= "vw75_2x"
        )
    
    #3x leverage
    backtest_dr1_3x = backtesting(
            returns_data=daily_returns,
            ou_parameters=ou_parameters_dr1,
            eigenportfolios=eigenportfolios_dr1,
            leverage = 3.0,
            save_prefix= "dr1_3x"
        )

    backtest_dr15_3x = backtesting(
            returns_data=daily_returns,
            ou_parameters=ou_parameters_dr15,
            eigenportfolios=eigenportfolios_dr15,
            leverage = 3.0,
            save_prefix= "dr15_3x"
        )

    backtest_dr55_3x= backtesting(
            returns_data=daily_returns,
            ou_parameters=ou_parameters_dr55,
            eigenportfolios=eigenportfolios_dr55,
            leverage = 3.0,
            save_prefix= "dr55_3x"
        )

    backtest_dr7_3x = backtesting(
            returns_data=daily_returns,
            ou_parameters=ou_parameters_dr75,
            eigenportfolios=eigenportfolios_dr75,
            leverage = 3.0,
            save_prefix= "dr75_3x"
        )
    
    backtest_vw15_3x = backtesting(
            returns_data=volume_weighted_returns,
            ou_parameters=ou_parameters_vw15,
            eigenportfolios=eigenportfolios_vw15,
            leverage = 3.0,
            save_prefix= "vw15_3x"
        )

    backtest_vw55_3x = backtesting(
            returns_data=volume_weighted_returns,
            ou_parameters=ou_parameters_vw55,
            eigenportfolios=eigenportfolios_vw55,
            leverage = 3.0,
            save_prefix= "vw55_3x"
        )

    backtest_vw75_3x = backtesting(
            returns_data=volume_weighted_returns,
            ou_parameters=ou_parameters_vw75,
            eigenportfolios=eigenportfolios_vw75,
            leverage = 3.0,
            save_prefix= "vw75_3x"
        )
    # 4x leverage
    backtest_dr1_4x = backtesting(
            returns_data=daily_returns,
            ou_parameters=ou_parameters_dr1,
            eigenportfolios=eigenportfolios_dr1,
            leverage = 4.0,
            save_prefix= "dr1_4x"
        )

    backtest_dr15_4x = backtesting(
            returns_data=daily_returns,
            ou_parameters=ou_parameters_dr15,
            eigenportfolios=eigenportfolios_dr15,
            leverage = 4.0,
            save_prefix= "dr15_4x"
        )

    backtest_dr55_4x = backtesting(
            returns_data=daily_returns,
            ou_parameters=ou_parameters_dr55,
            eigenportfolios=eigenportfolios_dr55,
            leverage = 4.0,
            save_prefix= "dr55_4x"
        )

    backtest_dr75_4x= backtesting(
            returns_data=daily_returns,
            ou_parameters=ou_parameters_dr75,
            eigenportfolios=eigenportfolios_dr75,
            leverage = 4.0,
            save_prefix= "dr75_4x"
        )
    
    backtest_vw15_4x = backtesting(
            returns_data=volume_weighted_returns,
            ou_parameters=ou_parameters_vw15,
            eigenportfolios=eigenportfolios_vw15,
            leverage = 4.0,
            save_prefix= "vw15_4x"
        )

    backtest_vw55_4x = backtesting(
            returns_data=volume_weighted_returns,
            ou_parameters=ou_parameters_vw55,
            eigenportfolios=eigenportfolios_vw55,
            leverage = 4.0,
            save_prefix= "vw55_4x"
        )

    backtest_vw75_4x = backtesting(
            returns_data=volume_weighted_returns,
            ou_parameters=ou_parameters_vw75,
            eigenportfolios=eigenportfolios_vw75,
            leverage = 4.0,
            save_prefix= "vw75_4x"
        )
    # 5x leverage
    backtest_dr1_5x = backtesting(
            returns_data=daily_returns,
            ou_parameters=ou_parameters_dr1,
            eigenportfolios=eigenportfolios_dr1,
            leverage = 5.0,
            save_prefix= "dr1_5x"
        )

    backtest_dr15_5x = backtesting(
            returns_data=daily_returns,
            ou_parameters=ou_parameters_dr15,
            eigenportfolios=eigenportfolios_dr15,
            leverage = 5.0,
            save_prefix= "dr15_5x"
        )

    backtest_dr55_5x = backtesting(
            returns_data=daily_returns,
            ou_parameters=ou_parameters_dr55,
            eigenportfolios=eigenportfolios_dr55,
            leverage = 5.0,
            save_prefix= "dr55_5x"
        )

    backtest_dr75_5x = backtesting(
            returns_data=daily_returns,
            ou_parameters=ou_parameters_dr75,
            eigenportfolios=eigenportfolios_dr75,
            leverage = 5.0,
            save_prefix= "dr75_5x"
        )
    
    backtest_vw15_5x = backtesting(
            returns_data=volume_weighted_returns,
            ou_parameters=ou_parameters_vw15,
            eigenportfolios=eigenportfolios_vw15,
            leverage = 5.0,
            save_prefix= "vw15_5x"
        )

    backtest_vw55_5x= backtesting(
            returns_data=volume_weighted_returns,
            ou_parameters=ou_parameters_vw55,
            eigenportfolios=eigenportfolios_vw55,
            leverage = 5.0,
            save_prefix= "vw55_5x"
        )

    backtest_vw75_5x = backtesting(
            returns_data=volume_weighted_returns,
            ou_parameters=ou_parameters_vw75,
            eigenportfolios=eigenportfolios_vw75,
            leverage = 5.0,
            save_prefix= "vw75_5x"
        )

if __name__ == "__main__":
    main()