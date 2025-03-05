import os
import pickle
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def download_stock_data(ticker_list, start_date, end_date):
    """
    Download historical stock data for given tickers.
    Drops stocks with incomplete data across the entire period.
    """
    try:
        print("Downloading stock data from Yahoo Finance...")
        stock_data = yf.download(ticker_list, start=start_date, end=end_date)
        
        adjusted_close_data = stock_data['Adj Close']
        volume_data = stock_data['Volume']
        
        adjusted_close_clean = adjusted_close_data.dropna(axis=1)
        volume_data_clean = volume_data[adjusted_close_clean.columns]
        
        dropped_stocks = set(adjusted_close_data.columns) - set(adjusted_close_clean.columns)
        if dropped_stocks:
            print(f"Dropped stocks due to incomplete data: {dropped_stocks}")
        
        print(f"Original stocks: {len(adjusted_close_data.columns)}")
        print(f"Stocks after cleaning: {len(adjusted_close_clean.columns)}")
        
        print("Download and cleaning complete.")
        return adjusted_close_clean, volume_data_clean
    
    except Exception as error:
        print(f"Error downloading stock data: {error}")
        return None, None
    

def calculate_daily_returns(price_data):
    """
    Calculate daily percentage returns for stocks.
    Ensures only stocks with consistent data are included.
    """
    daily_return_data = price_data.pct_change().dropna()
    return daily_return_data


def calculate_volume_weighted_returns(volume_data, daily_returns, lookback_period=10):
    """
    Compute volume-weighted returns using existing daily returns and trading volume.
    """
    clean_volume_data = volume_data.dropna(axis=1)
    clean_returns = daily_returns.dropna(axis=1)
    
    common_stocks = clean_volume_data.columns.intersection(clean_returns.columns)
    clean_volume_data = clean_volume_data[common_stocks]
    clean_returns = clean_returns[common_stocks]
    
    typical_daily_volume = clean_volume_data.rolling(window=lookback_period).mean().iloc[lookback_period:]
    
    cumulative_volume = np.abs(clean_volume_data.cumsum())
    cumulative_volume_diff = cumulative_volume.diff().iloc[lookback_period:]
    cumulative_volume_diff = cumulative_volume_diff.replace(0, np.nan)
    
    aligned_returns = clean_returns.iloc[lookback_period - 1:]
    
    volume_weighted_returns = (
        aligned_returns * typical_daily_volume / cumulative_volume_diff
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    
    volume_weighted_returns.index = aligned_returns.index
    
    return volume_weighted_returns


def save_to_pkl(result, file_name, folder_name="data"):
    current_dir = os.getcwd()
    folder_path = os.path.join(current_dir, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "wb") as file:
        pickle.dump(result, file)
    
def load_from_pkl(file_name, folder_name="data"):
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, folder_name, file_name)
    
    with open(file_path, "rb") as file:
        result = pickle.load(file)
    
    return result


def main():
    #list of nifty 100 tickers
    ticker_list = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS',
        'ITC.NS', 'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ASIANPAINT.NS', 'BAJFINANCE.NS',
        'LT.NS', 'AXISBANK.NS', 'HCLTECH.NS', 'MARUTI.NS', 'ULTRACEMCO.NS', 'ONGC.NS', 'M&M.NS',
        'TITAN.NS', 'SUNPHARMA.NS', 'WIPRO.NS', 'NESTLEIND.NS', 'DRREDDY.NS', 'BAJAJFINSV.NS',
        'POWERGRID.NS', 'NTPC.NS', 'ADANIPORTS.NS', 'TATASTEEL.NS', 'GRASIM.NS', 'INDUSINDBK.NS',
        'JSWSTEEL.NS', 'DIVISLAB.NS', 'SBILIFE.NS', 'HDFCLIFE.NS', 'BPCL.NS', 'TATAMOTORS.NS',
        'HEROMOTOCO.NS', 'COALINDIA.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'EICHERMOT.NS', 'TECHM.NS',
        'SHREECEM.NS', 'UPL.NS', 'APOLLOHOSP.NS', 'BAJAJ-AUTO.NS', 'ADANIENT.NS', 'TATACONSUM.NS',
        'DABUR.NS', 'HINDALCO.NS', 'PIDILITIND.NS', 'GODREJCP.NS', 'DMART.NS', 'BIOCON.NS',
        'YESBANK.NS', 'GAIL.NS', 'INDIGO.NS', 'MUTHOOTFIN.NS', 'AMBUJACEM.NS', 'TORNTPHARM.NS',
        'SIEMENS.NS', 'SRF.NS', 'PAGEIND.NS', 'PEL.NS', 'MRF.NS', 'BEL.NS',
        'ICICIPRULI.NS', 'PETRONET.NS', 'BOSCHLTD.NS', 'BANDHANBNK.NS', 'IDFCFIRSTB.NS', 'CONCOR.NS',
        'INDHOTEL.NS', 'VOLTAS.NS', 'LUPIN.NS', 'ICICIGI.NS', 'COLPAL.NS',
        'HAVELLS.NS', 'NAUKRI.NS', 'SBICARD.NS', 'BERGEPAINT.NS', 'PIIND.NS', 'HINDPETRO.NS',
        'BANKBARODA.NS', 'ABB.NS', 'M&MFIN.NS', 'MFSL.NS', 'HDFCAMC.NS', 'CROMPTON.NS',
        'INDUSTOWER.NS', 'GODREJPROP.NS', 'MANAPPURAM.NS', 'DLF.NS', 'ZOMATO.NS',
        'AUROPHARMA.NS', 'CANBK.NS', 'GLAND.NS', 'MAXHEALTH.NS', 'ADANIGREEN.NS'
    ]
    
    start_date = "2011-12-30"
    end_date = "2025-01-01"

    adjusted_close_clean, volume_clean_data = download_stock_data(ticker_list, start_date, end_date)
    nifty_price_data, nifty_volume_data = download_stock_data('^CNX100', start_date, end_date)

    save_to_pkl(adjusted_close_clean, "adjusted_close_clean.pkl")
    save_to_pkl(volume_clean_data, "volume_clean_data.pkl")
    save_to_pkl(nifty_price_data, "nifty_price_data.pkl")
    save_to_pkl(nifty_volume_data, "nifty_volume_data.pkl")
    
    if adjusted_close_clean is not None and volume_clean_data is not None:
        daily_returns = calculate_daily_returns(adjusted_close_clean)
        save_to_pkl(daily_returns, "daily_returns.pkl")
        
        tickers = daily_returns.columns.tolist()
        save_to_pkl(tickers, 'tickers.pkl')

        volume_weighted_returns = calculate_volume_weighted_returns(volume_clean_data, daily_returns, lookback_period=10)
        save_to_pkl(volume_weighted_returns, "volume_weighted_returns.pkl")
    
    if nifty_price_data is not None:
        nifty_returns = calculate_daily_returns(nifty_price_data)
        save_to_pkl(nifty_returns,"nifty_returns.pkl")

if __name__ == "__main__":
    main()
