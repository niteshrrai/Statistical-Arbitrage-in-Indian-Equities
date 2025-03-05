import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from typing import Union
from typing import Optional, Tuple, Dict
from data_preprocessing import save_to_pkl, load_from_pkl

def perform_pca(data: pd.DataFrame,
                window: int = 252,
                n_components: Optional[int] = None,
                variance_threshold: Optional[float] = None,
                save_file_name: Optional[str] = None) -> Dict:
    """
    Perform rolling window PCA on a data matrix.
    Parameters
    ----------
    data : pandas.DataFrame
        Data matrix (rows: observations, columns: variables), e.g., daily returns
    window : int, default=252
        Rolling window size in days
    n_components : int, optional
        Number of principal components to retain. If None, this option is ignored.
    variance_threshold : float, optional
        Desired cumulative explained variance fraction. Overrides n_components if provided.
    save_file_name : str, optional
        Name of the file to save PCA results. If None, results won't be saved.
    Returns
    -------
    dict
        Dictionary containing PCA results for each window:
        - pca_model: fitted PCA model
        - explained_variance: eigenvalues (numpy array)
        - principal_components: eigenvectors (numpy array)
        - explained_variance_ratio: variance explained ratios (numpy array)
        - date: end date of the window
    """

    dates = data.index.values
    data_values = data.values
    results = {}

    for end_idx in range(window, len(data_values)):
        window_data = data_values[end_idx-window:end_idx]
        window_date = dates[end_idx]
        window_date = np.datetime64(window_date, 'D')
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(window_data)

        correlation_matrix = np.corrcoef(normalized_data.T)

        if variance_threshold is not None:
            temp_pca = PCA()
            temp_pca.fit(correlation_matrix)
            cumulative_variance = np.cumsum(temp_pca.explained_variance_ratio_)
            n_components_thresh = int(np.sum(cumulative_variance <= variance_threshold)) + 1
            pca_model = PCA(n_components=n_components_thresh)
        elif n_components is not None:
            pca_model = PCA(n_components=n_components)
        else:
            pca_model = PCA()

        pca_model.fit(correlation_matrix)

        results[window_date] = {
            'pca_model': pca_model,
            'explained_variance': np.array(pca_model.explained_variance_),
            'principal_components': np.array(pca_model.components_),
            'explained_variance_ratio': np.array(pca_model.explained_variance_ratio_),
            'date': window_date
        }

    if save_file_name:
        try:
            save_to_pkl(results, f"{save_file_name}.pkl")
        except Exception as e:
            print(f"Error saving PCA results: {e}")

    return results

def plot_explained_variance(pca_results: Dict, 
                          date: Optional[np.datetime64] = None, 
                          n_components: Optional[int] = None) -> None:
    """
    Plot the explained variance and its distribution from PCA results for a specific date
    or the latest date if none is provided.
    Parameters
    ----------
    pca_results : dict
        Dictionary containing PCA results from perform_pca
    date : np.datetime64, optional
        Specific date to plot results for. If None, uses the latest date.
    n_components : int, optional
        Number of components to plot. If None, plots all available components.
    """
    dates = sorted(pca_results.keys())
    if date is None:
        date = dates[-1]  # Use latest date
    elif date not in pca_results:
        raise ValueError(f"No PCA results found for date {date}")

    explained_variance_ratio = pca_results[date]['explained_variance_ratio']

    if n_components is not None:
        if n_components > len(explained_variance_ratio):
            print(f"Warning: Requested {n_components} components but only {len(explained_variance_ratio)} are available.")
        else:
            explained_variance_ratio = explained_variance_ratio[:n_components]

    plt.figure(figsize=(12, 6))
    plt.margins(x=0)
    plt.grid(False)

    sns.barplot(
        x=np.arange(1, len(explained_variance_ratio) + 1),
        y=explained_variance_ratio,
        color="#000080"
    )
    plt.xlabel("Principal Components")
    plt.ylabel("Proportion of Explained Variance")
    title = "Explained Variance by Principal Components"
    if n_components is not None:
        title += f"\n(Date: {np.datetime_as_string(date, unit='D')})"
    plt.title(title)
    plt.show()

    plt.figure(figsize=(12, 6))    
    plt.grid(False)

    sns.histplot(
        explained_variance_ratio,
        bins=50,
        alpha=0.9,
        color="#8B0000"
    )
    plt.xlabel("Proportion of Explained Variance Ratio")
    plt.ylabel("Percentage of Principal Components")
    title = "Density of Explained Variance Ratio"
    if n_components is not None:
        title += f"\n(Date: {np.datetime_as_string(date, unit='D')})"
    plt.title(title)
    plt.show()

    print(f"\nPCA Analysis for date: {np.datetime_as_string(date, unit='D')}")
    print(f"First PC explained variance: {explained_variance_ratio[0]:.2%}")
    print(f"Top 3 PCs cumulative variance: {np.sum(explained_variance_ratio[:min(3, len(explained_variance_ratio))]):.2%}")
    print(f"Top 5 PCs cumulative variance: {np.sum(explained_variance_ratio[:min(5, len(explained_variance_ratio))]):.2%}")
    print(f"Number of components shown: {len(explained_variance_ratio)}")

    all_variance_ratio = pca_results[date]['explained_variance_ratio']
    thresholds = [0.55, 0.65, 0.75, 0.85, 0.95]
    cumulative_variance = np.cumsum(all_variance_ratio)
    print("\nComponents needed for variance thresholds:")
    for threshold in thresholds:
        n_components_needed = (cumulative_variance <= threshold).sum() + 1
        print(f"{threshold:.0%} variance: {n_components_needed} components")

def analyze_pca_components_over_time(pca_results: Dict,
                                   start_date: Optional[np.datetime64] = None,
                                   end_date: Optional[np.datetime64] = None,
                                   variance_threshold: Optional[float] = None,
                                   n_components: Optional[int] = None) -> None:
    """
    Analyze and plot either:
    1. Number of significant eigenvectors needed to explain variance at y% level over time
    2. Percentage of variance explained by top y eigenvectors over time
    
    Parameters
    ----------
    pca_results : dict
        Dictionary containing PCA results from perform_pca
    start_date : np.datetime64, optional
        Start date for analysis period. If None, uses earliest date.
    end_date : np.datetime64, optional
        End date for analysis period. If None, uses latest date.
    variance_threshold : float, optional
        Target explained variance ratio threshold (e.g., 0.85 for 85%)
    n_components : int, optional
        Number of top eigenvectors to analyze
    """
    if variance_threshold is None and n_components is None:
        raise ValueError("Must provide either variance_threshold or n_components")
    
    if isinstance(start_date, str):
        start_date = np.datetime64(start_date, 'D')
    if isinstance(end_date, str):
        end_date = np.datetime64(end_date, 'D')

    all_dates = np.array(sorted(pca_results.keys()))

    if start_date is not None:
        date_mask = all_dates >= start_date
        all_dates = all_dates[date_mask]
    if end_date is not None:
        date_mask = all_dates <= end_date
        all_dates = all_dates[date_mask]
    
    if len(all_dates) == 0:
        raise ValueError("No dates found in the specified range")
    
    def format_xaxis(ax):
        dates_mdates = mdates.date2num([pd.Timestamp(d) for d in all_dates])
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)

    if variance_threshold is not None:
        n_eigenvectors_needed = np.zeros(len(all_dates))
        
        for i, date in enumerate(all_dates):
            eigenvalues = pca_results[date]['explained_variance_ratio']
            cumulative_variance = np.cumsum(eigenvalues)
            n_eigenvectors_needed[i] = (cumulative_variance <= variance_threshold).sum() + 1

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.margins(x=0.005)
        ax.grid(axis='y')
        ax.plot(all_dates, n_eigenvectors_needed, color='#000080', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Number of Eigenvectors')
        ax.set_title(f'Significant Eigenvectors Needed to Explain {variance_threshold:.0%} of Variance')
        ax.yaxis.grid(True)
        format_xaxis(ax)
        mean_eigenvectors = np.mean(n_eigenvectors_needed)
        plt.tight_layout()
        plt.show()

        print(f"\nSignificant eigenvectors needed for {variance_threshold:.0%} variance:")
        print(f"Mean: {mean_eigenvectors:.2f}")
        print(f"Min: {np.min(n_eigenvectors_needed):.0f}")
        print(f"Max: {np.max(n_eigenvectors_needed):.0f}")
        print(f"Std: {np.std(n_eigenvectors_needed):.2f}")
    
    if n_components is not None:
        variance_explained = np.zeros(len(all_dates))

        for i, date in enumerate(all_dates):
            eigenvalues = pca_results[date]['explained_variance_ratio']
            if len(eigenvalues) < n_components:
                raise ValueError(f"Only {len(eigenvalues)} components available, but {n_components} requested")
            variance_explained[i] = np.sum(eigenvalues[:n_components])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.margins(x=0.005)
        ax.grid(axis='y')
        
        ax.plot(all_dates, variance_explained, color='green', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Proportion of Explained Variance')
        ax.set_title(f'Variance Explained by Top {n_components} Eigenvectors')
        ax.yaxis.grid(True) 
        format_xaxis(ax)
        ax.legend()
        plt.tight_layout()
        plt.show()
        
        mean_variance = np.mean(variance_explained)
        print(f"\nProportion of Variance explained by top {n_components} eigenvectors:")
        print(f"Mean: {mean_variance:.2%}")
        print(f"Min: {np.min(variance_explained):.2%}")
        print(f"Max: {np.max(variance_explained):.2%}")
        print(f"Std: {np.std(variance_explained):.2%}")

def main():
    daily_returns = load_from_pkl('daily_returns.pkl')
    volume_weighted_returns = load_from_pkl('volume_weighted_returns.pkl')

    pca_dr = perform_pca(
        daily_returns,
        window=252,
        save_file_name ='pca_dr')
    
    pca_vw = perform_pca(
        volume_weighted_returns,
        window=252,
        save_file_name ='pca_vw')
    
    # Fixed component analysis

    pca_dr1 = perform_pca(
        daily_returns,
        window=252,
        n_components=1,
        save_file_name='pca_dr1'
    )

    pca_dr15 = perform_pca(
        daily_returns,
        window=252,
        n_components=15,
        save_file_name='pca_dr15'
    )
    
    
    # Variance-based analysis
    
    pca_dr55 = perform_pca(
        daily_returns,
        window=252,
        variance_threshold=0.55,
        save_file_name='pca_dr55'
    )
    
    pca_dr75 = perform_pca(
        daily_returns,
        window=252,
        variance_threshold=0.75,
        save_file_name='pca_dr75'
    )
    
    # Volume-weighted returns analysis
    pca_vw15 = perform_pca(
        volume_weighted_returns,
        window=252,
        n_components=15,
        save_file_name='pca_vw15'
    )
    
    pca_vw55 = perform_pca(
        volume_weighted_returns,
        window=252,
        variance_threshold=0.55,
        save_file_name='pca_vw55'
    )
    
    pca_vw75 = perform_pca(
        volume_weighted_returns,
        window=252,
        variance_threshold=0.75,
        save_file_name='pca_vw75'
    )
    
    print("PCA Decomposition and analysis completed successfully.")

if __name__ == "__main__":
    main()
   