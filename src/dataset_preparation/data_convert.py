
from src.dataset.base import TimeSeriesData
import pandas as pd
import numpy as np

def MinMaxScaler(data):
  """Min Max normalizer.
  
  Args:
    - data: original data shape (time steps, column_variates)
  
  Returns:
    - norm_data: normalized data
  """
  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)
  norm_data = numerator / (denominator + 1e-7)
  return norm_data

def sample_to_array(sample_dict: dict[str, pd.Series], variates: list[str]) -> np.ndarray:
    """
    Convert a single sample dictionary into a NumPy array.
    
    Only uses variates that exist in the sample_dict.
    """
    # Filter out missing variates
    available_variates = [var for var in variates if var in sample_dict and not sample_dict[var].empty]

    # Stack into NumPy array (time_length, num_variates)
    array = np.stack([sample_dict[var][var].to_numpy() for var in available_variates], axis=1)

    return array

def dataset_to_numpy(ts_dataset: TimeSeriesData, seq_len: int) -> np.ndarray:
    """
    Convert a TimeSeriesData object into a shuffled NumPy array of sequences.
    
    Args:
        ts_dataset: An instance of TimeSeriesData (or subclass)
        seq_len: Length of each time-series sequence
    
    Returns:
        data: NumPy array of shape (num_sequences, seq_len, num_variates)
    """
    all_sequences = []
    num_samples = ts_dataset.how_many_unique_samples()
    variates = list(ts_dataset.what_variates())
    num_variates = len(variates)

    # Extract sequences for each sample
    for i in range(num_samples):
        sample_dict = ts_dataset[i]  # dict[str, pd.Series]
        
        # Convert dict of pandas Series -> NumPy array
        sample_array = sample_to_array(sample_dict, variates)
        sample_array = MinMaxScaler(sample_array)
        print(f'array for {i} is {sample_array.shape}')
        # Slide a window of seq_len
        T = sample_array.shape[0]
        C = sample_array.shape[1]
        for t in range(T - seq_len + 1):
            for c in range(C):
                seq = sample_array[t:t+seq_len, c:c+1]
                all_sequences.append(seq)
        
    # Convert to NumPy array and shuffle
    data = np.array(all_sequences)
    idx = np.random.permutation(len(data))
    data = data[idx]
    
    #num_sequences, seq_len, num_variates
    return data