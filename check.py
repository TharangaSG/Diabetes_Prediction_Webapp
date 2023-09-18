import numpy as np
from sklearn.preprocessing import StandardScaler

# function standard_normalization
def standard_normalization(data):
  mean = np.mean(data)
  std = np.std(data)
  scaled_data = (data - mean) / std
  return scaled_data

scaled_signal_standard = standard_normalization(signal)