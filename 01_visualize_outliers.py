"""Self-contained example script for data preprocessing / feature engineering.
Each script generates a synthetic dataset and performs the requested task.
Requirements: numpy, pandas, matplotlib, scipy, scikit-learn
Run: python scriptname.py
Outputs: Some scripts save plots as PNG in the script folder's 'outputs' subfolder.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
np.random.seed(0)
os.makedirs('outputs', exist_ok=True)

def make_sample_df(n=500):
    # numeric features with some skew/outliers, a categorical and a date column
    x1 = np.random.normal(loc=50, scale=10, size=n)               # roughly normal
    x2 = np.random.exponential(scale=2.0, size=n) * 10            # skewed
    x3 = np.random.normal(loc=100, scale=30, size=n)              # wide spread with outliers
    # inject some extreme outliers
    outliers_idx = np.random.choice(np.arange(n), size=max(1,n//50), replace=False)
    x3[outliers_idx] *= 8
    cat = np.random.choice(['A','B','C'], size=n)
    start = datetime(2020,1,1)
    dates = [start + timedelta(days=int(d)) for d in np.random.randint(0,1000,size=n)]
    df = pd.DataFrame({'x1':x1, 'x2':x2, 'x3':x3, 'category':cat, 'date':dates})
    return df

df = make_sample_df(500)
def remove_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5*iqr
    upper = q3 + 1.5*iqr
    return series[(series >= lower) & (series <= upper)]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    col = 'x3'
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.hist(df[col], bins=30)
    plt.title('Before outlier removal: '+col)
    cleaned = remove_outliers_iqr(df[col])
    plt.subplot(1,2,2)
    plt.hist(cleaned, bins=30)
    plt.title('After IQR outlier removal: '+col)
    plt.tight_layout()
    plt.savefig(os.path.join('outputs','01_hist_before_after.png'))
    print('Saved outputs/01_hist_before_after.png')
