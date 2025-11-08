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
import numpy as np
if __name__ == '__main__':
    col = 'x2'  # skewed positive
    plt.figure(figsize=(10,4))
    plt.subplot(1,3,1); plt.hist(df[col], bins=30); plt.title('Original')
    plt.subplot(1,3,2); plt.hist(np.sqrt(df[col]+1), bins=30); plt.title('Sqrt')
    plt.subplot(1,3,3); plt.hist(np.log1p(df[col]), bins=30); plt.title('Log1p')
    plt.tight_layout(); plt.savefig(os.path.join('outputs','05_transformations.png'))
    print('Saved outputs/05_transformations.png')
