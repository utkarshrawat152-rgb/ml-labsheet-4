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
from sklearn.preprocessing import OrdinalEncoder
if __name__ == '__main__':
    ord_map = ['low','medium','high']
    df['ord'] = np.random.choice(ord_map, size=len(df))
    enc = OrdinalEncoder(categories=[ord_map])
    df['ord_encoded'] = enc.fit_transform(df[['ord']]).astype(int)
    print(df[['ord','ord_encoded']].head())
    df[['ord','ord_encoded']].to_csv('outputs/17_label_encoded.csv', index=False)
    print('Saved outputs/17_label_encoded.csv')
