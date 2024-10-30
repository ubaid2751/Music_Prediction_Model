import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
def load_dataset():
    df = pd.read_csv(r"data/music_genre.csv", encoding="utf-8")
    return df

# Viewing unique values
def get_info(df: pd.DataFrame):
    for col in df.columns:
        print(f"{col} has \t{df[col].nunique()} unique values, \t{df[col].isnull().sum()} null values")

# Preprocessing the data
class PreprocessData:
    def __init__(self):
        self.df = load_dataset()
        
def correlation(df: pd.DataFrame):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    heat_map = sns.heatmap(corr, annot=True, cmap="Blues", annot_kws={'fontsize': 8, 'font': "Sans"})
    heat_map.set_title('Correlation Heatmap', fontdict={'fontsize':12})
    plt.show()


df = load_dataset()
correlation(df)
