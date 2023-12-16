import matplotlib.pyplot as plt
import seaborn as sns
import math

def plot_heatmap(df):
    f, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(df.corr(),annot=True,linewidth=.5,fmt='.1f',ax=ax)
    plt.show()

def plot_distplots(df):
    n_rows = math.floor(math.sqrt(df.shape[1]))
    n_cols = math.ceil(math.sqrt(df.shape[1]))
    # Create the subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
    for i, column in enumerate(df.columns):
        sns.distplot(df[column], ax=axes[i // n_cols, i % n_cols])

def plot_boxplots(df):
    n_rows = math.floor(math.sqrt(df.shape[1]))
    n_cols = math.ceil(math.sqrt(df.shape[1]))
    # Create the subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
    for i, column in enumerate(df.columns):
        sns.boxplot(df[column], ax=axes[i // n_cols, i % n_cols])

def plot_barplots(df):
    n_rows = math.floor(math.sqrt(df.shape[1]))
    n_cols = math.ceil(math.sqrt(df.shape[1]))
    # Create the subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
    for i, column in enumerate(df.columns):
        sns.barplot(df[column], ax=axes[i // n_cols, i % n_cols])