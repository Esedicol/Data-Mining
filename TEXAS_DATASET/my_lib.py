# Collection of functions useful for analysing TeXas InPateint Dataset 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats

import seaborn as sns
sns.set_style("darkgrid")

from IPython.display import display, Markdown
pd.set_option('display.max_columns', None)  

import glob, os

DEBUG = False
SEED = 42

# Function to calculate K-Fold Cross Validation score averages
def cross_val_avg(x,y):
    kernels = ['rbf', 'linear']
    C = [1,10,20]
    avg_scores = {}

    for kval in kernels:
        for cval in C:
            cv_scores = cross_val_score(svm.SVC(kernel=kval,C=cval,gamma='auto'),x, y, cv=5)
            avg_scores[kval + '_' + str(cval)] = np.average(cv_scores)

    avg_scores

# Find outliers function
def find_outliers(): 
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    
    interQuartileRange = q3 - q1
    floor = q1 - 1.5 * interQuartileRange
    ceiling = q3 + 1.5 * interQuartileRange
    outlier_ind = list(x.index[(x < floor) | (x > ceiling)])
    outlier_val = list(x[outlier_ind])
    
    return outlier_ind, outlier_val
    
# Fucntion for ploting a histogram. Takes in a dataframe column as an input
def plot_hist(x):
    plt.hist(x, color="blue", alpha=0.5)
    plt.title(f'Histogram of {x.name}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
    
# Fucntion for ploting a histogram x breaked down by our outcome variable y
def plot_hist_x_y(x,y):
    plt.hist(list(x[y=='short']), alpha=0.5, label="Short")
    plt.hist(list(x[y=='medium']), alpha=0.5, label="Medium")
    plt.hist(list(x[y=='long']), alpha=0.5, label="Long")
    plt.title(f'Histogram of {x.name}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()
    
def eda_categorical(df, feature, target, max_categories=20, labels=None, header=True, brief=False):
    
    print("\n")
    if header: display(Markdown("#### %s" % feature))
    
    if df[feature].nunique()>max_categories:
        print("Warning: number of columns (%s) in feature (%s) is too large (>%s)") % (df[feature].nunique(), feature, max_categories)
        return
        
    # 1. Distribution table
    display(Markdown("**Distribution**"))
    
    if labels: 
        display(df[feature].map(labels).value_counts(dropna=False))
    else: 
        display(df[feature].value_counts(dropna=False))

    # 2. Count plot
    print("\n")
    display(Markdown("**Count Plots**"))
    
    fig, ax = plt.subplots(figsize=(9,4), nrows=1, ncols=2)
    
    # left plot - freq of each category
    df_countplot = df.groupby(feature).size().sort_values()
    df_countplot.plot(kind='barh', ax=ax[0])

    # right plot - target breakdown within each category
    df_ft_countplot = pd.crosstab(df[feature], df[target], normalize='index')
    df_ft_countplot["total"] = df_countplot
    df_ft_countplot.sort_values("total", inplace=True)
    df_ft_countplot.drop(columns="total", inplace=True)
    df_ft_countplot.plot(kind='barh', stacked=True, ax=ax[1])

    ax[0].set_title=("Count plot of %s" % feature)
    ax[1].set_title=("Breakdown of %s" % target)
    fig.suptitle("Impact of feature '%s' on target '%s'" % (feature, target), fontsize="large")
    plt.show()
    
    # 3. Goodness of fit
    display(Markdown("**Chi-Sq Goodness of Fit**"))
    df_ft_countplot = pd.crosstab(df[feature], df[target])
    result = stats.chi2_contingency(df_ft_countplot)
    print('Chi-Square statistic %.4e (p=%.4e, dof=%d)' % result[0:3])

