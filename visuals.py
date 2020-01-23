
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')


import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time


def evaluate(results):
    """
    Evaluate is used to compare models graphically using matplotlib. To alter the method please update:
        - the second for loop with the metrics you would like to use.
        - Alter the ylabels
        - Alter the titles 
        - Alter the y_limits based on the range you are expecting 
    
    inputs:
        - results from the models you are comparing 
    """
  
    # Create figure
    fig, ax = pl.subplots(2, 3, figsize = (13,10))

    # Constants
    bar_width = 0.3
    colors = ['#A00000','#00A0A0','#00A000']
    
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'mean_absolute_train', 'R2_Score_train', 'pred_time', 
                                    'mean_absolute_test', 'R2_Score_test']):
            for i in np.arange(3):
                
                # Creative plot code
                ax[j//3, j%3].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])
                ax[j//3, j%3].set_xticks([0.45, 1.45, 2.45])
                ax[j//3, j%3].set_xticklabels(["1%", "10%", "100%"])
                ax[j//3, j%3].set_xlabel("Training Set Size")
                ax[j//3, j%3].set_xlim((-0.1, 3.0))
    
    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Mean Absolute Score")
    ax[0, 2].set_ylabel("R2 score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Mean Absolute Score")
    ax[1, 2].set_ylabel("R2 score")
    
    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Mean Absolute Score on Training Subset")
    ax[0, 2].set_title("R2 score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Mean Absolute Score on Testing Set")
    ax[1, 2].set_title("R2 score on Testing Set")
    
    
    # Set y-limits for score panels
    ax[0, 1].set_ylim((1000, 6000))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((1000, 6000))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color = colors[i], label = learner))
    pl.legend(handles = patches, bbox_to_anchor = (-.80, 2.53), \
               loc = 'upper center', borderaxespad = 0., ncol = 3, fontsize = 'x-large')
    
    # Aesthetics
    pl.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 16, y = 1.10)
    pl.tight_layout()
    pl.show()
    
    
def feature_plot(importances, X_train, y_train):
    '''
    inputs:
       importances:
           - The feature weights from model.feature_importances_
       X_train: 
           -features training set
       y_train: 
           -income training set
    '''
    # SHow top 5 features 
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Creat the plot
    fig = pl.figure(figsize = (9,5))
    pl.title("Normalized Weights for First Five Most Predictive Features", fontsize = 16)
    pl.bar(np.arange(5), values, width = 0.6, align="center", color = '#00A000', \
          label = "Feature Weight")
    pl.bar(np.arange(5) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', \
          label = "Cumulative Feature Weight")
    pl.xticks(np.arange(5), columns)
    pl.xlim((-0.5, 4.5))
    pl.ylabel("Weight", fontsize = 12)
    pl.xlabel("Feature", fontsize = 12)
    
    pl.legend(loc = 'upper center')
    pl.tight_layout()
    pl.show()  
    