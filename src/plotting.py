import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



def raster_plot(spktimes, neurons, t_start, t_stop, yticks = None, ax = None, s_scale = 1):
    '''
    plot raster plot

    spktimes -- Numpy array Nspikes x 2, first column is times and second is neurons
    neurons -- iterable of neuron indices 
    t_start, t_stop -- time window for plotting (does not need to match simulation)
    yticks -- neuron indices for y ticks (optional)
    ax -- matplotlin axes to place plot (optional)
    s_scale -- marker size scale
    '''
    df = pd.DataFrame(spktimes, columns = ["time", "neuron"])
    df = df[(df["time"] < t_stop) &( df["time"] > t_start) ]
    df = df[df["neuron"].isin(neurons)]
    if ax is None:
        fig, ax = plt.subplots()
    s = s_scale *1000
    sns.scatterplot(data = df, x = "time", y = "neuron", marker = "|" , s = s/(.5*len(neurons)), ax = ax, hue = "neuron",  palette = ["black"])
    plt.legend([],[], frameon=False)
    ax.get_legend().remove()
    if yticks is not None:
        ax.set_yticks(yticks)    
    if ax is None:
        return fig, ax

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')