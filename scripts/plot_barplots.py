import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os
import itertools



plt.style.use('paper_style.mplstyle')




with open("../results/compare_inhib/df.pkl", "rb") as f:
    df = pkl.load(f)

with open("../results/compare_inhib/raw_df.pkl", "rb") as f:
    raw_df = pkl.load(f)

with open("../results/compare_inhib/theory_df.pkl", "rb") as f:
    theory_df = pkl.load(f)

with open("../results/compare_inhib/decomposition_df.pkl", "rb") as f:
     decomp_df = pkl.load(file = f)

with open("../results/compare_inhib/index.pkl", "rb") as f:
    index_dict = pkl.load(file = f)


fig, axs = plt.subplot_mosaic([["a", "b"], 
                               ["c", "d"]], figsize = (4, 4))

baseline_df = raw_df.loc[raw_df["g"] ==1]
baseline_rate = np.mean(baseline_df.sim_rate_non_engram)
low_inhib_rate_df  =baseline_df.loc[:, ["h","sim_rate_engram", "sim_rate_non_engram"]]
low_inhib_cor_df  = baseline_df.loc[:, ["h", "sim_cor_engram_vs_engram", "sim_cor_engram_vs_non_engram","sim_cor_non_engram_vs_non_engram" ]]
low_inhib_rate_df =low_inhib_rate_df.melt(id_vars=['h'], var_name='region', value_name='rate')
low_inhib_rate_df["norm_rate"] = low_inhib_rate_df.rate/baseline_rate
low_inhib_cor_df = low_inhib_cor_df.melt(id_vars=['h'], var_name='region', value_name='correlation')


high_inhib_df = raw_df.loc[raw_df["g"] ==3]

high_inhib_rate_df  = high_inhib_df.loc[:, ["h","sim_rate_engram", "sim_rate_non_engram"]]
high_inhib_cor_df  = high_inhib_df.loc[:, ["h", "sim_cor_engram_vs_engram", "sim_cor_engram_vs_non_engram","sim_cor_non_engram_vs_non_engram" ]]
high_inhib_rate_df = high_inhib_rate_df.melt(id_vars=['h'], var_name='region', value_name='rate')
high_inhib_rate_df["norm_rate"] = high_inhib_rate_df.rate/baseline_rate
high_inhib_cor_df = high_inhib_cor_df.melt(id_vars=['h'], var_name='region', value_name='correlation')

sns.barplot(data =low_inhib_rate_df, x = "region", hue = "h", y = "norm_rate", ax = axs["a"])
axs["a"].get_legend().remove()
axs["a"].set(xticklabels=[])

sns.barplot(data =low_inhib_cor_df, x = "region", hue = "h", y = "correlation", ax = axs["b"])
axs["b"].get_legend().remove()
axs["b"].set(xticklabels=[])

sns.barplot(data =high_inhib_rate_df, x = "region", hue = "h", y = "norm_rate", ax = axs["c"])
axs["c"].get_legend().remove()
axs["c"].set(xticklabels=[])

sns.barplot(data =high_inhib_cor_df, x = "region", hue = "h", y = "correlation", ax = axs["d"])
axs["d"].get_legend().remove()
axs["d"].set(xticklabels=[])

sns.despine()
plt.tight_layout()
plt.savefig("../results/plots/barplots.pdf")
plt.show()
