import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import sys


from src.plotting import raster_plot

plt.style.use('paper_style.mplstyle')
engram_color =   "#F37343"
non_engram_color =  "#06ABC8"
cross_color = "#FEC20E"

if len(sys.argv) < 2:
    f = open("../results/most_recent.txt", "r")
    dirname = f.read()
else:
    dirname = sys.argv[1]

with open(dirname+"/index_dict.pkl", "rb") as file:
    index_dict = pkl.load(file)

with open(dirname + "/param_dict.pkl", "rb") as file:
    param_dict = pkl.load(file)

################################

yticks = [r[0] for r in index_dict.values()]
neurons = range(270)
tstop = 250
with open("../results/low_inhib/spikes_h={}.pkl".format(1.0), "rb") as file:
    spktimes = pkl.load(file)

fig, ax = plt.subplots(figsize = (2.8, 2.1))
raster_plot(spktimes, neurons, 0, tstop, ax = ax, yticks = yticks )
sns.despine()
plt.savefig("../results/plots/raster_h=1_g=1.pdf")
plt.show()

with open("../results/low_inhib/spikes_h={}.pkl".format(2.0), "rb") as file:
    spktimes = pkl.load(file)

fig, ax = plt.subplots(figsize = (2.8, 2.1))
raster_plot(spktimes, neurons, 0, tstop, ax = ax, yticks = yticks )
sns.despine()
plt.savefig("../results/plots/raster_h=2_g=1.pdf")
plt.show()


fig, axs = plt.subplots(1, 2, figsize = (7,2))


CA1E = index_dict["CA1E"]
CA1P = index_dict["CA1P"]
dt_spktrain = 1

CA1_neurons = list(CA1E) + list(CA1P)
N = param_dict["N"]



rate_df = pd.read_csv("../results/low_inhib/rate_df.csv")
cor_df = pd.read_csv("../results/low_inhib/cor_df.csv")
cor_df["regions"] = cor_df["region_i"] +"\n"+ cor_df["region_j"]


pred_rate_df = pd.read_csv("../results/low_inhib/pred_rates.csv")
pred_rate_df = pred_rate_df[pred_rate_df["region"].isin(["CA1E", "CA1P"])]
baseline_rate = np.mean(pred_rate_df[pred_rate_df["h"] == 1]["corr_rate"])

norm_pred_rate_df = pred_rate_df.copy()
norm_pred_rate_df["corr_rate"] = norm_pred_rate_df["corr_rate"]/baseline_rate
norm_pred_rate_df["tree_rate"] = norm_pred_rate_df["tree_rate"]/baseline_rate


norm_rate_df = rate_df.copy()
norm_rate_df["rate"] = rate_df["rate"]/baseline_rate

pred_cor_df = pd.read_csv("../results/low_inhib/pred_cors.csv")
pred_cor_df = pred_cor_df[pred_cor_df["region_i"].isin(["CA1E", "CA1P"])]
pred_cor_df = pred_cor_df[pred_cor_df["region_j"].isin(["CA1E", "CA1P"])]
pred_cor_df["regions"] = pred_cor_df["region_i"] +"\n"+ pred_cor_df["region_j"]
sns.lineplot(data = norm_pred_rate_df, x = "h", hue = "region", y = "corr_rate",  ax = axs[0], errorbar=None, palette= [engram_color, non_engram_color])
sns.lineplot(data = norm_pred_rate_df, x = "h", hue = "region", y = "tree_rate",  ax = axs[0], errorbar=None, linestyle='--', palette= [engram_color, non_engram_color])

sns.scatterplot(data= norm_rate_df, x = "h", hue = "region", y = "rate", ax = axs[0], palette= [engram_color, non_engram_color])
axs[0].get_legend().remove()
axs[0].set_ylabel("normalized rate")

sns.lineplot(data = pred_cor_df, x = "h", hue = "regions", y = "pred_cor",  ax = axs[1],errorbar=None, palette= [engram_color,  cross_color,non_engram_color])
sns.scatterplot(data= cor_df, x = "h", hue = "regions", y = "correlation", ax = axs[1], palette= [engram_color,cross_color,  non_engram_color])
axs[1].get_legend().remove()
axs[1].set_ylabel("correlation")







sns.despine(fig = fig)
plt.tight_layout(w_pad = .001)
plt.savefig("../results/plots/low_inhib.pdf")
plt.show()

