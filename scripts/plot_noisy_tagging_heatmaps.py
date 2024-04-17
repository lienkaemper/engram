import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import sys


from src.plotting import raster_plot
from src.noisy_tagging import rates_with_noisy_tagging, cor_with_noisy_tagging

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



fig, axs = plt.subplots(1,2, figsize = (4,2), sharex=True)


CA1E = index_dict["CA1E"]
CA1P = index_dict["CA1P"]
dt_spktrain = 1

CA1_neurons = list(CA1E) + list(CA1P)
N = param_dict["N"]
tstop = 500


rate_df = pd.read_csv("../results/low_inhib/rate_df.csv")
cor_df = pd.read_csv("../results/low_inhib/cor_df.csv")
pred_rate_df = pd.read_csv("../results/low_inhib/pred_rates.csv")
pred_cor_df = pd.read_csv("../results/low_inhib/pred_cors.csv")
cor_df["regions"] = cor_df["region_i"] +"\n"+ cor_df["region_j"]
pred_cor_df["regions"] = pred_cor_df["region_i"] +"\n"+ pred_cor_df["region_j"]

####  applying the noisy tagging to rates from simulation ####


n_noise = 20
noise_levels = np.linspace(0,1, n_noise)
hs = pred_rate_df.h.unique()
n_h = len(hs)
baseline = pred_rate_df[(pred_rate_df['h'] == 1) & (pred_rate_df['region'] == 'CA1E')].iloc[0]["corr_rate"]
baseline_cor = pred_cor_df[(pred_cor_df['h'] == 1) & (pred_cor_df['regions'] == 'CA1E\nCA1E')].iloc[0]["pred_cor"]


rates_matrix = np.zeros((n_noise, n_h))
cors_matrix = np.zeros((n_noise, n_h))

for i, h in enumerate(hs): 
    for j, p_FP in enumerate(noise_levels):
        rate_engram = pred_rate_df[(pred_rate_df['h'] == h) & (pred_rate_df['region'] == 'CA1E')].iloc[0]["corr_rate"]
        rate_non_engram = pred_rate_df[(pred_rate_df['h'] == h) & (pred_rate_df['region'] == 'CA1P')].iloc[0]["corr_rate"]

        cor_ee = pred_cor_df[(pred_cor_df['h'] == h) & (pred_cor_df['regions'] == 'CA1E\nCA1E')].iloc[0]["pred_cor"]
        cor_en = pred_cor_df[(pred_cor_df['h'] == h) & (pred_cor_df['regions'] == 'CA1E\nCA1P')].iloc[0]["pred_cor"]
        cor_nn = pred_cor_df[(pred_cor_df['h'] == h) & (pred_cor_df['regions'] == 'CA1P\nCA1P')].iloc[0]["pred_cor"]

        noisy_rate = rates_with_noisy_tagging(rate_engram, rate_non_engram, p_FP =p_FP)[0]
        noisy_cor = cor_with_noisy_tagging(cor_ee, cor_en, cor_nn, p_FP = p_FP)[0]

        rates_matrix[j, i] = noisy_rate/baseline
        cors_matrix[j, i] = noisy_cor/baseline_cor
    
cs = axs[0].imshow(rates_matrix, origin='lower',  extent=(1, 2, 0,1), cmap = 'magma')
plt.colorbar(cs, ax = axs[0])
axs[0].set_title("Rates")
axs[0].set_xlabel("Engram strength h")
axs[0].set_ylabel("False positive probability")

cs = axs[1].imshow(cors_matrix, origin='lower', extent=(1, 2, 0,1), cmap = 'magma')
axs[1].set_title("Correlations")
axs[1].set_xlabel("Engram strength h")
axs[1].set_ylabel("False positive probability")

plt.colorbar(cs, ax = axs[1])
plt.tight_layout()
plt.savefig("../results/plots/noisy_tagging_heatmaps.pdf")
plt.show()

