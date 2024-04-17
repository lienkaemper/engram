import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl


plt.style.use('paper_style.mplstyle')
size = 20



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





fig, ax = plt.subplots(1, figsize = (2,2))
sns.lineplot(data = theory_df, x = "g", y = "pred_cor_engram_vs_engram_ratio", ax = ax, color = "#F37343", label = "Engram vs. engram")
sns.scatterplot(data = df, x = "g", y = "sim_cor_engram_vs_engram_ratio", ax = ax, s = size, color = "#F37343")
sns.lineplot(data = theory_df, x = "g", y = "pred_cor_engram_vs_non_engram_ratio", ax = ax, label = "Engram vs. non-engram", color = "#FEC20E")
sns.scatterplot(data = df, x = "g", y = "sim_cor_engram_vs_non_engram_ratio", ax = ax, s = size, color = "#FEC20E")
sns.lineplot(data = theory_df, x = "g", y = "pred_cor_non_engram_vs_non_engram_ratio", ax =ax, label = "Non-engram vs. non-engram", color = "#06ABC8")
sns.scatterplot(data = df, x = "g", y = "sim_cor_non_engram_vs_non_engram_ratio", ax =ax, s = size, color = "#06ABC8")
ax.set_xlim([1,3])
ax.set_title("Correlation ratio")
ax.set_xlabel("Inhibition strength: g")
ax.set_ylabel("Correlation ratio")
ax.get_legend().remove()

sns.despine(fig = fig)
plt.tight_layout(w_pad = .001)
plt.savefig("../results/plots/correlation_ratio.pdf")
plt.show()