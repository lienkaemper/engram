import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl


with open("../results/compare_inhib/decomposition_df.pkl", "rb") as f:
     decomp_df = pkl.load(file = f)


fig, axs = plt.subplot_mosaic([["a", "b", "c"]], 
                               figsize = (7.2, 2))

sns.lineplot(data = decomp_df, x = "g", y = "CA1_internal", color = 'blue', style = "h", ax=axs["a"])

sns.lineplot(data = decomp_df, x = "g", y = "CA1_inherited", color = "green", style = "h", ax=axs["b"])
axs["a"].set_xlabel("Inhibitory strength g")
axs["a"].set_ylabel("Covariance")

axs["b"].set_xlabel("Inhibitory strength g")
axs["b"].set_ylabel("Covariance")
axs["b"].sharey(axs["a"])




decomp_df["CA3_total"] = decomp_df["from_CA3E"] + decomp_df["from_CA3N"] + decomp_df["from_CA3I"]
decomp_df["CA3_ext"] = decomp_df["from_CA3E"] + decomp_df["from_CA3N"] 
sns.lineplot(data = decomp_df, x = "g", y = "CA3_total", style="h", color= "black", ax = axs["c"])
axs["c"].set_title("Total")





# fig.supxlabel("Inhibitory strength g")
# fig.suptitle("CA3 engram-engram covariance")
sns.despine(fig = fig)
plt.tight_layout(w_pad = .001)
plt.savefig("../results/plots/CA1_cov_sources.pdf")
plt.show()