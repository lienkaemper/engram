import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl



plt.style.use('paper_style.mplstyle')
engram_color =   "#F37343"
non_engram_color =  "#06ABC8"
cross_color = "#FEC20E"

with open("../results/compare_inhib/plast_df.pkl", "rb") as f:
    plast_df = pkl.load( file = f)


    
plast_df= plast_df[plast_df["g"] == 1]


fig, ax = plt.subplots(figsize = (2,2))
sns.lineplot(data = plast_df, x = "h", y = "h_i3", ax = ax, label = "CA3", color = 'gray')
sns.lineplot(data = plast_df, x = "h", y = "h_i1", ax = ax, label = "CA1", color = 'black')
sns.despine()
plt.savefig("../results/plots/specific_inhibition.pdf")
plt.show()