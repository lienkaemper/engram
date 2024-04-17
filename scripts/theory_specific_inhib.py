import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os


from src.theory import  y_0_quad, find_iso_rate_input, find_iso_rate_ca3
from src.generate_connectivity import macro_weights






b_small = np.array([.4, .4, .5, .4, .4, .5])   #without excitability
J0 = .2
g_ii = 1
g_min = 1
g_max = 3
n_g = 3
h_min = 1
h_max = 2
n_h = 10
gs = np.linspace(g_min, g_max, n_g)
hs = np.linspace(h_min, h_max, n_h)

J_baseline =macro_weights(J=J0, h3 = 1, h1 =1, g =1)
y_baseline = y_0_quad(J_baseline, b_small)


h_list = []
g_list = []
h_i3_list = []
h_i1_list = []
for g in gs:
    J_small =macro_weights(J=J0, h3 = 1, h1 =1, g =g)
    b_iso =find_iso_rate_input(target_rate_3 = y_baseline[0], target_rate_1= y_baseline[3], J = J_small, b = b_small, b0_min = 0, b0_max = 1, n_points=1000, plot=False)
    for h in hs:
        h_i1, h_i3  = find_iso_rate_ca3(y_baseline[3],y_baseline[0], h=h, J0 = J0, g=g, g_ii=g_ii, b = b_iso, h_i_min = 1, h_i_max = 2, 
        type = "quadratic", n_points = 200)
        h_list.append(h)
        g_list.append(g)
        h_i3_list.append(h_i3)
        h_i1_list.append(h_i1)

plast_df = pd.DataFrame({"h" : h_list, "g":g_list, "h_i3": h_i3_list, "h_i1": h_i1_list})

with open("../results/compare_inhib/plast_df.pkl", "wb") as f:
    pkl.dump(obj = plast_df, file = f)


