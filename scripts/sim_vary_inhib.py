import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc

from src.simulation import sim_glm_pop
from src.theory import   y_0_quad,  find_iso_rate_input, cor_pred, find_iso_rate_ca3, loop_by_integration
from src.correlation_functions import rate, two_pop_correlation, mean_pop_correlation, sum_by_region, mean_by_region
from src.generate_connectivity import  gen_adjacency, hippo_weights, macro_weights



# generate adjacency matrix 
N_E =60
N_I = 15
cells_per_region =np.array([N_E, N_E, N_I,  N_E, N_E, N_I])
N = np.sum(cells_per_region)
pEE = .2
pIE = .8
pII = .8
pEI = .8

macro_connectivity = np.array([
             [pEE, pEE, pEI, pEE, pEE, pEI],
             [pEE, pEE, pEI, pEE, pEE, pEI],
             [pIE, pIE, pII, pIE, pIE, pII],
             [pEE, pEE, pEI, pEE, pEE, pEI],
             [pEE, pEE, pEI, pEE, pEE, pEI],
             [pIE, pIE, pII, pIE, pIE, pII]])
             
A, index_dict = gen_adjacency(cells_per_region, macro_connectivity)

with open("../results/compare_inhib/index.pkl", "wb") as f:
    pkl.dump(obj = index_dict, file = f)


#simulation parameters 
dt = 0.02
tstop = 50000


b_small = np.array([.4, .4, .5, .4, .4, .5])  #without excitability
J0 = .2
g_ii = 1
g_min = 1
g_max = 3
n_g = 5
h_h = 2
gs = np.linspace(g_min, g_max, n_g)


ys_pred_engram = []
ys_sim_engram = []
ys_pred_non_engram = []
ys_sim_non_engram = []
h_list = []
g_list = []
cors_ee = []
cors_en = []
cors_nn = []
sim_cors_ee = []
sim_cors_en = []
sim_cors_nn = []

for trial in range(8):
    A, index_dict = gen_adjacency(cells_per_region, macro_connectivity)
    J_baseline = macro_weights(J=J0, h3 = 1, h1 =1, g =1)
    y_baseline = y_0_quad(J_baseline, b_small)
    for g in gs:
        J_small =macro_weights(J=J0, h3 = 1, h1 =1, g =g)
        b_iso =find_iso_rate_input(target_rate_1= y_baseline[3], target_rate_3=y_baseline[0], J = J_small, b = b_small, b0_min = 0, b0_max = 1, n_points=1000, plot = False)
        y_baseline_new = y_0_quad(J_small, b_iso)
        for h in [1,2]:
            h_i1, h_i3  = find_iso_rate_ca3(yca1= y_baseline_new[3], yca3 = y_baseline_new[0], h=h, J0 = J0, g=g, g_ii=g_ii, b = b_iso, h_i_min = 1, h_i_max = 2, type = "quadratic", n_points = 1000)
            h_list.append(h)
            g_list.append(g)
            J =  hippo_weights(index_dict, A, h3 = h, h1 = h, g = g, J = J0,  g_ii = 1, i_plast = h_i1, i_plast_3=h_i3)
            b = np.concatenate([b_iso[i]*np.ones(cells_per_region[i]) for i in range(6)])
            J_small = sum_by_region(J, index_dict=index_dict)
        
            y_q_red = y_0_quad(J_small, b_iso)
            y_full =  y_0_quad(J, b)
           
            correction = mean_by_region(np.real(loop_by_integration(J, y_full, b)), index_dict)
            y_corrected = y_q_red + correction 

            ys_pred_engram.append(y_corrected[3])
            ys_pred_non_engram.append(y_corrected[4])

            gain =  2*(J_small@y_corrected+ b_iso)
            J_lin =J_small* gain[...,None]
            D = np.linalg.inv(np.eye(6) - J_lin)
            pred_cors = cor_pred( J = J_lin , Ns = cells_per_region, y= y_q_red)
            cors_ee.append(pred_cors[3,3])
            cors_en.append(pred_cors[3,4])
            cors_nn.append(pred_cors[4,4])
            gc.collect()
            v, spktimes = sim_glm_pop(J=J,  E=b, dt = dt, tstop=tstop,  v_th = 0, maxspikes = tstop * N, p = 2)
            with open("../results/compare_inhib/spktimes_g={}h={}.pkl".format(g,h), "wb") as f :
                pkl.dump(obj = spktimes, file = f)
            neurons = index_dict['CA1E']
            rates = [rate(spktimes, i, tstop) for i in neurons]
            mean_rate = np.mean(rates)
            ys_sim_engram.append(mean_rate)
            neurons = index_dict['CA1P']
            rates = [rate(spktimes, i, tstop) for i in neurons]
            mean_rate = np.mean(rates)
            ys_sim_non_engram.append(mean_rate)

            engram_cells = index_dict["CA1E"]
            non_engram_cells = index_dict["CA1P"]


            sim_cors_ee.append(mean_pop_correlation(spktimes, engram_cells, dt, tstop))
            sim_cors_en.append(two_pop_correlation(spktimes, engram_cells, non_engram_cells, dt, tstop))
            sim_cors_nn.append(mean_pop_correlation(spktimes, non_engram_cells, dt, tstop))


df = pd.DataFrame({"g" : g_list, "h" : h_list, "pred_rate_engram" : ys_pred_engram, "pred_rate_non_engram" : ys_pred_non_engram,  
                                               "sim_rate_engram" : ys_sim_engram, "sim_rate_non_engram" : ys_sim_non_engram, 
                                               "pred_cor_engram_vs_engram": cors_ee,  "pred_cor_non_engram_vs_non_engram": cors_nn,"pred_cor_engram_vs_non_engram": cors_en, 
                                               "sim_cor_engram_vs_engram": sim_cors_ee,  "sim_cor_non_engram_vs_non_engram": sim_cors_nn,"sim_cor_engram_vs_non_engram": sim_cors_en})

with open("../results/compare_inhib/raw_df.pkl", "wb") as f:
    pkl.dump(obj = df, file = f)

print(df.head())
df = df.groupby(["g", "h"]).mean()
print(df)
df = df.reset_index()
print(df.columns)

# Pivot the DataFrame
pivoted_df = df.pivot(index='g', columns='h', values=["pred_rate_engram", "pred_rate_non_engram", "sim_rate_engram" , "sim_rate_non_engram", 
"pred_cor_engram_vs_engram",  "pred_cor_non_engram_vs_non_engram" , "pred_cor_engram_vs_non_engram", 
"sim_cor_engram_vs_engram",  "sim_cor_non_engram_vs_non_engram" , "sim_cor_engram_vs_non_engram"])

# Flatten the multi-level columns
pivoted_df.columns = [f'{col[0]}_h={col[1]}' for col in pivoted_df.columns]

# Reset the index
df= pivoted_df.reset_index()

columns_to_process = [
    'pred_rate_engram_h=1', 'pred_rate_engram_h=2',
    'pred_rate_non_engram_h=1', 'pred_rate_non_engram_h=2',
    'sim_rate_engram_h=1', 'sim_rate_engram_h=2', 
    'sim_rate_non_engram_h=1', 'sim_rate_non_engram_h=2', 
    'pred_cor_engram_vs_engram_h=1', 'pred_cor_engram_vs_engram_h=2', 
    'pred_cor_engram_vs_non_engram_h=1', 'pred_cor_engram_vs_non_engram_h=2',
     "pred_cor_non_engram_vs_non_engram_h=1",  "pred_cor_non_engram_vs_non_engram_h=2" , 
     "sim_cor_engram_vs_engram_h=1",  "sim_cor_engram_vs_engram_h=2",  
     "sim_cor_non_engram_vs_non_engram_h=1" , "sim_cor_non_engram_vs_non_engram_h=2" , 
     "sim_cor_engram_vs_non_engram_h=1",  "sim_cor_engram_vs_non_engram_h=2"
]

for column in columns_to_process:
    if column.endswith('_h=1'):
        # Extract the corresponding column with h=2
        column_h2 = column.replace('_h=1', '_h=2')
        
        # Create a new column name for the ratio
        ratio_column_name = column.replace('_h=1', '_ratio')
        
        # Calculate the ratio and add it to the DataFrame
        df[ratio_column_name] = df[column_h2] / df[column]

with open("../results/compare_inhib/df.pkl", "wb") as f:
    pkl.dump(obj = df, file = f)

print(df)