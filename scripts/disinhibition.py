import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os


from src.theory import  y_0_quad, find_iso_rate_input, cor_pred, find_iso_rate_ca3
from src.theory import CA1_internal_cov_offdiag, CA1_inherited_cov, CA3_internal_cov, CA3_E_from_E, CA3_E_from_N, CA3_E_from_I
from src.correlation_functions import  sum_by_region
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

b_small = np.array([.4, .4, .5, .4, .4, .5])   #without excitability
J0 = .2
g_ii = 1
g_min = 1
g_max = 3
n_g = 3
h_h = 2
gs = np.linspace(g_min, g_max, n_g)
diff_min = 0
diff_max = .5
n_diffs = 10
b_diffs = np.linspace(diff_min, diff_max, n_diffs)

J =  hippo_weights(index_dict, A, h3 = 1, h1 = 1, g = 1, J = J0,  g_ii =  g_ii)
J_baseline = sum_by_region(J, index_dict=index_dict)


y_baseline = y_0_quad(J_baseline, b_small)
ys_pred_engram = []
ys_pred_non_engram = []
h_list = []
g_list = []
b_diff_list = []
cors_ee = []
cors_en = []
cors_nn = []

CA1_internal = []
CA1_inherited = []
CA3 = [] 
from_CA3I = []
from_CA3E = []
from_CA3N =[]

for g in gs:
    J_small =macro_weights(J=J0, h3 = 1, h1 =1, g =g)
    b_iso =find_iso_rate_input(target_rate_1= y_baseline[3], target_rate_3=y_baseline[0], J = J_small, b = b_small, b0_min = 0, b0_max = .5, n_points=2000, plot=False)
    for b_diff in b_diffs:
        for h in [1,2]:
            h_i1, h_i3  = find_iso_rate_ca3(y_baseline[3],y_baseline[0], h=h, J0 = J0, g=g, g_ii=g_ii, b = b_iso, h_i_min = 1, h_i_max = 2, type = "quadratic", n_points = 1000)
            h_list.append(h)
            g_list.append(g)
            b_diff_list.append(b_diff)
            b_dis = np.copy(b_iso)
            b_dis[-1] -= b_diff
            J =  hippo_weights(index_dict, A, h3 = h, h1 = h, g = g, J = J0,  g_ii = 1, i_plast = h_i1, i_plast_3=h_i3)
            J_small = sum_by_region(J, index_dict=index_dict)
        
            y_q_red = y_0_quad(J_small, b_dis)
        
            ys_pred_engram.append(y_q_red[3])
            ys_pred_non_engram.append(y_q_red[4])

            gain =  2*(J_small@y_q_red+ b_dis)
            J_lin =J_small* gain[...,None]
            D = np.linalg.inv(np.eye(6) - J_lin)
            pred_cors = cor_pred( J = J_lin , Ns = cells_per_region, y = y_q_red)
            cors_ee.append(pred_cors[3,3])
            cors_en.append(pred_cors[3,4])
            cors_nn.append(pred_cors[4,4])
            CA1_internal.append(CA1_internal_cov_offdiag(J = J_small, r = y_q_red, b = b_dis, N = cells_per_region)[0,0])
            CA1_inherited.append(CA1_inherited_cov(J = J_small, r = y_q_red, b = b_dis, N = cells_per_region)[0,0])
            CA3.append(CA3_internal_cov(J = J_small, r = y_q_red, b = b_dis, N = cells_per_region)[0,0])

            from_CA3I.append(CA3_E_from_I(J = J_small, r = y_q_red, b = b_dis, N = cells_per_region))
            from_CA3E.append(CA3_E_from_E(J = J_small, r = y_q_red, b = b_dis, N = cells_per_region)) 
            from_CA3N.append(CA3_E_from_N(J = J_small, r = y_q_red, b = b_dis, N = cells_per_region))  



decomp_df = pd.DataFrame({"g" : g_list, "h" : h_list, "b_diff": b_diff_list,  "CA1_internal" :  CA1_internal, "CA1_inherited" : CA1_inherited, "CA3" : CA3, 
"from_CA3I" :from_CA3I, "from_CA3E" : from_CA3E, "from_CA3N" : from_CA3N })

with open("../results/compare_inhib/decomposition_df_disinhib.pkl", "wb") as f:
    pkl.dump(obj = decomp_df, file = f)

df = pd.DataFrame({"g" : g_list, "h" : h_list,"b_diff": b_diff_list, "pred_rate_engram" : ys_pred_engram, "pred_rate_non_engram" : ys_pred_non_engram,  
                                               "pred_cor_engram_vs_engram": cors_ee,  "pred_cor_non_engram_vs_non_engram": cors_nn,"pred_cor_engram_vs_non_engram": cors_en})

print(df.head())
with open("../results/compare_inhib/df_disinhib.pkl", "wb") as f:
    pkl.dump(obj = df, file = f)