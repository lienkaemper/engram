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
gain_min = 0
gain_max = 1
n_gains = 10
alphas = np.linspace(gain_min, gain_max, n_gains)

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
dis_list = []



for g in gs:
    J_small =macro_weights(J=J0, h3 = 1, h1 =1, g =g)
    b_iso =find_iso_rate_input(target_rate_1= y_baseline[3], target_rate_3=y_baseline[0], J = J_small, b = b_small, b0_min = 0, b0_max = .5, n_points=2000, plot=False)
    for alpha_dis in alphas:
        for h in [1,2]:
            h_i1, h_i3  = find_iso_rate_ca3(y_baseline[3],y_baseline[0], h=h, J0 = J0, g=g, g_ii=g_ii, b = b_iso, h_i_min = 1, h_i_max = 2, type = "quadratic", n_points = 1000)
            h_list.append(h)
            g_list.append(g)
            dis_list.append(1-alpha_dis)
            J =  hippo_weights(index_dict, A, h3 = h, h1 = h, g = g, J = J0,  g_ii = 1, i_plast = h_i1, i_plast_3=h_i3)
            J_small = sum_by_region(J, index_dict=index_dict)
            alpha = np.ones(6)
            alpha[-1] = alpha_dis
            y_q_red = y_0_quad(J_small, b_iso, gain = alpha)
        
            ys_pred_engram.append(y_q_red[3])
            ys_pred_non_engram.append(y_q_red[4])

            gain =  alpha * 2*(J_small@y_q_red+ b_iso)
            J_lin =J_small* gain[...,None]
            D = np.linalg.inv(np.eye(6) - J_lin)
            pred_cors = cor_pred( J = J_lin , Ns = cells_per_region, y = y_q_red)
            cors_ee.append(pred_cors[3,3])
            cors_en.append(pred_cors[3,4])
            cors_nn.append(pred_cors[4,4])


df = pd.DataFrame({"g" : g_list, "h" : h_list,"disinhibition": dis_list, "pred_rate_engram" : ys_pred_engram, "pred_rate_non_engram" : ys_pred_non_engram,  
                                               "pred_cor_engram_vs_engram": cors_ee,  "pred_cor_non_engram_vs_non_engram": cors_nn,"pred_cor_engram_vs_non_engram": cors_en})

print(df.head())
with open("../results/compare_inhib/df_disinhib.pkl", "wb") as f:
    pkl.dump(obj = df, file = f)