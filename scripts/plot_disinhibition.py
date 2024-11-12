import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl

plt.style.use('paper_style.mplstyle')
engram_color =   "#F37343"
non_engram_color =  "#06ABC8"
cross_color = "#FEC20E"

with open("../results/compare_inhib/df_disinhib.pkl", "rb") as f:
    df =  pkl.load(file = f)

print(df.columns)

filtered_df = df[df['h'] == 2]

# Step 2: Melt the data to create long-format dataframe for easier plotting
melted_df = pd.melt(
    filtered_df,
    id_vars=['g', 'b_diff'], 
    value_vars=['pred_cor_engram_vs_engram', 'pred_cor_non_engram_vs_non_engram', 'pred_cor_engram_vs_non_engram'],
    var_name='correlation_type', 
    value_name='correlation_value'
)

# Rename values in the 'correlation_type' column
melted_df['correlation_type'] = melted_df['correlation_type'].replace({
    'pred_cor_engram_vs_engram': 'Engram vs. engram',
    'pred_cor_non_engram_vs_non_engram': 'Non-engram vs. non-engram',
    'pred_cor_engram_vs_non_engram': 'Engram vs. non-engram'
})

# Step 3: Plot
# Loop over each unique value of 'g' and make a plot
fig, axs = plt.subplots(1,3, figsize = (6,2))
for i, g_value in enumerate(melted_df['g'].unique()):
    subset = melted_df[melted_df['g'] == g_value]
    
    sns.lineplot(
        data=subset,
        x='b_diff', 
        y='correlation_value', 
        hue='correlation_type',
        palette= [engram_color, non_engram_color,  cross_color], ax = axs[i],
    )
    
    axs[i].set_title(f'g = {g_value}')
    axs[i].set_xlabel('Disinhibition')
    axs[i].set_ylabel('')
    if i == 0:
        axs[i].legend(title='')
    else:
        axs[i].get_legend().remove()


sns.despine()
plt.tight_layout()
plt.savefig("../results/plots/disinhibition_corrs.pdf")

plt.show()

# do the same thing for the rates
melted_df = pd.melt(
    filtered_df,
    id_vars=['g', 'b_diff'], 
    value_vars=['pred_rate_engram', 'pred_rate_non_engram'],
    var_name='rate_type', 
    value_name='rate_value',
)

# Rename values in the 'correlation_type' column
melted_df['rate_type'] = melted_df['rate_type'].replace({
    'pred_rate_engram': 'Engram',
    'pred_rate_non_engram': 'Non-engram'
})

# Step 3: Plot
# Loop over each unique value of 'g' and make a plot
fig, axs = plt.subplots(1,3, figsize = (6,2))
for i, g_value in enumerate(melted_df['g'].unique()):
    subset = melted_df[melted_df['g'] == g_value]
    
    sns.lineplot(
        data=subset,
        x='b_diff', 
        y='rate_value', 
        hue='rate_type',
        palette= [engram_color, non_engram_color], ax = axs[i],
    )
    
    axs[i].set_title(f'g = {g_value}')
    axs[i].set_xlabel('Disinhibition')
    axs[i].set_ylabel('Rate')
    if i == 0:
        axs[i].legend(title='')
    else:
        axs[i].get_legend().remove()

sns.despine()
plt.tight_layout()
plt.savefig("../results/plots/disinhibition_rates.pdf")
plt.show()