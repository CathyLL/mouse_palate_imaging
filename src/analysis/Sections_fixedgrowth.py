from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from src.analysis import data

data[['Stage', 'Sample', 'AP', 'CultureTime']] = data.Filename.str.split(' ', expand=True)

data = data.groupby(['Stage', 'Sample', 'AP']).agg({'Angle': 'mean', 'Length': 'mean'})
data = data.reset_index()

# %% ORGANISE DATA

data['AP'] = data['AP'].str.replace('ant','anterior').replace('mid','middle').replace('post','posterior')

# %% PLOT GRAPH

ax = sns.lineplot(
    data=data.loc[data['Sample'].str.contains('Fix'),:],
    x='Stage',
    y='Length',
    hue='AP',
    hue_order=['anterior', 'middle', 'posterior'],
    palette="mako_r",
    ci=95,
    err_style='bars',
    err_kws={'capsize': 8, 'elinewidth': 3, 'capthick': 2},
    lw=4
)
sns.stripplot(
    data=data.loc[data['Sample'].str.contains('Fix'), :],
    x='Stage',
    y='Length',
    hue='AP',
    hue_order=['anterior', 'middle', 'posterior'],
    palette="mako_r",
    jitter=True,
    size=5,
    edgecolor="gray",
    linewidth=.5,
    alpha=.7,
    ax=ax,
)

ax.set(ylim=(0.25, 1.0))
ax.set_xlabel('Embryonic Age (days)', fontsize=14)
ax.set_ylabel('Shelf Length (mm)', fontsize=14)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)

handles, labels = ax.get_legend_handles_labels()
l = plt.legend(
    handles[0:3],
    labels[0:3],
    title='Shelf Region',
    loc='upper left',
    frameon=True,
    fontsize=14,
    title_fontsize=14
)
sns.despine(left=True)
plt.grid(axis='y', alpha=.5)
plt.tight_layout()
plt.show()

# %% Regression

fixed = data['Sample'].str.contains('Fix')
data.loc[fixed, 'Stage'] = data.loc[fixed, 'Stage'].str.replace('E', '').astype(float)

data = data.loc[data['Sample'].str.contains('Fix') & ((data['Stage'] == 12.5) | (data['Stage'] == 13.5)), :]
data.Stage = data.Stage.astype(float)
data['Stage'] = data['Stage'] - 12.5

# %% REGRESSION
model_data = smf.ols('Length ~ Stage', data=data).fit()

# %% PERCENTAGE CHANGE
model_data_adj = smf.ols(
        'Length ~ Stage',
        data=data.assign(Length=data['Length'] / model_data.params['Intercept'])
    ).fit()

print(model_data_adj.params['Stage'])



