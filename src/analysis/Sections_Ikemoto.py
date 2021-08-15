from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.analysis import data

# %% ORGANISE DATA

data[['Stage', 'Sample', 'AP', 'CultureTime']] = data.Filename.str.split(' ', expand=True)
data['CultureTime'] = data['CultureTime'].astype(float)

data = data.groupby(['Stage', 'Sample', 'AP', 'CultureTime']).agg({'Angle': 'mean', 'Length': 'mean'})

data = data.reset_index()

# %% ORGANISE DATA FOR IKEMOTO GROWTH

data = data.loc[
       data['Stage'].str.contains('E12.5') &
       (((data['Sample'].str.contains('CulIk'))) |
       ((data['Sample'].str.contains('Cul')) & (data['CultureTime'] == 0)))
, :]

data = data.sort_values('CultureTime')
data['CultureTime'] = data['CultureTime'].astype(int).astype(str)

data['AP'] = data['AP'].str.replace('ant','anterior').replace('mid','middle').replace('post','posterior')

# %% PLOT GRAPH

ax = sns.lineplot(
    data=data,
    x='CultureTime',
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
    data=data,
    x='CultureTime',
    y='Length',
    hue='AP',
    hue_order=['anterior', 'middle', 'posterior'],
    palette="mako_r",
    jitter=.1,
    dodge=False,
    size=5,
    edgecolor="gray",
    linewidth=.5,
    alpha=.7,
    ax=ax,
)

ax.set(ylim=(0.25, 1.0), xlim=(-.18, 1.18))
ax.set_xlabel('Time in Ikemoto Culture (hours)', fontsize=14)
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
