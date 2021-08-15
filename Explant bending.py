from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import numpy as np

# %% LOAD DATA

data = pd.read_excel(r'./data/raw_sections\Additional.xlsx')

# g = data.groupby(['Time', 'Stage'])
# data['Mean Angle'] = g['Angle'].transform('mean')
# data['Mean Length'] = g['Length'].transform('mean')

# data = data.loc[(data['Time'] != 0) & (data['Time'] != 20) & (data['Time'] != 72)]

data = data.loc[(data['Time'] == 0) | (data['Time'] == 20) | (data['Time'] == 72)]

# %% PLOT GRAPH FOR ANGLE

ax = sns.barplot(
    data=data,
    x='Stage',
    y='Angle',
    hue='Time',
    order=['E12.5', 'E13.5'],
    hue_order=[0, 20, 72],
    saturation=0.8,
    ci=95,
    errwidth=1.5,
    capsize=0.06
)

ax = sns.stripplot(
    data=data,
    x='Stage',
    y='Angle',
    # order=['E12.5', 'E13.5', 'E15.5'],
    hue='Time',
    order=['E12.5', 'E13.5'],
    hue_order=[0, 20, 72],
    dodge=True,
    edgecolor="black",
    linewidth=.75,
    ax=ax,
)
#
ax.set(ylim=(0, 200))
ax.set_xlabel('Embryonic stage', fontsize=14)
ax.set_ylabel('Inter-Shelf Angle (degrees)', fontsize=14)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)

# handles, labels = ax.get_legend_handles_labels()
# l = plt.legend(
#     handles[2:5],
#     labels[2:5],
#     title='Time after dissection',
#     loc='upper left',
#     bbox_to_anchor=(1.04,1),
#     frameon=False,
#     fontsize=14,
#     title_fontsize=14
# )

sns.despine(top=True, right=True, left=True, bottom=False)
plt.tight_layout()
plt.show()

# %% PLOT GRAPH FOR LENGTH

ax = sns.barplot(
    data=data,
    x='Stage',
    y='Length',
    # order=['E12.5', 'E13.5', 'E15.5'],
    hue='Time',
    order=['E12.5', 'E13.5'],
    hue_order=[0, 20, 72],
    saturation=0.8,
    ci='sd',
    errwidth=1.5,
    capsize=0.06
)

ax = sns.stripplot(
    data=data,
    x='Stage',
    y='Length',
    # order=['E12.5', 'E13.5', 'E15.5'],
    hue='Time',
    order=['E12.5', 'E13.5'],
    hue_order=[0, 20, 72],
    dodge=True,
    edgecolor="black",
    linewidth=.75,
    ax=ax,
)

ax.set(ylim=(0.0, 1.0))
ax.set_xlabel('Embryonic stage', fontsize=14)
ax.set_ylabel('Distance between Hinge Regions (mm)', fontsize=14)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
handles, labels = ax.get_legend_handles_labels()
l = plt.legend(
    handles[2:5],
    labels[2:5],
    title='Time after dissection',
    loc='upper left',
    bbox_to_anchor=(1.04,1),
    frameon=True,
    fontsize=14,
    title_fontsize=14
)
sns.despine(top=True, right=True, left=True, bottom=False)
plt.tight_layout()
plt.show()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%% T TEST

data1 = data.loc[(data['Time'] == 0) & (data['Stage'].str.contains('E13.5')), 'Angle']
data2 = data.loc[(data['Time'] == 20) & (data['Stage'].str.contains('E13.5')), 'Angle']
# print(data1)
# print(data2)

print(ttest_ind(data1, data2, equal_var=False))