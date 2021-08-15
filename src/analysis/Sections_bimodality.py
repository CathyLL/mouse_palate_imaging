from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.analysis import data

data[['Stage', 'Sample', 'AP', 'CultureTime']] = data.Filename.str.split(' ', expand=True)

data = data.groupby(['Stage', 'Sample', 'AP']).agg({'Angle': 'mean', 'Length': 'mean'})

# %% CALCULATE Z SCORES GROUPED BY STAGE & AP

data.groupby(['Stage', 'AP']).mean()
data['Standardised Angle'] = (data['Angle'] - data.groupby(['Stage', 'AP']).Angle.transform('mean')) / data.groupby(['Stage', 'AP']).Angle.transform('std', ddof=0)
data['Standardised Length'] = (data['Length'] - data.groupby(['Stage', 'AP']).Length.transform('mean')) / data.groupby(['Stage', 'AP']).Length.transform('std', ddof=0)

data = data.drop(columns=['Angle', 'Length'])
data = data.reset_index()
data = data.set_index(['Stage', 'Sample'], append=True)

# %% PLOT GRAPH

g = sns.PairGrid(data, hue='AP')
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=2, bw_method=.5)
g.add_legend()

plt.show()