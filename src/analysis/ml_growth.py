from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_style("whitegrid")
sns.set_context('notebook')

results_folder = Path(__file__).parents[2] / 'results'

# %% Load Data

data = (
    pd.read_excel(results_folder / 'data.xlsx', sheet_name='Distances', index_col='File')
        .loc[:, ['Dist(Med shelf L, Post whis L)', 'Dist(Med shelf R, Post whis R)', 'Info', 'Culture', 'Time']]
        .set_index(['Info', 'Culture', 'Time'], append=True)
)

data = data[['Dist(Med shelf L, Post whis L)', 'Dist(Med shelf R, Post whis R)']].mean(axis=1).to_frame(name='Mean Shelf Width')
data = data.reset_index()

data['CultureType'] = np.where(
    data.Culture.str.contains('CulIk', regex=False),
    'Ikemoto',
    'Normal'
)

data['Stage/Fixed'] = np.where(
     (data.Culture.str.contains('Fix', regex=False)) & (data.CultureType.str.contains('Normal', regex=False)),
     'In vivo',
     data.Info.str.extract(r'(?P<capture>E[1-9\.]*)')['capture']
 )

cultured_mapping = {
    0: 12.5,
    17: 13,
    24: 13.5,
    41: 14,
    48: 14.5,
    65: 15,
    72: 15.5
}

not_fixed = data['Stage/Fixed'] != 'In vivo'
data.loc[not_fixed, 'Time'] = data.loc[not_fixed, 'Time'].map(cultured_mapping)
data.loc[not_fixed, 'Time'] += data.loc[not_fixed, 'Stage/Fixed'].str.replace('E', '').astype(float) - 12.5
data.loc[~not_fixed, 'Time'] = data.loc[~not_fixed, 'Info'].str.replace('E', '').astype(float)

duplicated_data = data.loc[(data.Time == 12.5) & (data['Stage/Fixed'].isin(['In vivo', 'E12.5']))].copy()
duplicated_data['Stage/Fixed'] = np.where(duplicated_data['Stage/Fixed'] == 'E12.5', 'In vivo', 'E12.5')

data = pd.concat([
    data,
    duplicated_data
])

duplicated_data = data.loc[(data.Time == 13.5) & (data['Stage/Fixed'].isin(['In vivo', 'E13.5']))].copy()  # Subset to time 13.5 but only in In vivo or E13.5 samples
duplicated_data['Stage/Fixed'] = np.where(duplicated_data['Stage/Fixed'] == 'E13.5', 'In vivo', 'E13.5')  # Mirror/flip the stage/fixed

data = pd.concat([
    data,
    duplicated_data
])


# %% Plot

# Ikemoto sample
#
# ax = sns.lineplot(
#     data=(
#         data.loc[
#                 ((data['CultureType'] == 'Ikemoto') | (data['Stage/Fixed'] == 'In vivo'))  # Either Ikemoto or In Vivo (whether Ikemoto or Normal, doesn't matter) (:Original code:)
#                 | ((data.Time == 12.5) & (data['Stage/Fixed'].isin(['E12.5', 'In vivo'])))
#                 | ((data.Time == 13.5) & (data['Stage/Fixed'].isin(['E13.5', 'In vivo'])))
#             ]
#             .sort_values(
#                 'Stage/Fixed',
#                 key=lambda x: x.str.replace('In vivo', '0').str.replace('E', '').astype(float)
#             )
#     ),
#     x='Time',
#     y='Mean Shelf Width',
#     hue='Stage/Fixed',
#     hue_order=['In vivo', 'E12.5', 'E13.5'],
#     err_style='bars',
#     palette={'E12.5': '#f210ea', 'E13.5': '#2AA61B', 'In vivo': '#000054'},
#     # linestyle='dashed',
#     lw=3
# )


ax = sns.lineplot(
    data=(
        data.loc[
                ((data['CultureType'] == 'Ikemoto') | (data['Stage/Fixed'] == 'In vivo'))  # Either Ikemoto or In Vivo (whether Ikemoto or Normal, doesn't matter) (:Original code:)
            ]
            .sort_values(
                'Stage/Fixed',
                key=lambda x: x.str.replace('In vivo', '0').str.replace('E', '').astype(float)
            )
    ),
    x='Time',
    y='Mean Shelf Width',
    hue='Stage/Fixed',
    hue_order=['In vivo', 'E12.5', 'E13.5'],
    err_style='bars',
    palette={'E12.5': '#f210ea', 'E13.5': '#2AA61B', 'In vivo': '#000054'},
    # linestyle='dashed',
    lw=3
)



ax = sns.lineplot(
    data=(
        pd.concat([
            (
                data
                    .loc[
                        ((data.Time == 12.5) & (data['Stage/Fixed'] == 'In vivo'))
                        | ((data.Time == 13.5) & (data['Stage/Fixed'] == 'E12.5') & (data['CultureType'] == 'Ikemoto'))
                    ]
                    .copy()
                    .assign(**{'Stage/Fixed': 'E12.5 Global'})
            ),
            (
                data
                    .loc[
                    ((data.Time == 13.5) & (data['Stage/Fixed'] == 'In vivo'))
                    | ((data.Time == 14.5) & (data['Stage/Fixed'] == 'E13.5') & (data['CultureType'] == 'Ikemoto'))
                    ]
                    .copy()
                    .assign(**{'Stage/Fixed': 'E13.5 Global'})
            ),
        ])
            .sort_values(
                'Stage/Fixed',
                key=lambda x: x.str.replace('In vivo', '0').str.replace('E', '').str.replace(' Global', '').astype(float)
            )
    ),
    x='Time',
    y='Mean Shelf Width',
    hue='Stage/Fixed',
    hue_order=['E12.5 Global', 'E13.5 Global'],
    err_style='bars',
    palette={'E12.5 Global': '#f210ea', 'E13.5 Global': '#2AA61B'},
    lw=3,
    ls='dashed',
    ax=ax,
)

ax.set_xlabel('Embryonic Age (days)', fontsize=16)
ax.set_ylabel('ML Shelf Width (mm)', fontsize=16)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
# plt.legend(title=None, fontsize=16)
handles, labels = ax.get_legend_handles_labels()
handles[-1].set_linestyle('dashed')
handles[-2].set_linestyle('dashed')
l = plt.legend(
    handles,
    labels,
    title=None,
    frameon=True,
    fontsize=14,
    title_fontsize=14
)

sns.despine(left=True)
plt.tight_layout()
plt.show()


# Normal sample
#
# ax = sns.lineplot(
#     data=(
#         data
#             .loc[
#                 ((data['CultureType'] == 'Normal') | (data['Stage/Fixed'] == 'In vivo'))  # Either Normal or In Vivo (whether Ikemoto or Normal, doesn't matter) (:Original code:)
#                 | ((data.Time == 12.5) & (data['Stage/Fixed'].isin(['E12.5', 'In vivo'])))
#                 | ((data.Time == 13.5) & (data['Stage/Fixed'].isin(['E13.5', 'In vivo'])))
#             ]
#             .sort_values(
#                 'Stage/Fixed',
#                 key=lambda x: x.str.replace('In vivo', '0').str.replace('E', '').astype(float)
#             )
#     ),
#     x='Time',
#     y='Mean Shelf Width',
#     hue='Stage/Fixed',
#     hue_order=['In vivo', 'E12.5', 'E13.5'],
#     err_style='bars',
#     palette={'E12.5': '#f210ea', 'E13.5': '#2AA61B', 'In vivo': '#000054'},
#     lw=3
# )
# ax.set_xlabel('Embryonic Age (days)', fontsize=16)
# ax.set_ylabel('ML Shelf Width (mm)', fontsize=16)
# plt.tick_params(axis='x', labelsize=14)
# plt.tick_params(axis='y', labelsize=14)
# plt.legend(title=None, fontsize=16)
# sns.despine(left=True)
# plt.tight_layout()
# plt.show()
