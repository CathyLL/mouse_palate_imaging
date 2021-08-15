from pathlib import Path

import pandas as pd

data_folder = Path(__file__).parents[2] / 'data/raw_sections'
all_files = data_folder.glob("*.csv")

data = pd.concat([
    pd.read_csv(file, index_col=0).assign(Filename=file.stem)
    for file in all_files
])