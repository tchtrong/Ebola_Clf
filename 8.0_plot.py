# %%
from utils.common import DIR, get_folder
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
from utils.plot import heatmap, annotate_heatmap


# %%
result_folder = get_folder(dir_type=DIR.LDA_RESULTS, no_X=True, fimo=True)
result_files = list(result_folder.glob('*/rbf/*.csv'))
for file in result_files:
    result_ = pd.read_csv(file, index_col=0)
    fig, ax = plt.subplots(figsize=(9,9))
    im, cbar = heatmap(result_.values, result_.index.values, result_.columns.values, ax=ax,
                    cmap="YlGn", cbarlabel="Accuracy")
    texts = annotate_heatmap(im, valfmt="{x:.2f}")
    fig.tight_layout()
    fig.savefig(file.with_suffix('.svg'))
    plt.close(fig)

# %%
