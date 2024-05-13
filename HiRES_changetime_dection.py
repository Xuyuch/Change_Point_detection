#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 15:22:05 2024

@author: yuchen
"""

import os
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as stats
import scipy.sparse as sparse
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import scvi
import plotnine as p9

HIRES = pd.read_csv('/storage/Yuchen/Change_Point/GSE223917_HiRES_emb.rna.umicount.tsv',sep="\t")
meta=pd.read_csv('/storage/Yuchen/Change_Point/GSE223917_HiRES_emb_metadata.csv')
HIRES.set_index(HIRES.columns[0], inplace=True)
adata = sc.AnnData(HIRES.T)
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
info_dict = meta.set_index('Cellname')[['Stage', 'Celltype']].T.to_dict('list')
adata.obs['Stage'] = adata.obs_names.map(lambda x: info_dict[x][0] if x in info_dict else 'Unknown')
adata.obs['Celltype'] = adata.obs_names.map(lambda x: info_dict[x][1] if x in info_dict else 'Unknown')
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata)
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata,color="Celltype")


cell_types = [
    "epiblast and PS",
    "neural ectoderm",
    "NMP",
    "neural tube",
    "notochord",
    "radial glias",
    "early neurons",
    "epithelial cells",
    "oligodendrocytes and progenitors",
    "endoderm"
]
adata_neuron = adata[adata.obs['Celltype'].isin(cell_types)].copy()
sc.pl.umap(adata_neuron,color="Celltype")
sc.pp.filter_cells(adata_neuron, min_genes=200)
sc.pp.filter_genes(adata_neuron, min_cells=3)

umap1 = adata_neuron.obsm['X_umap'][:, 0]
keep_cells = umap1 >= 7.7
adata_neuron_selected= adata_neuron[keep_cells].copy()
umap2 = adata_neuron_selected.obsm['X_umap'][:, 0]
second_lowest_index = np.argsort(umap2)[1] 
adata_neuron_selected.uns['iroot'] = second_lowest_index
#adata_neuron.uns['iroot'] = np.flatnonzero(adata_neuron.obs['Stage'] == 'E70')[0]
sc.tl.diffmap(adata_neuron_selected)
sc.tl.dpt(adata_neuron_selected)
sc.pl.umap(adata_neuron_selected,color="dpt_pseudotime")

adata_neuron_selected = adata_neuron_selected[ adata_neuron_selected.obs['dpt_pseudotime'].argsort(),:]
count_matrix = adata_neuron_selected.X

# If the count matrix is in a sparse format, you might want to convert it to a dense format for ease of use
import numpy as np
if sp.sparse.issparse(count_matrix):
    count_matrix = count_matrix.toarray()
gene_names = adata_neuron.var_names

# Get cell names (sorted by 'dpt_pseudotime')
cell_names = adata_neuron_selected.obs_names  # These are already ordered by 'dpt_pseudotime'

# Create a DataFrame from the count matrix for better readability and manipulation
df_counts = pd.DataFrame(data=count_matrix, index=cell_names, columns=gene_names)
#df_counts.to_csv('/storage/Yuchen/Change_Point/ordered_counts_by_pseudotime.csv')
array_data = df_counts.values
import ruptures as rpt 
model = "rbf"  # "l2", "rbf"
algo = rpt.KernelCPD(kernel="rbf").fit(array_data)

result = algo.predict(pen=1)



df_counts['pseudo_stage_7'] = 1  # Initialize all cells to stage 1 initially
for i, point in enumerate(result):
    df_counts.loc[df_counts.index[point:], 'pseudo_stage_7'] = i + 2  

adata_neuron_selected.obs = adata_neuron_selected.obs.join(df_counts[['pseudo_stage_7']])
sc.pl.umap(adata_neuron_selected,color='pseudo_stage_7')

fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # Adjust the size as necessary

# Flatten the axes array for easy indexing
axs = axs.flatten()

# Plot UMAP for 'pseudo_stage_7'
sc.pl.umap(adata_neuron_selected, color='pseudo_stage_7', color_map="RdYlBu", ax=axs[0], show=False, title='Pseudo Stage 7')

# Plot UMAP for 'Stage'
sc.pl.umap(adata_neuron_selected, color='Stage', ax=axs[1], show=False, title='Stage')

# Plot UMAP for 'Celltype'
sc.pl.umap(adata_neuron_selected, color='Celltype', ax=axs[2], show=False, title='Celltype')

# Plot UMAP for 'dpt_pseudotime'
sc.pl.umap(adata_neuron_selected, color='dpt_pseudotime', ax=axs[3], show=False, title='DPT Pseudotime')

# Adjust layout to prevent overlap and display the plots
plt.tight_layout()
plt.show()