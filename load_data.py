import multiprocessing as mp
import os
import warnings
from collections import Counter, defaultdict
from os.path import exists, join
from subprocess import check_call

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import sklearn
import xgboost as xgb
from IPython.display import Image
from sklearn.decomposition import PCA
from tqdm import tqdm
from xgboost import XGBClassifier

from sklearn.metrics import *
import sklearn


warnings.filterwarnings("ignore")

os.chdir("/mnt/ivy/thliao/project/ML_oxygen")


# shap.initjs()
# %pylab inline


tax_tab = "/home-user/thliao/.cache/ncbi-genome-download/taxonomy.tab"
tax_df = pd.read_csv(tax_tab, sep="\t", index_col=0)
genome2tax = tax_df.to_dict(orient="index")


def get_trait_data():
    NCBI_df = pd.read_csv("/home-user/thliao/project/ML_oxygen/training_sets/processed_data/NCBI_trait.tab",
                          sep='\t', index_col=0)
    extra_df = pd.read_csv(
        '/mnt/ivy/thliao/project/ML_oxygen/add_data/bacdiv_gid2info.csv', sep='\t', index_col=0)
    extra_df = extra_df.loc[~extra_df.index.duplicated(), :]
    gids = [_ for _ in open(
        '/mnt/ivy/thliao/project/ML_oxygen/add_data/gids.list').read().split('\n') if _]
    ngids = [_ for _ in gids if _.split('.')[0] in extra_df.index]
    extra_df = extra_df.reindex([_.split('.')[0] for _ in ngids])
    extra_df.index = ngids

    NCBI_df = pd.concat([NCBI_df, extra_df], axis=0)
    sub_NCBI_df = NCBI_df.loc[~NCBI_df['metabolism'].isna(), :]
    sub_NCBI_df = sub_NCBI_df.loc[~sub_NCBI_df.index.duplicated(), :]

    remap_metabolism = {'anaerobe': 'anaerobic',
                        'aerobe': 'aerobic',
                        'obligate anaerobe': 'obligate anaerobic',
                        'obligate aerobe': 'obligate aerobic',
                        'microaerophile': 'microaerophilic',
                        'facultative anaerobe': 'facultative',
                        'facultative aerobe': 'facultative',
                        'microaerotolerant': 'microaerophilic'
                        }
    sub_NCBI_df.loc[:, 'metabolism'] = [remap_metabolism.get(
        _, _) for _ in list(sub_NCBI_df['metabolism'])]
    sub_NCBI_df = sub_NCBI_df.drop(['GCA_902651685.1', 'GCA_000716135.1'])
    return sub_NCBI_df


sub_NCBI_df = get_trait_data()
aids = list(sub_NCBI_df.index)

specific_tax_df = pd.read_csv(
    "/home-user/thliao/project/ML_oxygen/taxdf.csv", sep="\t", index_col=0
)

bac_ids = list(
    specific_tax_df.index[specific_tax_df['superkingdom'] == 'Bacteria'])
print(len(bac_ids))

bac_ids = [_ for _ in bac_ids if _ in sub_NCBI_df.index]
print(len(bac_ids))
sub_NCBI_df = sub_NCBI_df.reindex(bac_ids)

complete_df = pd.read_csv(
    "/mnt/ivy/thliao/project/ML_oxygen/training_sets/micomplete_o/complet_df.tab",
    sep="\t",
    index_col=0,
)
#complete_df.index = [_[:-1]+'.'+_[-1] for _ in complete_df.index]

genome2completeness = dict(
    zip(complete_df.index, complete_df["Weighted completeness"]))
completeness_array = np.array(
    [genome2completeness.get(_, 0) for _ in bac_ids]
)

output_dir = f"/home-user/thliao/project/ML_oxygen/training_sets/processed_data"
prevalent_tax = ['Proteobacteria',
                 'Firmicutes',
                 'Actinobacteria',
                 'Bacteroidetes']
y_mapping = {"aerobic": 1,
             "obligate aerobic": 1,
             "anaerobic": 0,
             "facultative": 1,
             "microaerophilic": 1,
             "obligate anaerobic": 0,
             }

genome2oxy = sub_NCBI_df['metabolism'].to_dict()
genome2oxy_bin = {k: y_mapping[v]
                  for k, v in sub_NCBI_df['metabolism'].to_dict().items()}

# phy_array = np.array([genome2tax[_.split('.')[0]]['phylum']
#                       for _ in sub_NCBI_df.index])

kegg_bin_df = pd.read_csv(f"/mnt/ivy/thliao/project/ML_oxygen/training_sets/processed_data/20190810kegg_anno.tab", sep="\t", index_col=0)
# print(pfam_bin_df.shape, pfam_num_df.shape)
print(kegg_bin_df.shape)
print(Counter(sub_NCBI_df["metabolism"]))

y_mapping = {
    "aerobic": 1,
    "obligate aerobic": 1,
    "anaerobic": 0,
    "facultative": 1,
    "microaerophilic": 1,
    "obligate anaerobic": 0,
}
bac_data_df = sub_NCBI_df.reindex(bac_ids)
y_raw = bac_data_df["metabolism"].values

bac_kegg_bin_df = kegg_bin_df.reindex(bac_data_df.index).fillna(0)
bac_kegg_bin_df.loc[:,'completeness'] = complete_df.reindex(bac_data_df.index)["Completeness"]
bac_kegg_bin_df.loc[:,'number of Coding seqs'] = complete_df.reindex(bac_data_df.index)["CDs"]
X = bac_kegg_bin_df
X = X.loc[:,X.columns[:-2]]
#X = X.loc[:,X.sum(0)!=0]
sub_tax_df = tax_df.reindex([_.split('.')[0] for _ in bac_data_df.index])
sub_tax_df.index = bac_data_df.index
#sub_tax_df

y_bin = np.array([y_mapping[_] for _ in y_raw])



pfam_bin_df = pd.read_csv(f"/mnt/ivy/thliao/project/ML_oxygen/training_sets/processed_data/pfam_anno.tab", sep="\t", index_col=0)
bac_pfam_bin_df = pfam_bin_df.reindex(bac_data_df.index).fillna(0)
bac_pfam_bin_df.loc[:,'completeness'] = complete_df.reindex(bac_data_df.index)["Completeness"]
bac_pfam_bin_df.loc[:,'number of Coding seqs'] = complete_df.reindex(bac_data_df.index)["CDs"]
X_pfam = bac_pfam_bin_df
X_pfam = X_pfam.loc[:,X_pfam.columns[:-2]]