

import os
os.chdir('/mnt/storage3/thliao/project/ML_oxygen/comparison_softwares/bacdive-AI/')

import pandas as pd
df = pd.read_csv('/mnt/storage3/thliao/project/ML_oxygen/comparison_softwares/bacdive-AI/8943bac_granularity3.csv',index_col=1)


def get_trait_data():
    NCBI_df = pd.read_csv("/home-user/thliao/project/ML_oxygen/training_sets/processed_data/NCBI_trait.tab",
                          sep='\t', index_col=0)
    # 11063
    extra_df = pd.read_csv(
        '/mnt/ivy/thliao/project/ML_oxygen/add_data/bacdiv_gid2info.csv', sep='\t', index_col=0)
    extra_df = extra_df.loc[~extra_df.index.duplicated(), :]
    gids = [_ for _ in open(
        '/mnt/ivy/thliao/project/ML_oxygen/add_data/gids.list').read().split('\n') if _]
    ngids = [_ for _ in gids if _.split('.')[0] in extra_df.index]
    extra_df = extra_df.reindex([_.split('.')[0] for _ in ngids])
    extra_df.index = ngids
    # 1551
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


bacids = list(df.index)
sub_NCBI_df = sub_NCBI_df.loc[bacids,:]


bin_cols = []
for i in df.columns:
    if len(set(df[i].unique()))==2:
        bin_cols.append(i)
from sklearn import metrics

sub_NCBI_df.loc[:,'true_label'] = [genome2oxy_bin[_] for _ in sub_NCBI_df.index]
y_test = sub_NCBI_df['true_label']

bin2accu = {}
for bin_col in bin_cols:
    predict_r = df[bin_col]
    accuracy = metrics.balanced_accuracy_score(y_test, 
                                                   predict_r)
    bin2accu[bin_col] = accuracy
    
best_one = sorted(bin2accu.items(),key=lambda x:x[1])

#  ('Resource Acquisition:Substrate assimilation:S compounds:assimilatory sulfate reduction',
#   0.7917919012649595),
#  ('Resource Use:Chemotrophy:chemoorganoheterotrophy:aerobic respiration:electron transport chain: ETC complex IV',
#   0.8363499139228716)]


