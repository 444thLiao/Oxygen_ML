
import os
os.chdir('/mnt/storage4/thliao/ML_oxygen/bayesian')


tax_tab = "/home-user/thliao/.cache/ncbi-genome-download/taxonomy.tab"
tax_df = pd.read_csv(tax_tab, sep="\t", index_col=0)
genome2tax = tax_df.to_dict(orient="index")

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
aids = list(sub_NCBI_df.index)
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

specific_tax_df = pd.read_csv(
    "/home-user/thliao/project/ML_oxygen/taxdf.csv", sep="\t", index_col=0
)
bac_ids = list(
    specific_tax_df.index[specific_tax_df['superkingdom'] == 'Bacteria'])
print(len(bac_ids))

bac_ids = [_ for _ in bac_ids if _ in sub_NCBI_df.index]
print(len(bac_ids))

import pandas as pd
d = pd.read_csv('/mnt/storage3/thliao/project/ML_oxygen/training_sets/processed_data/pfam_anno.tab',sep='\t',index_col=0)
d.columns = [_.split('.')[0] for _ in d.columns]

pfam_anno = pd.read_csv('Additional_file_6_new.csv',sep=',',index_col=0)
id2name = pd.read_csv('/mnt/maple/thliao/data/protein_db/Pfam.v33.1/Pfam-A.clans.tsv',sep='\t',header=None)


subid2name = id2name.loc[id2name[3].isin(pfam_anno.columns)]
missing_p = set(pfam_anno.columns).difference(set(subid2name[3]))
print(len(missing_p))
# 1084


import numpy as np

likelihood = open('likelihoods.txt').readlines()
fams_used = [_.split('\t')[0] for _ in likelihood[1:]]
missing_fam = set(fams_used).intersection(set(missing_p))
print(len(missing_fam))
manual_curated_dict = {'Cna_B_2':'PF13715',
 'DUF1271':'PF06902',
 'DUF159':'PF02586',
 'DUF161':'PF02588',
 'DUF164':'PF02591',
 'DUF2029':'PF09594',
 'DUF204':'PF02659',
 'DUF208':'PF02677',
 'DUF2088':'PF09861',
 'DUF2360':'PF10152',
 'DUF255':'PF03190',
 'DUF318':'PF03773',
 'DUF3366':'PF11846',
 'DUF3448':None,
 'DUF4008':'PF13186',
 'DUF4098':'PF13351',
 'DUF519':'PF04378',
 'DUF521':'PF04412',
 'DUF552':'PF04472',
 'DUF59':'PF01883',
 'DUF721':'PF05258',
 'DUF814':'PF05670',
 'DUF869':'PF05911',
 'DisA_N':'PF10635',
 'ECH':None,
 'ECH_C':None,
 'Esterase_phd':'PF10503',
 'Mif2':None,
 'MttA_Hcf106':'PF02416',
 'Oxidored_q1':'PF00361',
 'Oxidored_q1_N':'PF00662',
 'PNPOx_C':'PF10590',
 'PepSY_TM_1':None,
 'PhnA_Zn_Ribbon':'PF08274',
 'Prenyltrans_1':'PF13243',
 'Pyridox_oxidase':'PF01243',
 'Repair_PSII':'PF04536',
 'Tic20':None,
 'Transaldolase':'PF00923',
 'UPF0051':'PF01458',
 'VPEP':None,
 'YWTD':None,
 'zf-MaoC':'PF13452'}


subid2name_dict = dict(zip(subid2name[0],subid2name[3]))
for k,v in manual_curated_dict.items():
    if v is not None:
        subid2name[v] = k
newd = d.reindex(columns=list(subid2name[0]))
newd.columns = [subid2name_dict[_] for _ in newd.columns]
newd = newd.reindex(columns=pfam_anno.columns).fillna(0)





newd.to_csv('./Test-matrix.txt',sep='\t',index=1)

## Run Additional_file_8.py


predicted_r = pd.read_csv('./All_predictions.txt',sep='\t',header=None)
predicted_r.columns = ['Genome','Aerobe','Anerobe','Facultative','']
predicted_r = predicted_r.set_index('Genome')
predicted_r = predicted_r.reindex([_ for _ in predicted_r.index if '_' in _])
predicted_r.loc[:,'prediction'] = [0 if row[:3].astype(float).argmax() ==1 else 1 
                                  for _,row in predicted_r.iterrows()]
predicted_r.to_csv('/mnt/storage4/thliao/ML_oxygen/bayesian/naivebayesian_8943bacprediction.tsv',sep='\t',index=1)
#predicted_r.loc[:,'true'] = [genome2oxy_bin[_] for _ in predicted_r['Genome']]


from sklearn import metrics
y_true = predicted_r.loc[bac_ids,'true']
preds = predicted_r.loc[bac_ids,'prediction']
accuracy = metrics.balanced_accuracy_score(y_true, 
                                           preds)
print(accuracy)
# 0.8880714564915462




