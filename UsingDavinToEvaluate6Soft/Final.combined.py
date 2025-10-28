



import os
import pandas as pd
os.chdir('/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/microtraits/')
microtraitsdf = pd.read_csv('bac_granularity3.csv',index_col=1)
target_col = 'Resource Use:Chemotrophy:chemoorganoheterotrophy:aerobic respiration:electron transport chain: ETC complex IV'
microtraitsbin_predicted = microtraitsdf[target_col]



##### ASE for the other software
##### merged the results of ASE together with the TOP40 results.

import pandas as pd
predicted_df = pd.read_csv('/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/traitar/phenotype_prediction/predictions_majority-vote_combined.txt',sep='\t',index_col=0)
aerobe_cols = ["Aerobe",
               "Anaerobe",
               "Capnophilic",
               "Facultative"]
sub_df = predicted_df.loc[:,aerobe_cols]
classified_aerobe = sub_df.index[(sub_df.loc[:,["Aerobe","Facultative"]]>0).any(1) & (sub_df.loc[:,["Anaerobe"]]==0).all(1)]
classified_anaerobe = sub_df.index[(sub_df.loc[:,["Anaerobe"]]>0).any(1) & (sub_df.loc[:,["Aerobe","Facultative"]]==0).all(1)]

identified_genomes = sub_df.index[(sub_df.iloc[:,:4]!=0).any(1)]
print("Unclassified genomes :  ",sub_df.shape[0]-len(identified_genomes))
# Unclassified genomes :   55

bacdiv_prediction = pd.read_csv('/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/bacdivAI/predicted_results.tsv',sep='\t',index_col=0)
davin_prediction = pd.read_csv('/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/davinPredictor/predicted_results.tsv',sep='\t')
davin_prediction.index = [_.split('/')[-1].replace('.faa','') for _ in davin_prediction['node']]

bayesian_prediction = pd.read_csv('/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/bayesian/predicted_results.tsv',sep='\t',index_col=0)

top40_prediction = pd.read_csv('/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/annotop40/top40.prediction',sep='\t',index_col=0)
####
gids = sorted(list(bayesian_prediction.index))
merged_df = pd.DataFrame(index=gids)

merged_df.loc[:,'bayesian:non-Anaerobe:prob'] = bayesian_prediction.reindex(gids)[['Aerobe','Facultative']].max(1)
merged_df.loc[:,'bayesian:non-Anaerobe:label'] = (bayesian_prediction.reindex(gids)[['Aerobe','Facultative']].max(1)>bayesian_prediction.reindex(gids)['Anerobe']).astype(int)

merged_df.loc[:,'Davin:non-Anaerobe:prob'] = davin_prediction.reindex(gids)['probability']
merged_df.loc[:,'Davin:non-Anaerobe:label'] = davin_prediction.reindex(gids)['prediction']

merged_df.loc[:,'microtraits:non-Anaerobe:prob'] = 'NA'
merged_df.loc[:,'microtraits:non-Anaerobe:label'] = microtraitsbin_predicted.reindex(gids)


merged_df.loc[:,'Traitar:non-Anaerobe:prob'] = 'NA'
merged_df.loc[:,'Traitar:non-Anaerobe:label'] = (sub_df.reindex(gids).loc[:,["Aerobe","Capnophilic","Facultative"]]>0).any(1).astype(int)

merged_df.loc[:,'bacdivAI:non-Anaerobe:prob'] = bacdiv_prediction.reindex(gids)['BacDive_ai:Aerobe']
merged_df.loc[:,'bacdivAI:non-Anaerobe:label'] = (bacdiv_prediction.reindex(gids)['BacDive_ai:Aerobe']>bacdiv_prediction.reindex(gids)['BacDive_ai:Anaerobe']).astype(int)
merged_df.loc[:,'top40:non-Anaerobe:prob'] = top40_prediction.reindex(gids)['LR prob']
merged_df.loc[:,'top40:non-Anaerobe:label'] = top40_prediction.reindex(gids)['LR']
merged_df.to_csv('/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/MergedAllprediction.tsv',sep='\t',index=1)

import itertools
labels = [_ for _ in merged_df.columns
         if ':label' in _]


from sklearn.metrics import jaccard_score
from collections import defaultdict
s2s_acc = defaultdict(dict)
for k,v in list(itertools.product(labels,labels)):
    #print(k,v)
    s1,s2 = k.split(':')[0],v.split(':')[0]
    v1,v2 = merged_df[k],merged_df[v]
    s2s_acc[s1][s2] = jaccard_score(v1, v2)
crossdf = pd.DataFrame.from_dict(s2s_acc)
crossdf.to_csv('/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/jaccard_compare.tsv',sep='\t',index=1)



# Ancestral reconstruction
from ete3 import Tree
from tqdm import tqdm
leafs = Tree('/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/reftree.newick',3).get_leaf_names()
bin_df = merged_df.reindex(leafs).reindex(columns=labels)
odir = '/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/'
os.system(f"mkdir -p {odir}/ACE_DiffSofts/")
for colname,col in tqdm(bin_df.iteritems()):
    with open(f"{odir}/ACE_DiffSofts/{colname.split(':')[0]}.tab",'w') as f1:
        f1.write(f"Gid\tstats\n")
        for k,v in col.to_dict().items():
            f1.write(f"{k}\t{v}\n") 
    cmd = f"/mnt/home-user/thliao/anaconda3/envs/r_env/bin/Rscript /mnt/storage3/thliao/project/ML_oxygen/testing_sets/ace.r {odir}/ACE_DiffSofts/{colname.split(':')[0]}.tab /mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/reftree.newick {odir}/ACE_DiffSofts/{colname.split(':')[0]}.anc"
    os.system(cmd)






##### pack up used protein files
import tarfile

def make_tar_gz(source_dir, output_filename):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=".")
        
make_tar_gz('/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/prepared_proteins', "1007proteins.tar.gz")

