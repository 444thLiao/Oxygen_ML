

from tqdm import tqdm
from glob import glob
import os

header = open("/home-user/thliao/software/traitar/traitar/data/sample_data/samples.txt").read().strip().split('\n')


done_sample = []
for ofile in tqdm(glob("/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/annoPfamv33.1_domtbl/*_domtblout.dat")):
    if open(ofile).readlines()[-1].strip()=="# [ok]":
        done_sample.append(ofile)

with open("/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/traitar/sample_1007.txt",'w') as f1:
    header= f"sample_file_name\tsample_name\tcategory\n"
    for _ in done_sample:
        genome = _.split('/')[-1].split("_domtblout")[0]
        cat = 'aerobe'
        header += f"{genome}.faa\t{genome}\t{cat}\n"
    f1.write(header)    
    
for _ in tqdm(done_sample): 
    genome = _.split('/')[-1].split("_domtblout")[0]
    os.system(f"ln -sf {_} /mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/traitar/annotation/pfam/{genome}_domtblout.dat")
            

# docker run -v /mnt/:/mnt ac4f53413a3e bash -c 'traitar phenotype /mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/prepared_proteins /mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/traitar/sample_1007.txt from_genes /mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/traitar/  --no_heatmap_phenotype_clustering'

# docker run -v /mnt:/mnt /home-user/thliao/:/home-user/thliao/ ac4f53413a3e bash -c 'predict.py /home-user/thliao/anaconda3/envs/py2/lib/python2.7/site-packages/traitar-1.1.2-py2.7.egg/traitar/data/models/phypat+PGL.tar.gz /mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/traitar/phenotype_prediction/phypat_PGL /mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/traitar/annotation/pfam/summary.dat -k 5'


import pandas as pd
predicted_df = pd.read_csv('/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/traitar/phenotype_prediction/predictions_majority-vote_combined.txt',sep='\t',index_col=0)
aerobe_cols = ["Aerobe",
               "Anaerobe",
               "Capnophilic",
               "Facultative"]
sub_df = predicted_df.loc[:,aerobe_cols]

true_labels = pd.read_csv('sample_9328.txt',sep='\t',index_col=0)
true_labels = true_labels.set_index('sample_name')['category']
true_labels = true_labels.reindex(bac_ids)
sub_df.loc[bac_ids,'true category'] = true_labels



