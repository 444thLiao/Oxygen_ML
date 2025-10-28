
"""
# example commands 
docker run -v /mnt:/mnt -it 02a16167973c bash

docker run -v $PWD:/mnt 02a16167973c bash -c 'traitar phenotype /home/traitar/traitar/traitar/data/sample_data/ /home/traitar/traitar/traitar/data/sample_data/samples.txt from_genes /mnt/traitar_out'

"""
from ..load_data import *

header = open("/home-user/thliao/software/traitar/traitar/data/sample_data/samples.txt").read().strip().split('\n')

for _ in sub_NCBI_df.index:
    cmd = f"ln -sf /mnt/maple/thliao/data/NCBI/modified_data/prokka_o/{_}/{_}.faa ./completeness_bac/"
    os.system(cmd)
sample_df = sub_NCBI_df.sample(100,random_state=42)
final_t = [header[0]]
for genome in sample_df.index:
    final_t.append('\t'.join([genome+'.faa',
                              genome,
                              'set1']))
with open('./comparative/traitar/samples.txt','w') as f1:
    f1.write('\n'.join(final_t))
    
# see pfam annotation part within the `annotations_distance.py`

done_sample = []
target_dir = "/mnt/maple/thliao/data/NCBI/modified_data/annotations/Pfam"
for ofile in tqdm(glob(join(target_dir, f"*_domtblout.dat"))):
    if open(ofile).readlines()[-1].strip()=="# [ok]":
        done_sample.append(ofile)

with open("/mnt/ivy/thliao/project/ML_oxygen/comparative/traitar/sample_9328.txt",'w') as f1:
    header= f"sample_file_name\tsample_name\tcategory\n"
    for _ in done_sample:
        genome = _.split('/')[-1].split("_domtblout")[0]
        t = genome2oxy_bin[genome]
        cat = 'aerobe' if t==1 else "anaerobe"
        header += f"{genome}.faa\t{genome}\t{cat}\n"
    f1.write(header)    
for _ in tqdm(done_sample): 
    if not exists(f"/mnt/ivy/thliao/project/ML_oxygen/comparative/traitar/full_o/annotation/pfam/{genome}_domtblout.dat"):
        os.system(f"ln -s {_} /mnt/ivy/thliao/project/ML_oxygen/comparative/traitar/full_o/annotation/pfam/{genome}_domtblout.dat")
            
                 
import random
samples = random.choices(done_sample,k=5)

    
    
# docker run -v /mnt:/mnt -it ac4f53413a3e bash
# traitar pfam --local /mnt/ivy/thliao/db/protein_db/Pfam.v33.1/

f"""
docker run -v /mnt/:/mnt ac4f53413a3e bash -c 'traitar phenotype /mnt/ivy/thliao/project/ML_oxygen/completeness_bac /mnt/ivy/thliao/project/ML_oxygen/comparative/traitar/sample_9328.txt from_genes /mnt/ivy/thliao/project/ML_oxygen/comparative/traitar/full_o/ --no_heatmap_phenotype_clustering'

traitar phenotype /mnt/ivy/thliao/project/ML_oxygen/completeness_bac /mnt/ivy/thliao/project/ML_oxygen/comparative/traitar/sample5.txt from_genes /mnt/ivy/thliao/project/ML_oxygen/comparative/traitar/test5/ --no_heatmap_phenotype_clustering

predict.py /home-user/thliao/anaconda3/envs/py2/lib/python2.7/site-packages/traitar-1.1.2-py2.7.egg/traitar/data/models/phypat.tar.gz /mnt/ivy/thliao/project/ML_oxygen/comparative/traitar/full_o/phenotype_prediction/phypat  /mnt/ivy/thliao/project/ML_oxygen/comparative/traitar/full_o//annotation/pfam/summary.dat -k 5  
predict.py /home-user/thliao/anaconda3/envs/py2/lib/python2.7/site-packages/traitar-1.1.2-py2.7.egg/traitar/data/models/phypat+PGL.tar.gz /mnt/ivy/thliao/project/ML_oxygen/comparative/traitar/full_o/phenotype_prediction/phypat+PGL /mnt/ivy/thliao/project/ML_oxygen/comparative/traitar/full_o//annotation/pfam/summary.dat -k 5

merge_preds.py /mnt/ivy/thliao/project/ML_oxygen/comparative/traitar/full_o/phenotype_prediction /mnt/ivy/thliao/project/ML_oxygen/comparative/traitar/full_o/phenotype_prediction/phypat /mnt/ivy/thliao/project/ML_oxygen/comparative/traitar/full_o/phenotype_prediction/phypat+PGL phypat phypat+PGL -k 5

heatmap.py /mnt/ivy/thliao/project/ML_oxygen/comparative/traitar/test5/phenotype_prediction/phypat/predictions_majority-vote.txt /mnt/ivy/thliao/project/ML_oxygen/comparative/traitar/test5/phenotype_prediction/heatmap_phypat.pdf  --sample_f /mnt/ivy/thliao/project/ML_oxygen/comparative/traitar/sample5.txt /usr/local/lib/python2.7/dist-packages/traitar-1.1.2-py2.7.egg/traitar/data/models/phypat.tar.gz /usr/local/lib/python2.7/dist-packages/traitar-1.1.2-py2.7.egg/traitar/data/colors.txt --column_method None

"""
# ['heatmap.py /mnt/ivy/thliao/project/ML_oxygen/comparative/traitar/test5/phenotype_prediction/phypat/predictions_majority-vote.txt /mnt/ivy/thliao/project/ML_oxygen/comparative/traitar/test5/phenotype_prediction/heatmap_phypat.pdf  --sample_f /mnt/ivy/thliao/project/ML_oxygen/comparative/traitar/sample5.txt /usr/local/lib/python2.7/dist-packages/traitar-1.1.2-py2.7.egg/traitar/data/models/phypat.tar.gz /usr/local/lib/python2.7/dist-packages/traitar-1.1.2-py2.7.egg/traitar/data/colors.txt  ', 
#  'heatmap.py /mnt/ivy/thliao/project/ML_oxygen/comparative/traitar/test5/phenotype_prediction/phypat+PGL/predictions_majority-vote.txt /mnt/ivy/thliao/project/ML_oxygen/comparative/traitar/test5/phenotype_prediction/heatmap_phypat+PGL.pdf  --sample_f /mnt/ivy/thliao/project/ML_oxygen/comparative/traitar/sample5.txt /usr/local/lib/python2.7/dist-packages/traitar-1.1.2-py2.7.egg/traitar/data/models/phypat+PGL.tar.gz /usr/local/lib/python2.7/dist-packages/traitar-1.1.2-py2.7.egg/traitar/data/colors.txt  --column_method None', 
#  'heatmap.py /mnt/ivy/thliao/project/ML_oxygen/comparative/traitar/test5/phenotype_prediction/predictions_majority-vote_combined.txt /mnt/ivy/thliao/project/ML_oxygen/comparative/traitar/test5/phenotype_prediction/heatmap_combined.pdf --secondary_model_tar /usr/local/lib/python2.7/dist-packages/traitar-1.1.2-py2.7.egg/traitar/data/models/phypat+PGL.tar.gz --sample_f /mnt/ivy/thliao/project/ML_oxygen/comparative/traitar/sample5.txt /usr/local/lib/python2.7/dist-packages/traitar-1.1.2-py2.7.egg/traitar/data/models/phypat.tar.gz /usr/local/lib/python2.7/dist-packages/traitar-1.1.2-py2.7.egg/traitar/data/colors.txt  --column_method None']


## modified a little bit for the original docker image (02a16167973c) to modified one ()




"""
evaluating the performance of traitar 

"""


import pandas as pd
import os
os.chdir('/mnt/ivy/thliao/project/ML_oxygen/comparative/traitar/')

predicted_df = pd.read_csv('./full_o/phenotype_prediction/predictions_majority-vote_combined.txt',sep='\t',index_col=0)
bac_ids = open('/mnt/ivy/thliao/project/ML_oxygen/training_sets/8943_bac.ids').read().strip().split('\n')
predicted_df = predicted_df.loc[bac_ids,:]
aerobe_cols = ["Aerobe",
               "Anaerobe",
               "Capnophilic",
               "Facultative"]

sub_df = predicted_df.loc[:,aerobe_cols]
true_labels = pd.read_csv('sample_9328.txt',sep='\t',index_col=0)
true_labels = true_labels.set_index('sample_name')['category']
true_labels = true_labels.reindex(bac_ids)
sub_df.loc[bac_ids,'true category'] = true_labels

identified_genomes = sub_df.index[(sub_df.iloc[:,:4]!=0).any(1)]
print("Unclassified genomes :  ",sub_df.shape[0]-len(identified_genomes))
# Unclassified genomes :   263

classified_aerobe = sub_df.loc[(sub_df.loc[:,["Aerobe","Facultative"]]>0).any(1) & (sub_df.loc[:,["Anaerobe"]]==0).all(1),
                               'true category']
classified_anaerobe = sub_df.loc[(sub_df.loc[:,["Anaerobe"]]>0).any(1) & (sub_df.loc[:,["Aerobe","Facultative"]]==0).all(1),
                                 'true category']

classfied_genomes = set(classified_aerobe.index).union(set(classified_anaerobe.index))
print("Clearly classified genomes: " ,len(classfied_genomes))
ambiguous_genomes = set(identified_genomes).difference(classfied_genomes)
print("Ambiguous classified genomes: " ,len(ambiguous_genomes))
# Clearly classified genomes:  8202
# Ambiguous classified genomes:  478

from collections import Counter
a = Counter(classified_aerobe)
# {'aerobe': 6497, 'anaerobe': 251}
b = Counter(classified_anaerobe)
# {'anaerobe': 1388, 'aerobe': 66}

print( (a['aerobe'] + b['anaerobe'])*100 /sub_df.shape[0])
# 88.16951805881






