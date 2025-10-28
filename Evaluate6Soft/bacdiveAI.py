

from ..load_data import *


# conda activate soft

gids = list(sub_NCBI_df.index)
runned_gids = []
for gid in tqdm(gids):
    if not exists(f"/mnt/maple/thliao/data/NCBI/modified_data/prokka_o/{gid}/{gid}.faa"):
        print(gid)
    if not exists(f"/mnt/storage3/thliao/project/ML_oxygen/comparison_softwares/bacdive-AI/training_sets/annotations/{gid}.faa.tsv"):
        runned_gids.append(gid)

from bin.multiple_sbatch import generate_sbatch_job_array,batch_iter



ingids_list = batch_iter(gids,batch_size=2000)
cmd = "/home-user/thliao/software/interproscan-5.63-95.0/interproscan.sh  -i /mnt/maple/thliao/data/NCBI/modified_data/prokka_o/{input}/{input}.faa -f tsv -d /mnt/storage3/thliao/project/ML_oxygen/comparison_softwares/bacdive-AI/training_sets/annotations/ -appl Pfam"
for _idx,a in enumerate(ingids_list):
    s = generate_sbatch_job_array(f'/home-user/thliao/project/ML_oxygen/comparison_softwares/bacdive-AI/batch_ipr_pfam_{_idx}.sbatch',
                          inputs=a,
                          command_template=cmd,
                          log_dir='/home-user/thliao/project/ML_oxygen/comparison_softwares/bacdive-AI/logs/',
                          percpu=2,
                          jobname=f'{_idx}train_ipr',)
    os.system(f"sbatch {s}")



db = "/home-user/thliao/software/interproscan-5.63-95.0/data/pfam/35.0/pfam_a.hmm"
software_dir = '/home-user/thliao/software/interproscan-5.63-95.0/bin/hmmer/hmmer3/3.3'

input = "{input}"
cmd = f"{software_dir}/hmmscan --cpu 8 --domtblout /mnt/storage3/thliao/project/ML_oxygen/comparison_softwares/bacdive-AI/training_sets/annotations/{input}.domtblout {db} /mnt/maple/thliao/data/NCBI/modified_data/prokka_o/{input}/{input}.faa > /mnt/storage3/thliao/project/ML_oxygen/comparison_softwares/bacdive-AI/training_sets/annotations/{input}.log "
for _idx,a in enumerate(ingids_list):
    s = generate_sbatch_job_array(f'/home-user/thliao/project/ML_oxygen/comparison_softwares/bacdive-AI/batch_ipr_pfam_{_idx}.sbatch',
                          inputs=a,
                          command_template=cmd,
                          log_dir='/home-user/thliao/project/ML_oxygen/comparison_softwares/bacdive-AI/logs/',
                          percpu=2,
                          jobname=f'{_idx}train_ipr',)
    os.system(f"sbatch {s}")

# /home-user/thliao/software/interproscan-5.63-95.0/bin/hmmer/hmmer3/3.3/hmmscan -E 1e-20 --domE 1e-20 --cpu 40 --domtblout /mnt/storage3/thliao/project/ML_oxygen/comparison_softwares/bacdive-AI/training_sets/annotations/GCA_001263205.1.domtblout /home-user/thliao/software/interproscan-5.63-95.0/data/pfam/35.0/pfam_a.hmm /mnt/maple/thliao/data/NCBI/modified_data/prokka_o/GCA_001263205.1/GCA_001263205.1.faa > /mnt/storage3/thliao/project/ML_oxygen/comparison_softwares/bacdive-AI/training_sets/annotations/GCA_001263205.1.log

tsv_files = glob('/mnt/storage3/thliao/project/ML_oxygen/comparison_softwares/bacdive-AI/training_sets/annotations/*.tsv')
print(len(tsv_files))



import pickle
import os
import argparse
import numpy as np
import csv

def predict(dump,pfams,label):
    clf = dump.get('model')
    categories = dump.get("categories")
    strains = dump.get('strains')
    
    # transform sample
    lst = dict(zip(*np.unique(list(pfams), return_counts=True)))
    X = [lst.get(k) if k in lst else 0 for k in categories]

    # predict with probability
    proba = clf.predict_proba([X])
    y = proba.argmax(axis=1)
    y = y[0]
    proba = proba[0]
    true_index = list(clf.classes_).index(y)

    # print result 
    prob = round(proba[true_index]*100, 2)
    return {label:(prob,y)}
    
aerobic_model = pickle.load(open('/mnt/storage3/thliao/project/ML_oxygen/comparison_softwares/bacdive-AI/models/aerobic_data.p', "rb"))
anaerobic_model = pickle.load(open('/mnt/storage3/thliao/project/ML_oxygen/comparison_softwares/bacdive-AI/models/anaerobic_data.p', "rb"))


EVALUE = 1e-20

from collections import defaultdict
gid2pred = defaultdict(dict)
gid2pfams = {}
for gid in tqdm(list(gids)):
    filename = f"/mnt/storage3/thliao/project/ML_oxygen/comparison_softwares/bacdive-AI/training_sets/annotations/{gid}.faa.tsv"
    if not exists(filename) or gid in gid2pred:
        continue
    pfams = []
    with open(filename, 'r', encoding='utf-8') as f:
        csv_file = csv.reader(f, delimiter='\t')
        for line in csv_file:
            evalue = float(line[8])
            if evalue > EVALUE:
                continue
            pfams.append(line[4])
    pfams = set(pfams)
    gid2pfams[gid] = pfams

    gid2pred[gid].update(predict(aerobic_model,pfams,'aerobic'))
    gid2pred[gid].update(predict(anaerobic_model,pfams,'anaerobic'))



from api_tools.tk import read_hmmsearch
# ofile = '/mnt/storage3/thliao/project/ML_oxygen/comparison_softwares/bacdive-AI/training_sets/annotations/GCA_001263205.1.domtblout'
# gid = 'GCA_001263205.1'
# filtered_df = read_hmmsearch(ofile)
# filtered_df = filtered_df.loc[filtered_df['E-value']<=EVALUE]
# pfams = [_.split('.')[0] for _ in list(filtered_df['target accession'])]
# pfams = set(pfams)
# gid2pfams[gid] = pfams
# gid2pred[gid].update(predict(aerobic_model,pfams,'aerobic'))
# gid2pred[gid].update(predict(anaerobic_model,pfams,'anaerobic'))





d = pd.DataFrame(index = gids,columns=['BacDive_ai:Aerobe',
                                       'BacDive_ai:Anaerobe',])
for gid,pred in tqdm(gid2pred.items()):
    prob,label = pred['aerobic']
    d.loc[gid,'BacDive_ai:Aerobe'] = 100-prob if label == 0 else prob
    prob,label = pred['anaerobic']
    d.loc[gid,'BacDive_ai:Anaerobe'] = 100-prob if label == 0 else prob    

d.loc[:,'true_label'] = [genome2oxy_bin[_] for _ in d.index]
d.to_csv('/mnt/storage3/thliao/project/ML_oxygen/comparison_softwares/bacdive-AI/predicted_results.tsv',sep='\t',index=1)


from sklearn import metrics
subd = d.loc[~d['BacDive_ai:Aerobe'].isna(),:]
y_test = subd['true_label']
preds = subd['BacDive_ai:Aerobe']
accuracy = metrics.balanced_accuracy_score(y_test, preds > 50)
print(accuracy)
# 0.8443258946935417


# auc = metrics.roc_auc_score(y_test, preds)
# AP = metrics.average_precision_score(y_test, preds)
# print(accuracy,auc,AP)