



# cd /mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/
# ls prepared_proteins/*faa |parallel /home-user/thliao/software/interproscan-5.63-95.0/bin/hmmer/hmmer3/3.3/hmmscan --domtblout bacdivAI/hmmsearch/{/.}.domtblout -o /dev/null --cpu 1 /home-user/thliao/software/interproscan-5.63-95.0/data/pfam/35.0/pfam_a.hmm {}



#### special environment setting....
# conda activate soft

from glob import glob
import pickle
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
from os.path import *

anno_files = glob('/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/bacdivAI/hmmsearch/*.domtblout')

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
import io
from collections import defaultdict
gid2pred = defaultdict(dict)
gid2pfams = {}
for filename in tqdm(anno_files):
    gid = filename.split('/')[-1].replace('.domtblout','')
    hmmer_colnames = ['target name','target accession','tlen','query name','accession','qlen','E-value','score_overall','bias_overall','#','of','c-Evalue','i-Evalue','score_domain','bias_domain','from_hmm','to_hmm','ali_from','ali_to','env_from','env_to','acc','description of target']
    cleaned = "\n".join(filter(lambda x: not x.startswith("#"), 
                             ["\t".join(i.split(None, 22))
                              for i in open(filename).readlines()]))
    m = pd.read_csv(io.StringIO(cleaned), sep = "\t",  header = None)
    m.columns = hmmer_colnames
    m = m.loc[m['E-value']<=EVALUE]
    pfams = [_.split('.')[0] for _ in m['target accession']]
    pfams = set(pfams)
    gid2pfams[gid] = pfams
    gid2pred[gid].update(predict(aerobic_model,pfams,'aerobic'))
    gid2pred[gid].update(predict(anaerobic_model,pfams,'anaerobic'))
    #print(gid2pred)
    #break
d = pd.DataFrame(index = list(gid2pred),columns=['BacDive_ai:Aerobe',
                                       'BacDive_ai:Anaerobe',])
for gid,pred in tqdm(gid2pred.items()):
    prob,label = pred['aerobic']
    d.loc[gid,'BacDive_ai:Aerobe'] = 100-prob if label == 0 else prob
    prob,label = pred['anaerobic']
    d.loc[gid,'BacDive_ai:Anaerobe'] = 100-prob if label == 0 else prob    

d.to_csv('/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/bacdivAI/predicted_results.tsv',sep='\t',index=1)

