

import pandas as pd
output_dir = '/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/'
anno_file = f"{output_dir}/pfam_anno.tab"
d = pd.read_csv(anno_file,sep='\t',index_col=0)
d = d.applymap(lambda x: 0 if pd.isna(x) else 1)
d.columns = [_.split('.')[0] for _ in d.columns]

pfam_anno = pd.read_csv('/mnt/storage4/thliao/ML_oxygen/bayesian/Additional_file_6_new.csv',sep=',',index_col=0)
id2name = pd.read_csv('/mnt/maple/thliao/data/protein_db/Pfam.v33.1/Pfam-A.clans.tsv',sep='\t',header=None)
subid2name = id2name.loc[id2name[3].isin(pfam_anno.columns)]
missing_p = set(pfam_anno.columns).difference(set(subid2name[3]))
print(len(missing_p))
# 1084

import numpy as np
likelihood = open('/mnt/storage4/thliao/ML_oxygen/bayesian/likelihoods.txt').readlines()
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
newd = newd.reindex(columns=[_ for _ in pfam_anno.columns if _ in fams_used]).fillna(0)


newd.to_csv('/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/bayesian/Test-matrix.txt',sep='\t',index=1)

output = open('/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/bayesian/predictions_.txt','w')
output_true = open('/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/bayesian/predictions.txt','w')
output1 = open('/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/bayesian/All_predictions.txt','a')
parse_results = open('/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/bayesian/Test-matrix.txt').readlines() 



predicted_r = pd.read_csv('/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/bayesian/All_predictions.txt',sep='\t',header=None)
predicted_r.columns = ['Genome','Aerobe','Anerobe','Facultative','']
predicted_r = predicted_r.iloc[1:,:-1]
predicted_r.to_csv('/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/bayesian/predicted_results.tsv',sep='\t',index=0)




