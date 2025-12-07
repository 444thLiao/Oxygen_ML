


from os.path import exists,dirname
from glob import glob
import os
from tqdm import tqdm


hal_file = '/mnt/ivy/thliao/project/ML_oxygen/training_sets/extra_ValidateGBDT40/phyloglm_40.hal'
odir = f"/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/annoPhyloglm40"
indir = '/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/prepared_proteins'
os.chdir(indir)

from bin.multiple_sbatch import generate_sbatch_job_array

if not exists(odir):
    os.system(f"mkdir -p {odir}")
gids = []
for inprotein in tqdm(glob(f'{indir}/*.faa')):
    gid = inprotein.split('/')[-1].replace('.faa','')
    gids.append(gid)
    
gid = "{input}"
inprotein = indir+'/{input}.faa'
refcmd = f"/home-user/thliao/software/kofamscan/exec_annotation -p {hal_file} -k /mnt/home-db/pub/protein_db/kegg/v20230301/ko_list --tmp-dir {odir}/.{gid} -o {odir}/{gid}.kofamout -f mapper-one-line --no-report-unannotated {inprotein} && rm -rf {odir}/.{gid}"
s = generate_sbatch_job_array(f'{odir}/anno.sbatch',
                    inputs=gids,
                    command_template=refcmd,
                    log_dir=f'{odir}/logs/',
                    percpu=5,
                    jobname=f'anno',)
os.system(f"sbatch {s}")



#! parse data
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from glob import glob
def parse_o(inf):
    l2ko = {}
    for row in open(inf).read().strip().split('\n'):
        rows = row.split('\t')
        l2ko[rows[0]] = ';'.join(sorted(rows[1:]))
    return l2ko

dfs = []
for kofamout in tqdm(glob(f'{odir}/*.kofamout')):
    genome = kofamout.split('/')[-1].replace('.kofamout','')
    gid2kegg2locus_info = defaultdict(lambda :defaultdict(list))
    l2ko = parse_o(kofamout)
    for locus,ko_l in l2ko.items():
        if not locus and not ko_l:
            gid2kegg2locus_info[genome] = {}
        else:
            for ko in ko_l.split(';'):
                gid2kegg2locus_info[genome][ko].append(locus)
    gid2kegg2locus_info = {genome:{ko:','.join(list(set(l_list))) for ko,l_list in _d.items()} for genome,_d in gid2kegg2locus_info.items()}
    sub_df = pd.DataFrame.from_dict(gid2kegg2locus_info, orient='index')
    if sub_df.shape==(0,0):
        sub_df = pd.DataFrame(index=[genome],)
    sub_df.to_csv(kofamout.replace('.kofamout','_anno.tab'),sep='\t',index=1)
    
    dfs.append(sub_df)
final_df = pd.concat(dfs,axis=0)

#final_df.to_csv(f"{dirname(odir)}/Top40_details.tab",sep='\t',index=1)
bin_df = final_df.applymap(lambda x: 0 if pd.isna(x) else 1)
#bin_df.to_csv(f"{dirname(odir)}/Top40_bin.tab",sep='\t',index=1)


# prediction
import pickle
LR_model = pickle.load(open("/mnt/ivy/thliao/project/ML_oxygen/training_sets/extra_ValidateGBDT40/phyloglm_40.model",'rb'))

top40 = open('/mnt/ivy/thliao/project/ML_oxygen/training_sets/extra_ValidateGBDT40/phyloglm_40.list').read().strip().split('\n')
extant_predicted = bin_df.reindex(columns=top40)
extant_predicted.loc[:,'LR'] = LR_model.predict(extant_predicted.loc[:,top40])
extant_predicted.loc[:,'LR prob'] = LR_model.predict_proba(extant_predicted.loc[:,top40])[:,LR_model.classes_==1].reshape(-1)
extant_predicted.to_csv('/mnt/ivy/thliao/project/ML_oxygen/training_sets/extra_ValidateGBDT40/phyloglm_40.prediction',sep='\t',index=1)


####################################################################################






hal_file = '/mnt/ivy/thliao/project/ML_oxygen/training_sets/extra_ValidateGBDT40/GBDT6_phyloglm34.hal'
odir = f"/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/annoGBDT6_phyloglm34"
indir = '/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/prepared_proteins'
os.chdir(indir)

if not exists(odir):
    os.system(f"mkdir -p {odir}")
gids = []
for inprotein in tqdm(glob(f'{indir}/*.faa')):
    gid = inprotein.split('/')[-1].replace('.faa','')
    gids.append(gid)
    
gid = "{input}"
inprotein = indir+'/{input}.faa'
refcmd = f"/home-user/thliao/software/kofamscan/exec_annotation -p {hal_file} -k /mnt/home-db/pub/protein_db/kegg/v20230301/ko_list --tmp-dir {odir}/.{gid} -o {odir}/{gid}.kofamout -f mapper-one-line --no-report-unannotated {inprotein} && rm -rf {odir}/.{gid}"
s = generate_sbatch_job_array(f'{odir}/anno.sbatch',
                    inputs=gids,
                    command_template=refcmd,
                    log_dir=f'{odir}/logs/',
                    percpu=5,
                    jobname=f'anno',)
os.system(f"sbatch {s}")


#! parse data
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from glob import glob
def parse_o(inf):
    l2ko = {}
    for row in open(inf).read().strip().split('\n'):
        rows = row.split('\t')
        l2ko[rows[0]] = ';'.join(sorted(rows[1:]))
    return l2ko

dfs = []
for kofamout in tqdm(glob(f'{odir}/*.kofamout')):
    genome = kofamout.split('/')[-1].replace('.kofamout','')
    gid2kegg2locus_info = defaultdict(lambda :defaultdict(list))
    l2ko = parse_o(kofamout)
    for locus,ko_l in l2ko.items():
        if not locus and not ko_l:
            gid2kegg2locus_info[genome] = {}
        else:
            for ko in ko_l.split(';'):
                gid2kegg2locus_info[genome][ko].append(locus)
    gid2kegg2locus_info = {genome:{ko:','.join(list(set(l_list))) for ko,l_list in _d.items()} for genome,_d in gid2kegg2locus_info.items()}
    sub_df = pd.DataFrame.from_dict(gid2kegg2locus_info, orient='index')
    if sub_df.shape==(0,0):
        sub_df = pd.DataFrame(index=[genome],)
    sub_df.to_csv(kofamout.replace('.kofamout','_anno.tab'),sep='\t',index=1)
    
    dfs.append(sub_df)
final_df = pd.concat(dfs,axis=0)

#final_df.to_csv(f"{dirname(odir)}/Top40_details.tab",sep='\t',index=1)
bin_df = final_df.applymap(lambda x: 0 if pd.isna(x) else 1)
#bin_df.to_csv(f"{dirname(odir)}/Top40_bin.tab",sep='\t',index=1)


# prediction
import pickle
LR_model = pickle.load(open("/mnt/ivy/thliao/project/ML_oxygen/training_sets/extra_ValidateGBDT40/GBDT6_phyloglm34.model",'rb'))

top40 = open('/mnt/ivy/thliao/project/ML_oxygen/training_sets/extra_ValidateGBDT40/GBDT6_phyloglm34.list').read().strip().split('\n')
extant_predicted = bin_df.reindex(columns=top40)
extant_predicted.loc[:,'LR'] = LR_model.predict(extant_predicted.loc[:,top40])
extant_predicted.loc[:,'LR prob'] = LR_model.predict_proba(extant_predicted.loc[:,top40])[:,LR_model.classes_==1].reshape(-1)
extant_predicted.to_csv('/mnt/ivy/thliao/project/ML_oxygen/training_sets/extra_ValidateGBDT40/GBDT6_phyloglm34.prediction',sep='\t',index=1)


####################################################################################


hal_file = '/mnt/ivy/thliao/project/ML_oxygen/training_sets/extra_ValidateGBDT40/GBDT6.hal'
odir = f"/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/annoGBDT6"
indir = '/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/prepared_proteins'
os.chdir(indir)

if not exists(odir):
    os.system(f"mkdir -p {odir}")
gids = []
for inprotein in tqdm(glob(f'{indir}/*.faa')):
    gid = inprotein.split('/')[-1].replace('.faa','')
    gids.append(gid)
    
gid = "{input}"
inprotein = indir+'/{input}.faa'
refcmd = f"/home-user/thliao/software/kofamscan/exec_annotation -p {hal_file} -k /mnt/home-db/pub/protein_db/kegg/v20230301/ko_list --tmp-dir {odir}/.{gid} -o {odir}/{gid}.kofamout -f mapper-one-line --no-report-unannotated {inprotein} && rm -rf {odir}/.{gid}"
s = generate_sbatch_job_array(f'{odir}/anno.sbatch',
                    inputs=gids,
                    command_template=refcmd,
                    log_dir=f'{odir}/logs/',
                    percpu=5,
                    jobname=f'anno',)
os.system(f"sbatch {s}")


#! parse data
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from glob import glob
def parse_o(inf):
    l2ko = {}
    for row in open(inf).read().strip().split('\n'):
        rows = row.split('\t')
        l2ko[rows[0]] = ';'.join(sorted(rows[1:]))
    return l2ko

dfs = []
for kofamout in tqdm(glob(f'{odir}/*.kofamout')):
    genome = kofamout.split('/')[-1].replace('.kofamout','')
    gid2kegg2locus_info = defaultdict(lambda :defaultdict(list))
    l2ko = parse_o(kofamout)
    for locus,ko_l in l2ko.items():
        if not locus and not ko_l:
            gid2kegg2locus_info[genome] = {}
        else:
            for ko in ko_l.split(';'):
                gid2kegg2locus_info[genome][ko].append(locus)
    gid2kegg2locus_info = {genome:{ko:','.join(list(set(l_list))) for ko,l_list in _d.items()} for genome,_d in gid2kegg2locus_info.items()}
    sub_df = pd.DataFrame.from_dict(gid2kegg2locus_info, orient='index')
    if sub_df.shape==(0,0):
        sub_df = pd.DataFrame(index=[genome],)
    sub_df.to_csv(kofamout.replace('.kofamout','_anno.tab'),sep='\t',index=1)
    dfs.append(sub_df)
final_df = pd.concat(dfs,axis=0)

#final_df.to_csv(f"{dirname(odir)}/Top40_details.tab",sep='\t',index=1)
bin_df = final_df.applymap(lambda x: 0 if pd.isna(x) else 1)
#bin_df.to_csv(f"{dirname(odir)}/Top40_bin.tab",sep='\t',index=1)


# prediction
import pickle
LR_model = pickle.load(open("/mnt/ivy/thliao/project/ML_oxygen/training_sets/extra_ValidateGBDT40/GBDT6.model",'rb'))

top40 = open('/mnt/ivy/thliao/project/ML_oxygen/training_sets/extra_ValidateGBDT40/GBDT6.list').read().strip().split('\n')
extant_predicted = bin_df.reindex(columns=top40)
extant_predicted.loc[:,'LR'] = LR_model.predict(extant_predicted.loc[:,top40])
extant_predicted.loc[:,'LR prob'] = LR_model.predict_proba(extant_predicted.loc[:,top40])[:,LR_model.classes_==1].reshape(-1)
extant_predicted.to_csv('/mnt/ivy/thliao/project/ML_oxygen/training_sets/extra_ValidateGBDT40/GBDT6.prediction',sep='\t',index=1)


