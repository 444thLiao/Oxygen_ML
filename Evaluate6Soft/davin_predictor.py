
### From 
# https://huggingface.co/wwood/aerobicity
###########################################

# conda activate /mnt/storage3/thliao/project/ML_oxygen/comparison_softwares/bacterial_dating_aerobic_predictor/env

# ┌──────────────────┬───────┐
# │ Oxygen tolerance ┆ count │
# │ ---              ┆ ---   │
# │ str              ┆ u32   │
# ╞══════════════════╪═══════╡
# │ anaerobe         ┆ 1055  │
# │ aerobe           ┆ 1959  │
# └──────────────────┴───────┘

# wget https://data.gtdb.ecogenomic.org/releases/release202/202.0/ar122_metadata_r202.tar.gz
# wget https://data.gtdb.ecogenomic.org/releases/release202/202.0/bac120_metadata_r202.tar.gz


# ./9_expand_incompletenss_and_contamination4.py --input-file data/all_gene_annotations.tsv --output-file data/all_gene_annotations.added_incompleteness_and_contamination.tsv
# ./10_split_training_test_sets.py --input-file data/all_gene_annotations.added_incompleteness_and_contamination.tsv --training-families data/training_families.txt --testing-families data/testing_families.txt --output-training data/all_gene_annotations.added_incompleteness_and_contamination.training.tsv --output-testing data/all_gene_annotations.added_incompleteness_and_contamination.testing.tsv


# ./11_generate_models.py --training-file data/all_gene_annotations.added_incompleteness_and_contamination.training.tsv --testing-file data/all_gene_annotations.added_incompleteness_and_contamination.testing.tsv -y data/bacdive_scrape_20230315.json.parsed.anaerobe_vs_aerobe.with_cyanos.csv --model-output-dir data/bacdive_scrape_20230315.json.parsed.anaerobe_vs_aerobe.with_cyanos.csv.models --cross-validation-data-output-dir data/bacdive_scrape_20230315.json.parsed.anaerobe_vs_aerobe.with_cyanos.csv.cv_data




#### For our datasets
gids = list(sub_NCBI_df.index)
runned_gids = []
for gid in tqdm(gids):
    if not exists(f"/mnt/maple/thliao/data/NCBI/modified_data/prokka_o/{gid}/{gid}.faa"):
        print(gid)
    else:
        os.system(f"ln -sf /mnt/maple/thliao/data/NCBI/modified_data/prokka_o/{gid}/{gid}.faa /mnt/storage4/thliao/ML_oxygen/training_sets/proteins/")
        
# ls proteins/*faa |parallel hmmsearch --tblout davin_annotations/hmmsearch/{/}.hmmsearch_tblout.csv -o /dev/null --notextw --cpu 1 /mnt/storage4/thliao/ML_oxygen/aerobicity/data/kofam-2022-01-30-profiles.hmm {}
######### now we can pass the csv file to parameter "--kofam-tsv-file"



########## preparing file for this parameter "eggnog_annotation_file"
# conda activate /mnt/storage4/thliao/ML_oxygen/aerobicity/env
# ls proteins/*faa |parallel emapper.py -m diamond -i {} --target_orthologs one2one --query_cover 50.0 --evalue 0.0000001 --cpu 20 -o davin_annotations/{/}.egg --data_dir /mnt/storage4/thliao/ML_oxygen/aerobicity/eggNOG --override


# 18/09/2025 06:39:30 PM INFO: Reading or calculating annotations not from a table ..
# 18/09/2025 06:39:30 PM INFO: Created temp working directory /tmp/tmpmsfbv23p
# 18/09/2025 06:39:30 PM INFO: Reading in COG whitelist
# 18/09/2025 06:39:30 PM INFO: Found 2700 COGs in the whitelist
# 18/09/2025 06:39:30 PM INFO: Running eggnog-mapper
# 18/09/2025 09:06:21 PM INFO: Running hmmsearch
# 18/09/2025 09:52:15 PM INFO: Read in 151228 annotations
# 18/09/2025 09:52:21 PM INFO: Found 4633 unique HMM annotations
# 18/09/2025 09:52:21 PM INFO: Read in 4810 cog_annotations
# 18/09/2025 09:52:22 PM INFO: Loading model XGBoost.model
# 18/09/2025 09:52:22 PM INFO: Loaded model XGBoost.model
# 18/09/2025 09:52:22 PM INFO: Wrote predictions to predictions.csv


# ./17_apply_to_proteome.py --protein-fasta ./test.faa --eggnog-data-dir eggNOG/ --models XGBoost.model --output-annotations ./test.anno --working-directory ./work

bac_data_df.loc[:,'y'] = [genome2oxy_bin[_] for _ in bac_data_df.index]
anaerobic_id = bac_data_df.index[bac_data_df.y==0]

_anaerobic_id = []
for gid in tqdm(anaerobic_id):
    ofile = f"/mnt/storage4/thliao/ML_oxygen/training_sets/davin_annotations/{gid}.faa.egg.emapper.annotations"
    if not exists(ofile) or os.path.getsize(ofile)==0:
        _anaerobic_id.append(gid)
        
print(len(_anaerobic_id))

os.system(f"EGGNOG_DATA_DIR=/mnt/storage4/thliao/ML_oxygen/aerobicity/eggNOG && emapper.py -m diamond -i /mnt/maple/thliao/data/NCBI/modified_data/prokka_o/{gid}/{gid}.faa --target_orthologs one2one --query_cover 50.0 --evalue 0.0000001 --cpu 20 -o /mnt/storage4/thliao/ML_oxygen/training_sets/davin_annotations/{gid}.faa.egg --data_dir /mnt/storage4/thliao/ML_oxygen/aerobicity/eggNOG --override")
## EGGNOG_DATA_DIR=/mnt/storage4/thliao/ML_oxygen/aerobicity/eggNOG emapper.py -m diamond -i /mnt/storage4/thliao/ML_oxygen/aerobicity/test.faa --target_orthologs one2one --query_cover 50.0 --evalue 0.0000001 --cpu 30 -o eggnog_output


from bin.multiple_sbatch import generate_sbatch_job_array,batch_iter
from bin.multiple_sbatch import batch_iter
gid = '{input}'
refcmd = f"EGGNOG_DATA_DIR=/mnt/storage4/thliao/ML_oxygen/aerobicity/eggNOG && /mnt/storage4/thliao/ML_oxygen/aerobicity/env/bin/python /mnt/storage4/thliao/ML_oxygen/aerobicity/env/bin/emapper.py -m diamond -i /mnt/maple/thliao/data/NCBI/modified_data/prokka_o/{gid}/{gid}.faa --target_orthologs one2one --query_cover 50.0 --evalue 0.0000001 --cpu 10 -o /mnt/storage4/thliao/ML_oxygen/training_sets/davin_annotations/{gid}.faa.egg --data_dir /mnt/storage4/thliao/ML_oxygen/aerobicity/eggNOG --override"

ingids = []
for gid in bac_ids:
    ofile = f"/mnt/storage4/thliao/ML_oxygen/training_sets/davin_annotations/{gid}.faa.egg.emapper.annotations"
    if not exists(ofile) or os.path.getsize(ofile)==0:
        ingids.append(gid)
ingids_list = batch_iter(ingids,batch_size=2000)
for _idx,a in enumerate(ingids_list):
    s = generate_sbatch_job_array(f'/mnt/storage4/thliao/ML_oxygen/aerobicity/anno{_idx}.sbatch',
                          inputs=a,
                          command_template=refcmd,
                          log_dir='/mnt/storage4/thliao/ML_oxygen/aerobicity/logs/',
                          percpu=10,
                          jobname=f'{_idx}davin',)
    os.system(f"sbatch {s}")
indir = '/mnt/storage4/thliao/ML_oxygen/aerobicity/'
import os
gids = list(sub_NCBI_df.index)
# for gid in tqdm(gids):
#     os.system(f"conda run -p /mnt/storage4/thliao/ML_oxygen/aerobicity/env {indir}/17_apply_to_proteome.py --protein-fasta /mnt/maple/thliao/data/NCBI/modified_data/prokka_o/{gid}/{gid}.faa --eggnog-data-dir {indir}/eggNOG/ --models {indir}/XGBoost.model --output-annotations /mnt/storage4/thliao/ML_oxygen/aerobicity/annotations/{gid}.anno --working-directory /mnt/storage4/thliao/ML_oxygen/aerobicity/annotations/{gid}/work --threads 30")

from bin.multiple_sbatch import generate_sbatch_job_array,batch_iter
from bin.multiple_sbatch import batch_iter
gid = '{input}'
refcmd = f"conda run -p /mnt/storage4/thliao/ML_oxygen/aerobicity/env {indir}/17_apply_to_proteome.py --protein-fasta /mnt/maple/thliao/data/NCBI/modified_data/prokka_o/{gid}/{gid}.faa --eggnog-data-dir {indir}/eggNOG/ --models {indir}/XGBoost.model --output-annotations /mnt/storage4/thliao/ML_oxygen/aerobicity/annotations/{gid}.anno --working-directory /mnt/storage4/thliao/ML_oxygen/aerobicity/annotations/{gid}/work --threads 10"

ingids_list = batch_iter(gids,batch_size=2000)
for _idx,a in enumerate(ingids_list):
    s = generate_sbatch_job_array(f'/mnt/storage4/thliao/ML_oxygen/aerobicity/anno{_idx}.sbatch',
                          inputs=a,
                          command_template=refcmd,
                          log_dir='/mnt/storage4/thliao/ML_oxygen/aerobicity/logs/',
                          percpu=10,
                          jobname=f'{_idx}davin',)
    #os.system(f"sbatch {s}")


import time
for i in list(range(len(ingids_list))):
    print(i)

    while 1:
        r = os.popen(f'sbatch /mnt/storage4/thliao/ML_oxygen/aerobicity/anno{i}.sbatch')
        if 'Submitted batch job' in str(r.read()):
            break
        time.sleep(180)
        
        
from joblib import load
model = load('XGBoost.model')
indir = '/mnt/storage4/thliao/ML_oxygen/aerobicity/'


from glob import glob
from tqdm import tqdm
from os.path import exists
import pandas as pd
from sklearn import metrics
import time
from collections import Counter
# parent_parse and main is came from 17_apply_to_proteome.py (downloaded from https://huggingface.co/wwood/aerobicity)

while 1: 
    for anno in tqdm(glob('/mnt/storage4/thliao/ML_oxygen/training_sets/davin_annotations/*.faa.egg.emapper.hits')):
        if os.path.getsize(anno)!=0:
            gid = anno.split('/')[-1].split('.faa.egg')[0]
            if exists(f"/mnt/storage4/thliao/ML_oxygen/aerobicity/annotations/{gid}.prediction"):
                continue
            if os.path.getsize(f"/mnt/storage4/thliao/ML_oxygen/training_sets/davin_annotations/{gid}.faa.egg.emapper.annotations")==0:
                print(f'empty {gid}')
                #os.system(f"EGGNOG_DATA_DIR=/mnt/storage4/thliao/ML_oxygen/aerobicity/eggNOG && /mnt/storage4/thliao/ML_oxygen/aerobicity/env/bin/python /mnt/storage4/thliao/ML_oxygen/aerobicity/env/bin/emapper.py -m diamond -i /mnt/maple/thliao/data/NCBI/modified_data/prokka_o/{gid}/{gid}.faa --target_orthologs one2one --query_cover 50.0 --evalue 0.0000001 --cpu 10 -o /mnt/storage4/thliao/ML_oxygen/training_sets/davin_annotations/{gid}.faa.egg --data_dir /mnt/storage4/thliao/ML_oxygen/aerobicity/eggNOG --override")
                continue
            #print(gid)
            args = parent_parser.parse_args(f"--protein-fasta /mnt/maple/thliao/data/NCBI/modified_data/prokka_o/{gid}/{gid}.faa --models {indir}/XGBoost.model --output-annotations /mnt/storage4/thliao/ML_oxygen/aerobicity/annotations/{gid}.anno --output-predictions /mnt/storage4/thliao/ML_oxygen/aerobicity/annotations/{gid}.prediction --threads 10 --kofam-tsv-file /mnt/storage4/thliao/ML_oxygen/training_sets/davin_annotations/hmmsearch/{gid}.faa.hmmsearch_tblout.csv --eggnog-annotation-file /mnt/storage4/thliao/ML_oxygen/training_sets/davin_annotations/{gid}.faa.egg.emapper.annotations".split(' '))
            main(args)

    collect_pred = []
    for f in glob('/mnt/storage4/thliao/ML_oxygen/aerobicity/annotations/*.prediction'):
        d = pd.read_csv(f,sep='\t')
        collect_pred.append(d)
    collect_pred = pd.concat(collect_pred,axis=0)

    collect_pred.loc[:,'gid']=[_.split('/')[-1].replace('.faa','') for _ in collect_pred['node']]
    collect_pred = collect_pred.set_index('gid')

    collect_pred.loc[:,'true'] = [genome2oxy_bin[_] for _ in collect_pred.index]

    y_test = collect_pred['true']
    preds = collect_pred['prediction']
    accuracy = metrics.balanced_accuracy_score(y_test, preds )
    print(accuracy)
    print(Counter(y_test))
    # 0.9070795319400204
    # Counter({1: 7073, 0: 1870})
    
    print('\n\nSleeping 1h and start next round\n\n\n\n')
    time.sleep(3600)
    
    if collect_pred.shape[0]==8943:
        break
collect_pred.pop('node')
collect_pred.to_csv(f'/mnt/storage4/thliao/ML_oxygen/aerobicity/8943bac_prediction.tsv',sep='\t',index=1)



# XGBoost.model
# Pipeline(steps=[('maxabsscaler', MaxAbsScaler()),
#                 ('xgbclassifier',
#                  XGBClassifier(base_score=None, booster=None, callbacks=None,
#                                colsample_bylevel=None, colsample_bynode=None,
#                                colsample_bytree=None,
#                                early_stopping_rounds=None,
#                                enable_categorical=False, eval_metric=None,
#                                feature_types=None, gamma=None, gpu_id=None,
#                                grow_policy=None, importance_type=None,
#                                interaction_constraints=None, learning_rate=None,
#                                max_bin=None, max_cat_threshold=None,
#                                max_cat_to_onehot=None, max_delta_step=None,
#                                max_depth=None, max_leaves=None,
#                                min_child_weight=None, missing=nan,
#                                monotone_constraints=None, n_estimators=100,
#                                n_jobs=64, num_parallel_tree=None,
#                                predictor=None, random_state=None, ...))])