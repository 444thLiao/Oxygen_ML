






# cd /mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/
# ls prepared_proteins/*faa |parallel hmmsearch --tblout davinPredictor/hmmsearch/{/}.hmmsearch_tblout.csv -o /dev/null --notextw --cpu 1 /mnt/storage4/thliao/ML_oxygen/aerobicity/data/kofam-2022-01-30-profiles.hmm {}
######### now we can pass the csv file to parameter "--kofam-tsv-file"



########## preparing file for this parameter "eggnog_annotation_file"
# conda activate /mnt/storage4/thliao/ML_oxygen/aerobicity/env
# ls prepared_proteins/*faa |parallel -j 10 --joblog emapper.log emapper.py -m diamond -i {} --target_orthologs one2one --query_cover 50.0 --evalue 0.0000001 --cpu 1 -o davinPredictor/eggnog/{/}.egg --data_dir /mnt/storage4/thliao/ML_oxygen/aerobicity/eggNOG --override



from bin.multiple_sbatch import generate_sbatch_job_array,batch_iter
from bin.multiple_sbatch import batch_iter
gid = '{input}'
refcmd = f"EGGNOG_DATA_DIR=/mnt/storage4/thliao/ML_oxygen/aerobicity/eggNOG && /mnt/storage4/thliao/ML_oxygen/aerobicity/env/bin/python /mnt/storage4/thliao/ML_oxygen/aerobicity/env/bin/emapper.py -m diamond -i /mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/prepared_proteins/{gid}.faa --target_orthologs one2one --query_cover 50.0 --evalue 0.0000001 --cpu 10 -o /mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/davinPredictor/eggnog/{gid}.faa.egg --data_dir /mnt/storage4/thliao/ML_oxygen/aerobicity/eggNOG --override"
ingids = []
for ofaa in glob('/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/prepared_proteins/*.faa'):
    gid = ofaa.split('/')[-1].replace('.faa','')
    ofile = f"/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/davinPredictor/eggnog/{gid}.faa.egg.emapper.annotations"
    if not exists(ofile) or os.path.getsize(ofile)==0:
        ingids.append(gid)
ingids_list = batch_iter(ingids,batch_size=2000)
for _idx,a in enumerate(ingids_list):
    s = generate_sbatch_job_array(f'/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/davinPredictor/eggnog{_idx}.sbatch',
                          inputs=a,
                          command_template=refcmd,
                          log_dir='/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/davinPredictor/eggnog_logs/',
                          percpu=10,
                          jobname=f'{_idx}_1007',)
    os.system(f"sbatch {s}")

# scontrol release <jobid>
# scontrol hold <jobid>

#### split merged into multiple

from tqdm import tqdm
from glob import glob
from os.path import exists,join
import os


# FUNCTIONS parent_parser and main is came from 17_apply_to_proteome.py (found from )
indir = '/mnt/storage4/thliao/ML_oxygen/aerobicity/'
for anno in tqdm(glob('/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/davinPredictor/eggnog/*.faa.egg.emapper.hits')):
    if os.path.getsize(anno)!=0:
        gid = anno.split('/')[-1].split('.faa.egg')[0]
        if exists(f"/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/davinPredictor/prediction/{gid}.prediction"):
            continue
        if os.path.getsize(f"/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/davinPredictor/eggnog/{gid}.faa.egg.emapper.annotations")==0:
            print(f'empty {gid}')
            continue
        faa = f'/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/prepared_proteins/{gid}.faa'
        args = parent_parser.parse_args(f"--protein-fasta {faa} --models {indir}/XGBoost.model --output-annotations /mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/davinPredictor/prediction/{gid}.anno --output-predictions /mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/davinPredictor/prediction/{gid}.prediction --threads 10 --kofam-tsv-file /mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/davinPredictor/hmmsearch/{gid}.faa.hmmsearch_tblout.csv --eggnog-annotation-file /mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/davinPredictor/eggnog/{gid}.faa.egg.emapper.annotations".split(' '))
        main(args)
        
        
import pandas as pd
collect_pred = []
for f in glob('/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/davinPredictor/prediction/*.prediction'):
    d = pd.read_csv(f,sep='\t')
    collect_pred.append(d)
collect_pred = pd.concat(collect_pred,axis=0)

collect_pred.loc[:,'gid']=[_.split('/')[-1].replace('.faa','') for _ in collect_pred['node']]
collect_pred = collect_pred.set_index('gid')
collect_pred.to_csv('/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/davinPredictor/predicted_results.tsv',sep='\t',index=0)





    