import dendropy
tree = dendropy.Tree.get(path="/mnt/storage3/thliao/project/ML_oxygen/scripts/CompareFinalEffect/Figure3_timetree.nex", schema="nexus")
used_genomes = [_.taxon.label for _ in tree.leaf_nodes()]
print(len(used_genomes))




import pandas as pd
from os.path import *
import os
os.chdir('/mnt/storage3/thliao/project/ML_oxygen/scripts/CompareFinalEffect')
d = pd.read_excel('./MDF.xlsx',sheet_name='1.General_information')
code2genome = dict(zip(d['ShortCode'],d['accession']))

predicted_info = pd.read_excel('./MDF.xlsx',sheet_name='4.Gene-content_classifier')
predicted_info = predicted_info.loc[predicted_info['model']=='XGBoost',:]



meta_df = pd.read_csv('/mnt/home-db/pub/gtdb/release214/metadata/bac120_metadata_r214.tsv',sep='\t',index_col=0)
refined_code2genome = {}
for code,gid in code2genome.items():
    faa = f"/mnt/storage4/thliao/ML_oxygen/gtdbr214/gtdbr214_faa/{gid}.faa"
    if not exists(faa):
        refined_gid = [_ for _ in meta_df.index if gid.split('_')[-1].split('.')[0] in _ ]
        if len(refined_gid)!=1:
            print(gid)
        else:
            refined_code2genome[code] = refined_gid[0]
    else:
        refined_code2genome[code] = gid


missing_faa = {'RS_GCF_000688095.1':'/mnt/maple/thliao/data/NCBI/modified_data/prokka_o/GCA_000688095.1/GCA_000688095.1.faa',
               'RS_GCF_000012525.1':'/mnt/maple/thliao/data/NCBI/modified_data/prokka_o/GCA_000012525.1/GCA_000012525.1.faa',
               'RS_GCF_000245055.1': '/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/extra_tmp/Desua53/Desua53.faa',
                   #'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/012/525/GCF_000012525.1_ASM1252v1/GCF_000012525.1_ASM1252v1_protein.faa.gz',
               'RS_GCF_000023645.1':'/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/extra_tmp/RS_GCF_000023645.1.faa',
               'GB_GCA_004338625.1':'/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/extra_tmp/GB_GCA_004338625.1.faa',
               'RS_GCF_003268875.1':'/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/extra_tmp/RS_GCF_003268875.1.faa'
               }      

import os
code2faa = {}
for code,gid in refined_code2genome.items():
    faa = f"/mnt/storage4/thliao/ML_oxygen/gtdbr214/gtdbr214_faa/{gid}.faa"
    if exists(faa):
        code2faa[code] = faa    
    else:
        if exists(missing_faa.get(gid,'')):
            code2faa[code] = missing_faa[gid]
            continue
        _gid = meta_df.loc[gid,'ncbi_genbank_assembly_accession']
        potential_faa = f"/mnt/maple/thliao/data/NCBI/modified_data/prokka_o/{_gid}/{_gid}.faa"
        if exists(potential_faa):
            code2faa[code] = potential_faa
print(len(code2faa),len(code2genome))

missing_code = [(code,genome) for code,genome in code2genome.items() if code not in code2faa]
for code,gid in missing_code:
    if exists(missing_faa.get(gid,'')):
        code2faa[code] = missing_faa[gid]
# [('Acid29', 'GB_GCA_002333435.1'),
#  ('Acti30', 'GB_GCA_001871795.1'),
#  ('Acti49', 'RS_GCF_000688095.1'),
#  ('Camp7', 'GB_GCA_001816265.1'),
#  ('Cyan44', 'RS_GCF_000012525.1'),
#  ('Desua52', 'RS_GCF_000245055.1'),
#  ('Desua53', 'RS_GCF_000023645.1'),
#  ('Fibr1', 'GB_GCA_002414805.1'),
#  ('FirmA72', 'RS_GCF_005845105.1'),
#  ('Firms28', 'GB_GCA_004338625.1'),
#  ('ProtG59', 'RS_GCF_003268875.1'),
#  ('Spir10', 'RS_GCF_000513775.1')]
for code,genome in missing_code:
    row = str(os.popen(f"grep {genome.split('_')[-1]} /mnt/home-user/thliao/.cache/ncbi-genome-download/genbank_bacteria_assembly_summary.txt").read())
    rows = row.split('\t')
    if len(rows)<=1:
        print(code,genome)
        continue
    ftplink = rows[-19]
    if not ftplink:
        print(code,genome)
    os.system(f"wget https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/513/775/GCA_000513775.1_ASM51377v1/GCA_000513775.1_ASM51377v1_genomic.fna.gz -O /mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/extra_tmp/{genome}.genomic.gz && gzip -d /mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/extra_tmp/{genome}.genomic.gz")
for code,genome in missing_code:
    os.system(f"~/anaconda3/bin/prokka /mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/extra_tmp/{genome}.genomic --outdir /mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/extra_tmp/{code} --force --prefix {code} --locustag {code} --cpus 20")
    
for code,genome in missing_code:
    code2faa[code] = f'/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/extra_tmp/{code}/{code}.faa'
    
from tqdm import tqdm
for code, faa in tqdm(code2faa.items()):
    if exists(f"/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/prepared_proteins/{code}.faa"):
        continue
    os.system(f"cp {faa} /mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/prepared_proteins/{code}.faa")


tree = dendropy.Tree.get(path="/mnt/storage3/thliao/project/ML_oxygen/scripts/CompareFinalEffect/Figure3_timetree.nex", schema="nexus")
print(len(tree.leaf_nodes()),len(tree.nodes()))
used_genomes = [_.taxon.label for _ in tree.leaf_nodes()]
tree.prune_taxa_with_labels([_ for _ in used_genomes  if _ not in code2faa])
print(len(tree.leaf_nodes()),len(tree.nodes()))
tree.write(path='/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/reftree.nexus',schema='nexus')
tree.write(path='/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/reftree.newick',schema='newick')

        
# ~/anaconda3/bin/prokka RS_GCF_000245055.1_genomic.fasta --outdir ./extra_tmp/Desua53 --force --prefix Desua53 --locustag Desua53 --cpus 20


def tsv2list(f):
    rows = open(f).read().strip().split('\n')
    with open(f.replace('_contigs.rsv','.dna.list'),'w') as f1:
        for i in rows[1:]:
            f1.write(i.split('\t')[1]+'\n')    
# tsv2list('JIAT01_contigs.tsv')






#### pfam annotations
from glob import glob
from os.path import *
        
# ls prepared_proteins/*faa |parallel hmmsearch --tblout /mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/annoPfamv33.1/{/.}_domtblout.dat -o /dev/null --notextw --cpu 1 /mnt/ivy/thliao/db/protein_db/Pfam.v33.1/Pfam-A.hmm {}

# ls prepared_proteins/*faa |parallel -j 5 hmmsearch --domtblout /mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/annoPfamv33.1_domtbl/{/.}_domtblout.dat -o /dev/null --notextw --cpu 5 /mnt/ivy/thliao/db/protein_db/Pfam.v33.1/Pfam-A.hmm {}


from collections import defaultdict
output_dir = '/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/'
pfam_out_dir = "/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/annoPfamv33.1/"
suffix = '_domtblout.dat'
anno_file = f"{output_dir}/pfam_anno.tab"
gid2pfam2locus_info = {}
import io
for ofile in tqdm(glob(join(pfam_out_dir, f"*{suffix}"))):
    gid = ofile.split('/')[-1].replace(suffix,'')

    
    hmmer_colnames = ['target name','target accession','query name','accession','E-value','score_overall','bias_overall']
    cleaned = "\n".join(filter(lambda x: not x.startswith("#"), 
                             ["\t".join(i.split(None, 7))
                              for i in open(ofile).readlines()]))
    filtered_df = pd.read_csv(io.StringIO(cleaned), sep = "\t",  header = None,low_memory=False)
    filtered_df = filtered_df.iloc[:,:len(hmmer_colnames)]
    filtered_df.columns = hmmer_colnames

    locus_unique2df = filtered_df.sort_values('E-value').groupby('target name').head(1)
    pfam2locus_info = defaultdict(list)
    for idx,row in locus_unique2df.iterrows():
        pfam2locus_info[row['accession']].append(row['target name'])
    gid2pfam2locus_info[gid] = {ko:','.join([_ for _ in l_list]) for ko,l_list in pfam2locus_info.items()}
final_df = pd.DataFrame.from_dict(gid2pfam2locus_info, orient='index')
final_df.to_csv(anno_file, sep='\t', index=1)

