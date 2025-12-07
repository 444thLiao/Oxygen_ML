





hal_file = '/mnt/ivy/thliao/project/ML_oxygen/training_sets/processed_data/CBdist_based_filter/top40_2023.hal'
odir = f"/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/annotop40"
indir = '/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/prepared_proteins'

from os.path import exists,dirname
from glob import glob
import os
from tqdm import tqdm
os.chdir(indir)
for outfile in tqdm(glob(f'{indir}/*.faa')):
    gid = outfile.split('/')[-1].replace('.faa','')
    
    cmd = f"/home-user/thliao/software/kofamscan/exec_annotation -p {hal_file} -k /mnt/home-db/pub/protein_db/kegg/v20230301/ko_list --tmp-dir {odir}/.{gid} -o {odir}/{gid}.kofamout -f mapper-one-line --no-report-unannotated {outfile} && rm -rf {odir}/.{gid}"
    # cmds.append(cmd)
    os.system(cmd)
# 47:22 mins



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

final_df.to_csv(f"{dirname(odir)}/Top40_details.tab",sep='\t',index=1)
bin_df = final_df.applymap(lambda x: 0 if pd.isna(x) else 1)
bin_df.to_csv(f"{dirname(odir)}/Top40_bin.tab",sep='\t',index=1)



# prediction
import pickle
LR_model = pickle.load(open("/mnt/storage3/thliao/project/ML_oxygen/testing_sets/trained_LR.model",'rb'))

top40 = open('/mnt/ivy/thliao/project/ML_oxygen/training_sets/processed_data/CBdist_based_filter/top40.txt').read().strip().split('\n')
extant_predicted = bin_df.reindex(columns=top40)
extant_predicted.loc[:,'LR'] = LR_model.predict(extant_predicted.loc[:,top40])
extant_predicted.loc[:,'LR prob'] = LR_model.predict_proba(extant_predicted.loc[:,top40])[:,LR_model.classes_==1].reshape(-1)
extant_predicted.to_csv('/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/annotop40/top40.prediction',sep='\t',index=1)

# Ancestral reconstruction
from ete3 import Tree
gids = Tree('/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/reftree.newick',3).get_leaf_names()
bin_df = bin_df.reindex(gids)
os.system(f"mkdir -p {odir}/ACE/")
for ko,col in tqdm(bin_df.iteritems()):
    with open(f'{odir}/ACE/{ko}.tab','w') as f1:
        f1.write(f"Gid\tstats\n")
        for k,v in col.to_dict().items():
            f1.write(f"{k}\t{v}\n") 
    cmd = f"/mnt/home-user/thliao/anaconda3/envs/r_env/bin/Rscript /mnt/storage3/thliao/project/ML_oxygen/testing_sets/ace.r {odir}/ACE/{ko}.tab /mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/reftree.newick {odir}/ACE/{ko}.anc"
    os.system(cmd)
    

dfs = []
for f in tqdm(glob(f'{odir}/ACE/*.anc')):
    d = pd.read_csv(f,sep='\t',index_col=0)
    #d.index = ['OROOT'] + list(d.index)[1:]
    name = f.split('/')[-1].split('.')[0]
    d.loc[:,name] = [r[1] for _,r in d.iterrows()]
    dfs.append(d.loc[:,[name]])
total_df = pd.concat(dfs,axis=1)
total_df = total_df.reindex(columns=top40).fillna(0)
ances_predicted = total_df.copy()

ances_predicted.loc[:,'LR'] = LR_model.predict(total_df)
ances_predicted.loc[:,'LR prob'] = LR_model.predict_proba(total_df)[:,LR_model.classes_==1].reshape(-1)

from collections import Counter
Counter(ances_predicted['LR'])



import dendropy
tree = dendropy.Tree.get(path='/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/reftree.nexus', schema="nexus")
node_ref2info = {}
for node in tqdm(tree.postorder_internal_node_iter()):
    leaf_labels = tuple(sorted([lf.taxon.label for lf in node.leaf_nodes()]))
    if node.annotations:
        node_ref2info[int(node.annotations.get_value('index'))] = node.annotations.get_value('age_95%_HPD')
subances_predicted = ances_predicted.loc[:,['LR','LR prob']]
for i,(cil,cir) in node_ref2info.items():
    cil,cir = float(cil),float(cir)
    subances_predicted.loc[i,'CI'] = f"{round(cil,2)}-{round(cir,2)}"
    subances_predicted.loc[i,'divtime'] = round((cil+cir)/2,2)

max_divtime = max(subances_predicted['divtime'])/100
d1 = ances_predicted[['LR prob']]
d1.columns = ['A']
d1.loc[:,'N'] = 1-d1['A']
d1.loc[:,'dist'] = max_divtime-subances_predicted.reindex(d1.index)['divtime']/100
d1.loc[:,'divtime'] = subances_predicted.reindex(d1.index)['divtime']/100
d1.loc[:,'CI'] = subances_predicted.reindex(d1.index)['CI']
print(d1.loc[d1['A']>=d1['N'],:].sort_values('divtime'))
#d1 is all genomes
d2 = extant_predicted[['LR prob']]
d2.columns = ['A']
d2.loc[:,'N'] = 1-d2['A']
d2.loc[:,'dist'] = max_divtime
#d2 is all internal nodes
merged = pd.concat([d1,d2],axis=0)

import plotly.graph_objects as go
import numpy as np
col = 'dist'
hist_d = np.histogram(merged[col], 
                      bins=20, range = (0,max(merged[col])) )
xs = []
ys1 = []
ys2 = []
ys = []
left = 0
for right in tqdm(hist_d[1]):
    #print(left,right)
    sdf = merged.loc[(merged[col]>=left) & (merged[col]<right),:]
    if sdf.shape[0]==0:continue
    num_anaerobe = len(sdf.index[sdf['A']<0.5])
    num_aerobe = sdf.shape[0]-num_anaerobe
    ys2.append(num_aerobe/sdf.shape[0])
    ys1.append(num_anaerobe/sdf.shape[0])
    ys.append(sdf.shape[0]/(merged.shape[0]-d1.shape[0]))
    xs.append(max(merged[col])-(left+right)/2)
    left = right

fig = go.Figure()
fig.add_scatter(x=xs,y=ys2,mode='lines',opacity=0.7,fillcolor='#FFC75F', 
                marker=dict(size=4),
                name='Aerobe',fill='tozeroy',stackgroup='one',line=dict(width=0),showlegend=False)
fig.add_scatter(x=xs,y=ys1,mode='lines',opacity=0.7,fillcolor='#D65DB1',
                name='Anaerobe',fill='tonexty',stackgroup='one',line=dict(width=0),showlegend=False)
# fig.add_scatter(x=xs,y=ys,mode='markers+lines',name='Ratio of anaerobe')
fig.update_layout(height=400,width=600,xaxis_range=[max(merged[col]),0],yaxis_zeroline=False,yaxis_range=[-0.01,1],
                  yaxis_dtick=0.1,yaxis_tickformat=',.0%',
                  font_size=20,
                 template='simple_white')
fig.write_image('/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/TOP40_timeChanges.png')




#              A         N     dist  divtime               CI
# 249   0.903023  0.096977  41.6859   0.0106        0.09-2.04
# 1270  0.563743  0.436257  41.0277   0.6688     26.01-107.75
# 1689  0.695546  0.304454  40.4144   1.2821     52.33-204.09
# 1605  0.695897  0.304103  40.2698   1.4267      58.95-226.4
# 1193  0.914407  0.085593  40.2188   1.4777      65.5-230.03
# ...        ...       ...      ...      ...              ...
# 14    0.558067  0.441933  20.9510  20.7455   1900.1-2249.01
# 179   0.819514  0.180486  20.6471  21.0494  2004.17-2205.72
# 394   0.647246  0.352754  20.1659  21.5306  1988.76-2317.35
# 178   0.692933  0.307067  19.6176  22.0789  2097.76-2318.02
# 393   0.541015  0.458985  19.1213  22.5752  2109.68-2405.37

