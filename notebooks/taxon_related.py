import sys
sys.path.insert(0,'/mnt/ivy/thliao/project/ML_oxygen/scripts/')
from general_func import *
from load_data import *
def vis_wf(wf,tax='',show=False,SD=True):
#def vis(self,tax='',show=False):
    if not tax:
        tax = wf.study_taxa
    else:
        tax = [tax]
    figs = []
    for taxon in tax:
        a,b,c,d = wf.AllTaxaM2specificTaxa[taxon],wf.AllTaxaM2otherTaxa[taxon],\
            wf.taxaSpecM2specificTaxa[taxon],wf.taxaSpecM2otherTaxa[taxon]
        
        z = [[round(np.mean(a), 4), 
                round(np.mean(b), 4)],
            [round(np.mean(c), 4), 
                round(np.mean(d), 4)]]
        x = ['Taxon-Specific<Br>dataset', 'Other-Taxa<Br>dataset']
        y = ['Taxa-Mixed<Br>model', "Taxa-Specific<Br>model"]
        
        z_text = [[f"{round(np.mean(a), 2)}<Br>\u00B1{round(np.std(a),2)}", 
                   f"{round(np.mean(b), 2)}<Br>\u00B1{round(np.std(b),2)}"],
                  [f"{round(np.mean(c), 2)}<Br>\u00B1{round(np.std(c),2)}", 
                   f"{round(np.mean(d), 2)}<Br>\u00B1{round(np.std(d),2)}"]]
        if SD:
            _f = ff.create_annotated_heatmap(z, annotation_text = z_text, font_colors=['#000000'])
        else:
            _f = ff.create_annotated_heatmap(z,  font_colors=['#000000'])
        fig = px.imshow(1-np.array(z),
                        x=x,
                        y=y,
                        color_continuous_scale='RdBu',
                        range_color=[0, 0.5],)
        # disable the colorbar
        fig.update(layout_coloraxis_showscale=False)
        fig.layout.annotations = _f.layout.annotations
        fig.layout.title = taxon
        fig.layout.title.x = 0.5
        fig.layout.title.y = 0.85
        fig.layout.height = 300
        fig.layout.width = 350
        figs.append(fig)
        if show:
            img_bytes = fig.to_image(format="png", engine="kaleido")
            display(Image(img_bytes, format='png'))
    return figs



tids = ['phylum', 'class', 'order', 'family', 'genus', 'species',
       'superkingdom', 'gid','label']

def get_ndata(reduced_data,y_bin):
    sub_tax_df = tax_df.reindex([_.split('.')[0] for _ in bac_data_df.index])
    sub_tax_df.index = bac_data_df.index
    ndata = reduced_data.copy()
    ndata = pd.concat([ndata,sub_tax_df.reindex(ndata.index)],axis=1)
    ndata.loc[:,'label'] = y_bin
    return ndata

def get_data(ndata,y_bin):
#     ndata = ndata.loc[:,[_ for _ in ndata if _ not in tids]]
    sssp = StratifiedShuffleSplit(n_splits=3)
    for train_index, test_index in sssp.split(ndata,y_bin):
        X_train, X_test = ndata.iloc[train_index,:], ndata.iloc[test_index, :]
        break
    return X_train,X_test

from sklearn.metrics import confusion_matrix
def validate_it(data1, data2, m_f = LR_m,return_confusion_matrix=False):
    y2 = data2['label']
    c = data2.columns.difference(tids) # remove uncessary features
    model = m_f(data2.loc[:,c], y2, X_test=None, y_test=None)
    tax2auc = {}
    tax2acc = {}
    tax2AP = {}
    tax2confusion_matrix = {}
    for name,p in [('Actinobacteria','phylum'),
                   ('Firmicutes','phylum'),
                   ('Bacteroidetes','phylum'),
                   ('Proteobacteria','phylum'),
                   ('Alphaproteobacteria','class'),
                   ('Betaproteobacteria','class'),
                   ('Gammaproteobacteria','class')]:
        sub_data = data1.loc[data1[p]==name,:]
        sub_y = sub_data['label']
        sub_data = sub_data.loc[:,c]
        num_s = sub_data.shape[0]
        if type(model) == xgb.Booster:
            sub_data = xgb.DMatrix(sub_data)
        if 'predict_proba' in dir(model):
            y_pred = model.predict_proba(sub_data)
            auc = metrics.roc_auc_score(sub_y, y_pred[:, 1])
            AP = metrics.average_precision_score(sub_y, y_pred[:, 1])
            tax2auc[f"{name} ({sub_data.shape[0]})"] = auc
            tax2AP[f"{name} ({num_s})"] = AP
        y_pred_label = model.predict(sub_data)
        if len(set(y_pred_label))==2:
            accuracy = metrics.balanced_accuracy_score(sub_y, y_pred_label)
            tn, fp, fn, tp = confusion_matrix(sub_y, y_pred_label).ravel()
            tax2confusion_matrix[f"{name} ({num_s})"] = tn, fp, fn, tp
        else:
            # only probility can used for auc and AP calculation
            # maybe GBDT model y_pred_label is a prob
            accuracy = metrics.balanced_accuracy_score(sub_y, y_pred_label>0.5)
            auc = metrics.roc_auc_score(sub_y, y_pred_label)
            tax2auc[f"{name} ({num_s})"] = auc
            AP = metrics.average_precision_score(sub_y, y_pred_label)
            tax2auc[f"{name} ({num_s})"] = auc
            tax2AP[f"{name} ({num_s})"] = AP
            tn, fp, fn, tp = confusion_matrix(sub_y, y_pred_label>0.5).ravel()
            tax2confusion_matrix[f"{name} ({num_s})"] = tn, fp, fn, tp            
        tax2acc[f"{name} ({num_s})"] = accuracy
    if return_confusion_matrix:
        return tax2auc,tax2acc,tax2AP,tax2confusion_matrix
    return tax2auc,tax2acc,tax2AP

def gen_drawdf(indata):
    draw_df = pd.DataFrame()
    for k,v in indata.items():
        name = k.split(' ')[0]
        num = int(k.split('(')[-1].strip(')'))
        for _ in v:
            draw_df = draw_df.append(pd.DataFrame(np.array([name,num,_]).reshape(1,-1)))
    draw_df.columns = ['tax','numbers','values']
    return draw_df
