



"""R code
library(phytools)
library(ape)
library(caper)
library(phylolm)
library(progress)

keggbin <- read.table("/mnt/ivy/thliao/project/ML_oxygen/training_sets/processed_data/CBdist_based_filter/keggbin_reduced.tsv", stringsAsFactors = F)
supertree <- read.tree("/mnt/storage3/thliao/project/ML_oxygen/training_sets/trees/phylophlan/phylo/in_faa.tre.treefile")
tree_am <- drop.tip(supertree,supertree$tip.label[!(supertree$tip.label %in% rownames(keggbin) )])
tree_am$edge.length[tree_am$edge.length == 0] <- 1e-6
tree_am_root <- midpoint.root(tree_am)
keggbin_sub = keggbin[tree_am_root$tip.label,]




########## single KO (phyloglm) ############################
model <- phyloglm(x ~ 1, phy=tree_am, method="logistic_MPLE")

summary(model)
############################################################

########## multiple KOs (phyloglm) ##########################
n<- length(colnames(keggbin_sub))

pb <- progress_bar$new(
  format = "  Processing [:bar] :percent ETA: :eta",
  total = n, clear = FALSE, width = 60
)

alpha_results <- data.frame(trait = character(), alpha = numeric(), aic=numeric(),logLik=numeric(),stringsAsFactors = FALSE)

for (i in 1:n) {
    col <- colnames(keggbin_sub)[i]
    x <- keggbin_sub[[col]]
    names(x) <- rownames(keggbin_sub)
    if (length(unique(x)) < 2) next
     
    model <- phyloglm(x ~ 1, phy = tree_am_root, method = "logistic_MPLE")
    alpha <- if (!is.null(model$alpha)) model$alpha else if (!is.null(model$optpar)) model$optpar[1] else NA
    intercept <- as.numeric(coef(model)[1])
    pval <- summary(model)$coefficients[1, "p.value"]
    aic <- model$aic
    logLik <- model$logLik

    alpha_results <- rbind(alpha_results, data.frame(trait = col, alpha = alpha,aic=aic,logLik=logLik))
    pb$tick()
  }
  
write.table(alpha_results,'/mnt/ivy/thliao/project/ML_oxygen/training_sets/phylo_D_results/phyloglm_results.tsv')

# ✅ α（alpha）
# 代表 系统发育信号的强度（phylogenetic correlation strength）：
# 如果 α → 0，说明系统发育信号很强（特征高度保守，phylogenetic inertia 强）；
# 如果 α → ∞，说明系统发育信号几乎不存在（独立演化，类似普通逻辑回归）；
# 你这里的 α = 1.631，属于 中等水平的系统发育相关性。
# 换句话说：你的二元性状在系统发育树上的相关性存在，但不是非常强。

############################################################

########## single KO (phylo.d D statistic) ##################
df <- data.frame(species = names(x), trait = x)
comp_data <- comparative.data(tree_am_root, df, names.col="species", vcv=TRUE)
phylo_d <- phylo.d(comp_data, binvar=trait, permut=1000)
print(phylo_d)
# D 值范围	含义
# D = 1	性状在树上完全随机分布（无系统发育信号）
# D = 0	性状在树上分布符合 Brownian motion（强系统发育保守性）
# D < 0	比 Brownian 模型还更强的系统发育聚集（极度保守）
# D > 1	比随机还更分散（性状在树上几乎无关）

# 在你这里：
# 🟡 D = -0.246，意味着你的性状在系统发育树上高度聚集，
# 比 Brownian 过程还略强一些（即“phylogenetically conserved”）。
############################################################

########## Multiple KO (phylo.d D statistic) ##################
n<- length(colnames(keggbin_sub))

pb <- progress_bar$new(
  format = "  Processing [:bar] :percent ETA: :eta",
  total = n, clear = FALSE, width = 60
)

phylo.d_results <- data.frame(trait = character(), Destimate = numeric(), pval0=numeric(),pval1=numeric(),stringsAsFactors = FALSE)


for (col in colnames(keggbin_sub)) {
    x <- keggbin_sub[[col]]
    names(x) <- rownames(keggbin_sub)    
    df <- data.frame(species = names(x), trait = x)
    comp_data <- comparative.data(tree_am_root, df, names.col="species", vcv=TRUE)
    phylo_d <- phylo.d(comp_data, binvar=trait, permut=1000)

    phylo.d_results <- rbind(phylo.d_results, data.frame(trait = col, Destimate = phylo_d$DEstimate[['Obs']],pval0=phylo_d$Pval0,pval1=phylo_d$Pval1))
    pb$tick()
  }
  
phylo.d_results
############################################################




"""

from ete3 import Tree

import pandas as pd
keggbin = pd.read_csv("/mnt/ivy/thliao/project/ML_oxygen/training_sets/processed_data/CBdist_based_filter/keggbin_reduced.tsv",sep='\t',index_col=0)

tre = Tree("/mnt/storage3/thliao/project/ML_oxygen/training_sets/trees/phylophlan/phylo/in_faa.tre.treefile")






"""
library(caper)
library(parallel)
library(pbmcapply)
library(phytools)

keggbin <- read.table("/mnt/ivy/thliao/project/ML_oxygen/training_sets/processed_data/CBdist_based_filter/keggbin_reduced.tsv", stringsAsFactors = F)


supertree <- read.tree("/mnt/storage3/thliao/project/ML_oxygen/training_sets/trees/phylophlan/phylo/in_faa.tre.treefile")
tree_am <- drop.tip(supertree,supertree$tip.label[!(supertree$tip.label %in% rownames(keggbin) )])
tree_am$edge.length[tree_am$edge.length == 0] <- 1e-6
tree_am_root <- midpoint.root(tree_am)
keggbin_sub = keggbin[tree_am_root$tip.label,]
save(keggbin_sub, tree_am_root,
     file = "/mnt/ivy/thliao/project/ML_oxygen/training_sets/phylo_D_results/scripts/phylo_d_used.RData")
     
# 设定核心数，自动检测可用核心
ncore <- detectCores() - 1

# 定义单个 trait 的函数
run_phylod <- function(col) {
  x <- keggbin_sub[[col]]
  names(x) <- rownames(keggbin_sub)
  df <- data.frame(species = names(x), trait = x)
  
  # 捕获错误（避免某些 trait 出错导致整个任务中断）
  out <- tryCatch({
    comp_data <- comparative.data(tree_am_root, df, names.col = "species", vcv = TRUE)
    ph <- phylo.d(comp_data, binvar = trait, permut = 1000)
    data.frame(
      trait = col,
      DEstimate = ph$DEstimate[["Obs"]],
      pval0 = ph$Pval0,
      pval1 = ph$Pval1,
      stringsAsFactors = FALSE
    )
  }, error = function(e) {
    data.frame(trait = col, DEstimate = NA, pval0 = NA, pval1 = NA)
  })
  return(out)
}

# 使用 pbmclapply（自动进度条 + 并行）
phylo.d_results <- pbmclapply(colnames(keggbin_sub), run_phylod, mc.cores = ncore)


"""




from bin.multiple_sbatch import generate_sbatch_job_array,batch_iter
from bin.multiple_sbatch import batch_iter
import pandas as pd
import os
df = pd.read_csv('/mnt/ivy/thliao/project/ML_oxygen/training_sets/processed_data/CBdist_based_filter/keggbin_reduced.tsv',sep='\t',index_col=0)

koids = list(df.columns)
refcmd = "/mnt/home-user/thliao/anaconda3/envs/r_env/bin/Rscript /mnt/ivy/thliao/project/ML_oxygen/training_sets/phylo_D_results/scripts/single.R --col {input}"

ingids_list = batch_iter(koids,batch_size=2000)
for _idx,a in enumerate(ingids_list):
    s = generate_sbatch_job_array(f'/mnt/ivy/thliao/project/ML_oxygen/training_sets/phylo_D_results/scripts/phyloD{_idx}.sbatch',
                          inputs=a,
                          command_template=refcmd,
                          log_dir='/mnt/ivy/thliao/project/ML_oxygen/training_sets/phylo_D_results/scripts/logs/',
                          percpu=5,
                          jobname=f'{_idx}phyloD',)
    os.system(f"sbatch {s}")



keggbin = pd.read_csv('/mnt/ivy/thliao/project/ML_oxygen/training_sets/processed_data/CBdist_based_filter/keggbin_reduced.tsv',sep='\t',index_col=0)

df = pd.read_csv('/mnt/ivy/thliao/project/ML_oxygen/training_sets/phylo_D_results/phyloglm_results.tsv',sep=' ',index_col=0)
df = df.set_index('trait')

missing_traits = [_ for _ in keggbin.columns if _ not in df.index]

top40_ko = open('/mnt/ivy/thliao/project/ML_oxygen/training_sets/processed_data/CBdist_based_filter/top40.txt').read().strip().split('\n')
df.loc[:,'GBDT40'] = ['Yes' if _ in top40_ko else 'No' for _ in df.index]

from glob import glob
for f in glob('/mnt/ivy/thliao/project/ML_oxygen/training_sets/phylo_D_results/phylod_o/phyloD_*.csv'):
    dstat = pd.read_csv(f)
    ko = f.split('_')[-1].replace('.csv','')
    df.loc[ko,'Dstats'] = dstat.iloc[0,2]

notuniquekos = [col for col,v in keggbin.iteritems() if len(v.unique())==2]
df = df.reindex(notuniquekos)

df.to_csv('/mnt/ivy/thliao/project/ML_oxygen/training_sets/phylo_D_results/phylosignal_summary.tsv',sep='\t',index=1)
df.loc[df.GBDT40=='Yes',:]

