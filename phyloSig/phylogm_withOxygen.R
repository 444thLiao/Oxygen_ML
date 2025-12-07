
## evaluate the relationship between the oxygen requirement and the presence or absence of keeg gene. 


#
# df = pd.read_csv('/mnt/ivy/thliao/project/ML_oxygen/training_sets/processed_data/CBdist_based_filter/keggbin_reduced.tsv',sep='\t',index_col=0)
# df.loc[:,'oxygen'] = [genome2oxy_bin[_] for _ in df.index]
# df.to_csv('/mnt/storage3/thliao/project/ML_oxygen/training_sets/extra_ValidateGBDT40/allKO.tsv')

infile = f"/mnt/ivy/thliao/project/ML_oxygen/training_sets/processed_data/CBdist_based_filter/forward_selection/GBDT_SHAP_KEGG/40_fulldf.tsv"
fulldf_GBDT = pd.read_csv(infile,sep='\t',index_col=0)
gbdt40 = set(fulldf_GBDT.index[:-3])

#
library(phytools)
library(ape)
library(caper)
library(phylolm)
library(progress)

keggbin <- read.table("/mnt/storage3/thliao/project/ML_oxygen/training_sets/extra_ValidateGBDT40/allKO.tsv", stringsAsFactors = F)
supertree <- read.tree("/mnt/storage3/thliao/project/ML_oxygen/training_sets/trees/phylophlan/phylo/in_faa.tre.treefile")
tree_am <- drop.tip(supertree,supertree$tip.label[!(supertree$tip.label %in% rownames(keggbin) )])
tree_am$edge.length[tree_am$edge.length == 0] <- 1e-6
tree_am_root <- midpoint.root(tree_am)
keggbin_sub = keggbin[tree_am_root$tip.label,]


fit <- phyloglm(oxygen ~ K00383,data=keggbin_sub, phy=tree_am_root, method="logistic_MPLE")
n<- length(colnames(keggbin_sub)) - 1


pb <- progress_bar$new(
  format = "  Processing [:bar] :percent ETA: :eta",
  total = n, clear = FALSE, width = 60
)


alpha_results <- data.frame(trait = character(), alpha = numeric(), aic=numeric(),logLik=numeric(),stringsAsFactors = FALSE)

for (i in 1:n) {
    col <- colnames(keggbin_sub)[i]
    x <- keggbin_sub[[col]]
    if (length(unique(x)) < 2) next
    formula <- as.formula(paste("oxygen ~", col))
    model <- phyloglm(formula, data=keggbin_sub,phy = tree_am_root, method = "logistic_MPLE")
    alpha <- if (!is.null(model$alpha)) model$alpha else if (!is.null(model$optpar)) model$optpar[1] else NA
    intercept <- as.numeric(coef(model)[1])
    pval <- summary(model)$coefficients[1, "p.value"]
    zval <- summary(model)$coefficients[1, "z.value"]
    estimate <- summary(model)$coefficients[1,'Estimate']
    aic <- model$aic
    logLik <- model$logLik

    alpha_results <- rbind(alpha_results, data.frame(trait = col, estimate=estimate,alpha = alpha,aic=aic,logLik=logLik,
    pvalue=pval,zvalue=zval))
    pb$tick()
  }

write.table(alpha_results,'/mnt/ivy/thliao/project/ML_oxygen/training_sets/extra_ValidateGBDT40/phyloglmWithOxygen_results.tsv',sep='\t')





"""
df = pd.read_csv('/mnt/ivy/thliao/project/ML_oxygen/training_sets/extra_ValidateGBDT40/phyloglmWithOxygen_results.tsv',sep='\t',index_col=0)

top40_ko = open('/mnt/ivy/thliao/project/ML_oxygen/training_sets/processed_data/CBdist_based_filter/top40.txt').read().strip().split('\n')
df.loc[:,'GBDT40'] = ['Yes' if _ in top40_ko else 'No' for _ in df['trait']]


df.loc[df.GBDT40=='Yes',:]
"""