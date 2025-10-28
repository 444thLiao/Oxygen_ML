#!/usr/bin/env Rscript

## ------------------- Libraries ------------------- ##
suppressMessages({
  library(caper)
  library(optparse)
})

## ------------------- Command-line arguments ------------------- ##
option_list <- list(
  make_option(c("--col"), type = "character", help = "Column name (KO) to process")
)
opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$col)) {
  stop("Please specify a KO column using --col argument, e.g. Rscript run_phylod.R --col K00164")
}

col <- opt$col
message("▶ Running phylo.d for KO: ", col)

## ------------------- Load data ------------------- ##
# 请修改为你自己的路径
load("/mnt/ivy/thliao/project/ML_oxygen/training_sets/phylo_D_results/scripts/phylo_d_used.RData")

## ------------------- Run phylo.d ------------------- ##
if (!(col %in% colnames(keggbin_sub))) {
  stop(paste("Column", col, "not found in keggbin_sub"))
}

x <- keggbin_sub[[col]]
names(x) <- rownames(keggbin_sub)
df <- data.frame(species = names(x), trait = x)

result <- tryCatch({
  comp_data <- comparative.data(tree_am_root, df, names.col = "species", vcv = TRUE)
  ph <- phylo.d(comp_data, binvar = trait, permut = 1000)
  data.frame(
    trait = col,
    DEstimate = ph$DEstimate[["Obs"]],
    pval0 = ph$Pval0,
    pval1 = ph$Pval1
  )
}, error = function(e) {
  message("⚠️ Error in ", col, ": ", e$message)
  data.frame(trait = col, DEstimate = NA, pval0 = NA, pval1 = NA)
})

## ------------------- Save results ------------------- ##
outfile <- paste0("/mnt/ivy/thliao/project/ML_oxygen/training_sets/phylo_D_results/phylod_o/phyloD_", col, ".csv")
write.csv(result, outfile, row.names = FALSE)
message("✅ Result saved to ", outfile)
