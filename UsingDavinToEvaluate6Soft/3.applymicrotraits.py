

# conda activate r_env

# R
# export TMPDIR=/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/microtraits/mytmp

# library(microtrait)
# dir.create("/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/microtraits/mytmp", recursive = TRUE)
# protein_files = list.files('/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/prepared_proteins', full.names = T, recursive = T, pattern = ".faa$")
# message("Number of cores:", parallel::detectCores(), "\n")


# library("tictoc")
# tictoc::tic.clearlog()
# tictoc::tic(paste0("Running microtrait for ", length(protein_files), " genomes"))
# microtrait_results = parallel::mclapply(1:length(protein_files),
#                     function(i) {
#                         returnList = extract.traits(protein_files[i], '/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/microtraits/', 
#                                                     save_tempfiles = F, type = 'protein',growthrate_predict=F,optimalT_predict=F)
#                         returnList
#                     },
#                     mc.cores = 10)
# tictoc::toc(log = "TRUE")

# rds_files = unlist(parallel::mclapply(microtrait_results, "[[", "rds_file", mc.cores = 10))
# genomeset_results = make.genomeset.results(rds_files = rds_files,
#                                            ids = sub(".microtrait.rds", "", basename(rds_files)),
#                                            growthrate=F,optimumT=F,
#                                            ncores = 10)
write.csv(genomeset_results$trait_matrixatgranularity3, 
          file = "/mnt/storage3/thliao/project/ML_oxygen/davin_genomes_goinganalysis/microtraits/bac_granularity3.csv", 
          row.names = TRUE)






