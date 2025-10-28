


# conda activate r_env

# R

export TMPDIR=/mnt/storage4/thliao/ML_oxygen/training_sets/mytmp

library(microtrait)
dir.create("/mnt/storage4/thliao/ML_oxygen/training_sets/mytmp", recursive = TRUE)

genome_file <- "/mnt/storage4/thliao/ML_oxygen/training_sets/GCA_008015785.1.fna"
result = extract.traits(genome_file,
                        out_dir='/mnt/storage4/thliao/ML_oxygen/training_sets/microtraits/')
genomes_files = list.files('/mnt/storage4/thliao/ML_oxygen/training_sets/', full.names = T, recursive = T, pattern = ".fna$")
message("Number of cores:", parallel::detectCores(), "\n")

library("tictoc")
tictoc::tic.clearlog()
tictoc::tic(paste0("Running microtrait for ", length(genomes_files)))
microtrait_results = extract.traits.parallel(genomes_files, 
                                             dirname(genomes_files), 
                                             '/mnt/storage4/thliao/ML_oxygen/training_sets/microtraits/',
                                             ncores = 10)

tictoc::toc(log = "TRUE")


rds_files = unlist(parallel::mclapply(microtrait_results, "[[", "rds_file", mc.cores = 10))


genomeset_results = make.genomeset.results(rds_files = rds_files,
                                           ids = sub(".microtrait.rds", "", basename(rds_files)),
                                           ncores = 1)


write.csv(genomeset_results$trait_matrixatgranularity3, 
          file = "8943bac_granularity3.csv", 
          row.names = TRUE)


