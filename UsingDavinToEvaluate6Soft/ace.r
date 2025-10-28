#!
options <- commandArgs(trailingOnly = TRUE)
library("phytools")
library("phangorn")
library("diversitree")
# setwd('/mnt/storage3/thliao/project/ML_oxygen/testing_sets/GTDB_r214/sequential_dating')


intab = options[1]
intree = options[2]
ofile = options[3]
tree<-read.tree(intree)
data<-read.table(intab,sep='\t',header=TRUE)
data.v<-data[,2]
names(data.v)<-data[,1]
tree$tip.state<-data.v
x<-tree$tip.state
system.time({fitER<-ace(x,tree,type="discrete")})
ace_n <- fitER$lik.anc
ace_n <- as.data.frame(ace_n)
row.names(ace_n)<-tree$node.label
write.table(ace_n,file=ofile,quote=FALSE,sep='\t')



