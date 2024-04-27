# R script to run pc algorithm
###############################################################################
# install packages
###############################################################################
# BiocManager::install("RBGL")
# install.packages("pcalg")
# BiocManager::install("Rgraphviz")

# library(pcalg, lib.loc="/home/vnastl//R/x86_64-pc-linux-gnu-library/4.3")
setwd("/Users/vnastl/Seafile/My Library/mpi project causal vs noncausal/causal-features/tmp_preprocessed")
# setwd("/home/vnastl/causal-features/tmp_preprocessed") # cluster

###############################################################################
# get task and alpha
###############################################################################

args = commandArgs(trailingOnly=TRUE)
task = "unemployment" #args[1] # "diabetes"
disc_task = paste(task,"discrete","5",sep="_")
alpha = 0.01 # args[2] # 0.0001

###############################################################################
# use preprocessed data
###############################################################################
dataset_name <- paste(disc_task,"csv",sep=".")
data <- read.csv(dataset_name, header = TRUE, sep = ",")
data <- sapply(data, as.numeric )
print("sucessfully loaded data")

# PC algorithm
dm <- subset(data, select = -c(domain))
dm <- dm[,apply(dm, MARGIN=2, function(x)(length(unique(x))>1))]
V <- colnames(dm)
## define sufficient statistics
suffStat <- list(dm = dm, adaptDF = FALSE)
## estimate CPDAG
pc_results <- pc(suffStat,
                 ## independence test: G^2 statistic
                 indepTest = disCItest, alpha = alpha, labels = V, verbose = TRUE)

save(pc_results,file=paste(paste(task,"pc","alpha",format(alpha, scientific = FALSE),sep="_"),"RData",sep="."))

# if (require(Rgraphviz)) {
#   ## show estimated CPDAG
#   png(file=paste(paste(task,"pc","alpha",format(alpha, scientific = FALSE),sep="_"),"png",sep="."))
#   plot(pc_results, main = "Estimated CPDAG")
#   dev.off()
# }