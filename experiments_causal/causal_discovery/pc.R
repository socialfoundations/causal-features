library(pcalg)

setwd("/Users/vnastl/Seafile/My Library/mpi project causal vs noncausal/causal-features/tmp_preprocessed") 

task = "diabetes"
disc_task = paste(task,"discrete","5",sep="_")

###############################################################################
# use preprocessed data
###############################################################################
dataset_name <- paste(disc_task,"csv",sep=".")
data <- read.csv(dataset_name, header = TRUE, sep = ",")

# PC algorithm
dm <- subset(data, select = -c(domain))
V <- colnames(dm)
## define sufficient statistics
suffStat <- list(dm = dm, adaptDF = FALSE)
## estimate CPDAG
pc_results <- pc(suffStat,
            ## independence test: G^2 statistic
            indepTest = disCItest, alpha = 0.0001, labels = V, verbose = TRUE)