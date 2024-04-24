library(InvariantCausalPrediction)
library(pcalg)

setwd("/Users/vnastl/Seafile/My Library/mpi project causal vs noncausal/causal-features/tmp_preprocessed") 

dataset_name <- "diabetes_discrete_5"
data <- read.csv(cat(dataset_name,".csv"), header = TRUE, sep = ",")

# get dataframe of features, target as factors and domains as column
X <- subset(data, select = -c(target, domain))
y <- as.factor(subset(data, select = target))
domains <- subset(data, select = target)

dm <- subset(data, select = -c(domain))

# PC algorithm
V <- colnames(dm)
## define sufficient statistics
suffStat <- list(dm = dm, adaptDF = FALSE)
## estimate CPDAG
pc.D <- pc(suffStat,
            ## independence test: G^2 statistic
            indepTest = disCItest, alpha = 0.01, labels = V, verbose = TRUE)

# ICP algorithm
# icp <- ICP(X,y,domains)