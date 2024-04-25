require(InvariantCausalPrediction)
require(dplyr)
require(forstringr)
require(jsonlite)

setwd("/Users/vnastl/Seafile/My Library/mpi project causal vs noncausal/causal-features/tmp_preprocessed") 
# cluster setwd("/home/vnastl/tmp_preprocessed") 

task = "unemployment"

dataset_name <- paste(task,"csv",sep=".")
data <- read.csv(dataset_name, header = TRUE, sep = ",")

# get dataframe of features, target as factors and domains as column
X <- as.matrix(subset(data, select = -c(target, domain)))
y <- as.factor(data$target)
domains <- data$domain

###############################################################################
# ICP with boosting
###############################################################################
icp_boosting <- ICP(X,y,domains,selection="boosting", alpha=0.05)
save(icp_boosting,file=paste(task,"icp_boost.RData",sep="_"))
summary(icp_boosting)
plot(icp_boosting)

# identify the selected and significant variables
ci <- icp_boosting$ConfInt
if (is.null(ci)){
  selected_ci = numeric()
  selected_names_boosting = character()
  significant_names_boosting = character()
} else{
  zero_columns <- colSums(ci == 0) == nrow(ci)
  selected_ci <- ci[, !zero_columns]
  
  selected_pp_names <- colnames(selected_ci)
  selected_names_boosting <- str_split_i(selected_pp_names, pattern = "_", i=1)
  
  significant_ci <- selected_ci[, (selected_ci[1, ] < 0) | (selected_ci[2, ] > 0)]
  significant_pp_names<- colnames(significant_ci)
  significant_names_boosting <- str_split_i(significant_pp_names, pattern = "_", i=1)
}

# save the results
write.csv(selected_ci, paste(task,"icp_boost_ci.csv",sep="_"))

json_string <- toJSON(selected_names_boosting)
writeLines(json_string, paste(task,"icp_boost_selected.json",sep="_"))

json_string <- toJSON(significant_names_boosting)
writeLines(json_string, paste(task,"icp_boost_significant.json",sep="_"))
###############################################################################
# ICP with lasso
###############################################################################

icp_lasso <- ICP(X,y,domains,selection="lasso", alpha=0.05)
save(icp_lasso,file=paste(task,"icp_lasso.RData",sep="_"))

# # identify the selected variables
# ci <- icp_lasso$ConfInt
# zero_columns <- colSums(ci == 0) == nrow(ci)
# selected_ci <- ci[, !zero_columns]
# selected_pp_names <- colnames(selected_ci)
# 
# selected_names_lasso <- str_split_i(selected_pp_names, pattern = "_", i=1)
# # Save as json file
# json_string <- toJSON(selected_names_lasso)
# writeLines(json_string, paste(task,"icp_lasso_selected.json",sep="_"))


###############################################################################
# idea: suuse discretized data
###############################################################################
# disc_task = paste(task,"discrete","5",sep="_")
# 
# dataset_name <- paste(disc_task,"csv",sep=".")
# data <- read.csv(dataset_name, header = TRUE, sep = ",")
# 
# # get dataframe of features, target as factors and domains as column
# X <- as.matrix(subset(data, select = -c(target, domain)))
# y <- as.factor(data$target)
# domains <- data$domain
# 
# # ICP algorithm
# icp_boosting <- ICP(X,y,domains,selection="boosting")
# save(icp_boosting,file=paste(disc_task,"icp_boost",sep="_"))
# icp_lasso <- ICP(X,y,domains,selection="lasso")
# save(icp_lasso,file=paste(disc_task,"icp_lasso",sep="_"))
