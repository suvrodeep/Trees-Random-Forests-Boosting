#Include packages
library(ggplot2)
library(caret)
library(ROCR)
library(pROC)
library(dummies)
library(stats)
library(class)
library(e1071)
library(tree)
library(randomForest)
library(gdm)
library(gbm)
#
#
#Setting file name
filename <- "data.csv"
#
#
#Reading csv into dataframe and create success class
df <- read.csv(filename, header = TRUE)
#
#
#Creating new variable
df$NPV <- as.vector(df$NPV)
df$AMOUNT_REQUESTED <- as.vector(df$AMOUNT_REQUESTED)
df$PROFITABLE <- ifelse((df$NPV > 0), 1, 0)
df$PROFITABLE <- as.factor(df$PROFITABLE)
head(df)
str(df)
#
#
#Creating factors
#dummy_cl_names <- c("CHK_ACCT", "SAV_ACCT", "HISTORY", "JOB", "TYPE")
df$CHK_ACCT <- as.factor(df$CHK_ACCT)
df$SAV_ACCT <- as.factor(df$SAV_ACCT)
df$HISTORY <- as.factor(df$HISTORY)
df$JOB <- as.factor(df$JOB)
df$TYPE <- as.factor(df$TYPE)
str(df)
#
#
#Sampling the dataset
set.seed(12345)
train <- sample(nrow(df), 0.7*nrow(df))
df_train <- df[train,]
df_test <- df[-train,]
str(df_train)
str(df_test)
#
#
#Run classification tree
credit.tree <- tree(PROFITABLE ~ . -OBS. -CREDIT_EXTENDED -NPV, data = df_train)
credit.tree
plot(credit.tree)
text(credit.tree, pretty = 0)
summary(credit.tree)
df_test.ctree <- predict(credit.tree, df_test, type = "class")
df_test.ctree
#
#
#K fold cross validation for pruning
set.seed(123)
cv.df_train <- cv.tree(credit.tree, FUN = prune.misclass, K = 10)
names(cv.df_train)
cv.df_train
plot(cv.df_train$size, cv.df_train$dev, type = "b")
best_index <- which.min(cv.df_train$dev)
best <- cv.df_train$size[best_index]
credit.prune <- prune.misclass(credit.tree, best = best)
credit.prune
plot(credit.prune)
text(credit.prune, pretty = 0)
summary(credit.prune)
df_test.ptree <- predict(credit.prune, df_test, type = "class")
df_test.ptree
#
#
#Confusion matrices for trees
cm <- confusionMatrix(df_test.ptree, df_test$PROFITABLE)
cm
cm1 <- confusionMatrix(df_test.ctree, df_test$PROFITABLE)
cm1
#
#
#Classify another record
#Setting file name
filename <- "newdata.csv"
#
#
#Reading csv into dataframe and create success class
df1 <- read.csv(filename, header = TRUE)
#
#
#Preparing new data
df1$NPV <- as.numeric(df1$NPV)
df1$AMOUNT_REQUESTED <- as.vector(df1$AMOUNT_REQUESTED)
df1$CREDIT_EXTENDED <- as.numeric(df1$CREDIT_EXTENDED)
df1$OBS. <- as.numeric(df1$OBS.)
df1$CHK_ACCT <- as.factor(df1$CHK_ACCT)
df1$SAV_ACCT <- as.factor(df1$SAV_ACCT)
df1$HISTORY <- as.factor(df1$HISTORY)
df1$JOB <- as.factor(df1$JOB)
df1$TYPE <- as.factor(df1$TYPE)
#
#
#Predict for new record
df1.ctree <- predict(credit.tree, df1, type = "class")
df1.ctree
df1.ctree.prob <- predict(credit.tree, df1, type = "vector")
df1.ctree.prob
#
#
#Pruned tree with 4 terminal nodes
credit.prune1 <- prune.tree(credit.tree, best = 4)
credit.prune1
plot(credit.prune1)
text(credit.prune1, pretty = 0)
summary(credit.prune1)
#
#
#Regression Tree
credit.tree1 <- tree(NPV ~ . -OBS. -CREDIT_EXTENDED -PROFITABLE, data = df_train)
credit.tree1
plot(credit.tree1)
text(credit.tree1, pretty = 0)
summary(credit.tree1)
#
#
#Pruning the regression tree using cross validation
set.seed(123)
cv.reg <- cv.tree(credit.tree1, FUN = prune.tree, K = 10)
names(cv.reg)
cv.reg
plot(cv.reg$size, cv.reg$dev, type = "b")
best_index <- which.min(cv.reg$dev)
best <- cv.reg$size[best_index]
credit.prune2 <- prune.tree(credit.tree1, best = best)
credit.prune2
plot(credit.prune2)
text(credit.prune2, pretty = 0)
summary(credit.prune2)
#
#
#Scoring the datasets
df_train.ptree <- predict(credit.prune2, newdata = df_train)
df_test.ptree <- predict(credit.prune2, newdata = df_test)
#
#
#Number of records at each node
table(df_test.ptree)
#
#
#Number of profitbale accounts in test data
df_test$PRED_NPV_REG <- predict(credit.prune2, newdata = df_test)
df_test$PRED_PROF_RTREE <- ifelse((df_test$PRED_NPV_REG > 0), 1, 0)
no_of_prof <- sum(df_test$PRED_PROF_RTREE)
no_of_prof
#
#
#Average profit per customer
avg_prof <- sum(df_test$PRED_NPV_REG[which(grepl(1, df_test$PRED_PROF_RTREE))])/no_of_prof
avg_prof
#
#
#Overall profit for all customers
overall_profit <- sum(df_test$PRED_NPV_REG[which(grepl(1, df_test$PRED_PROF_RTREE))])
overall_profit
#
#
#Comparison of extending credit
avg_pft_all <- sum(df_test$PRED_NPV_REG)/nrow(df_test)
overall_profit_all <- sum(df_test$PRED_NPV_REG)
avg_pft_all
overall_profit_all
#
#
#Linear regression
reg.model <- lm(NPV ~ . -OBS. -CREDIT_EXTENDED -PROFITABLE, data = df_train)
summary(reg.model)
df_train$PRED_NPV <- predict(reg.model, newdata = df_train)
#
#
#
df2 <- df_train[order(df_train$PRED_NPV),]
net_prof <- data.frame()
for(i in 1:nrow(df2)) {
  cutoff <- df2$PRED_NPV[i]
  net_prof[i,1] <- sum(df2$NPV[i:nrow(df2)])
  net_prof[i,2] <- cutoff
}
max_index <- which.max(net_prof$V1)
cutoff <- net_prof$V2[max_index]
cutoff
#
#
#Applying cutoff to test sample
df_test$PRED_NPV <- predict(reg.model, newdata = df_test)
df_test$PRED_PROF_LM <- ifelse((df_test$PRED_NPV > cutoff), 1, 0)
#
#Number of profitable accounts in test data
no_of_prof <- sum(df_test$PRED_PROF_LM)
no_of_prof
#
#Average profit per customer
avg_prof <- sum(df_test$PRED_NPV[which(grepl(1, df_test$PRED_PROF_LM))])/no_of_prof
avg_prof
#
#Overall profit for all customers
overall_profit <- sum(df_test$PRED_NPV[which(grepl(1, df_test$PRED_PROF_LM))])
overall_profit
#
#
#Random Forest
#Sampling the dataset
set.seed(12345)
train <- sample(nrow(df), 0.7*nrow(df))
df_train <- df[train,]
df_test <- df[-train,]
#
#
rf.model <- randomForest(NPV ~. -OBS. -CREDIT_EXTENDED -PROFITABLE, mtry =20, data = df_train)
summary(rf.model)
df_test$PRED_NPV_RF <- predict(rf.model, df_test)
df_test$PRED_PROF_RF <- ifelse((df_test$PRED_NPV_RF > 0), 1, 0)
prof <- sum(df_test$PRED_NPV_RF[which(grepl(1, df_test$PRED_PROF_RF))])
prof

cm <- confusionMatrix(df_test$PRED_PROF_RF, df_test$PROFITABLE)
cm
#
#
#Boosting
#Sampling the dataset
set.seed(12345)
train <- sample(nrow(df), 0.7*nrow(df))
df_train <- df[train,]
df_test <- df[-train,]
#
#
boost.model <- gbm(NPV ~. -OBS. -CREDIT_EXTENDED -PROFITABLE, data = df_train, distribution = "gaussian", n.trees = 5000, interaction.depth = 4)
summary(boost.model)
df_test$PRED_NPV_BST <- predict(boost.model, newdata = df_test, n.trees = 5000)
df_test$PRED_PROF_BST <- ifelse((df_test$PRED_NPV_BST > 0), 1, 0)
prof <- sum(df_test$PRED_NPV_BST[which(grepl(1, df_test$PRED_PROF_BST))])
prof
cm <- confusionMatrix(df_test$PRED_PROF_BST, df_test$PROFITABLE)
cm 
#
#
#Bagging
#Sampling the dataset
set.seed(12345)
train <- sample(nrow(df), 0.7*nrow(df))
df_train <- df[train,]
df_test <- df[-train,]
#
#
bag.model <- randomForest(NPV ~. -OBS. -CREDIT_EXTENDED -PROFITABLE, mtry = 20, data = df_train, importance = TRUE)
summary(bag.model)
df_test$PRED_NPV_BAG <- predict(bag.model, df_test)
df_test$PRED_PROF_BAG <- ifelse((df_test$PRED_NPV_BAG > 0), 1, 0)
prof <- sum(df_test$PRED_NPV_BAG[which(grepl(1, df_test$PRED_PROF_BAG))])
prof
cm <- confusionMatrix(df_test$PRED_PROF_BAG, df_test$PROFITABLE)
cm 
  

