#######################
#R code: endocrine profiling with outcome variable = female puberty status

#In this project I establish optimal hyperparameters for the 'xgboost' machine learning algorithm. 
#The model was subsequently applied to the Bergen Growth Study 2 [vekststudien.no] female dataset,
#where outcome variable is puberty stage & dependent variables is anabolic/developmental hormone profile
#SUPERVISED MACHINE LEARNING
#=======================================================================================

install.packages(c("e1071", "caret", "doSNOW", "ipred", "xgboost"))
library(caret)
library(doSNOW)



#=================================================================
# Set up dataframe from the bigger dataframe, eliminating NAs
#=================================================================

Outcome <- Data$Outcome #Outcome variable e.g. Tanner puberty stage
V1 <- Data$V1    #feature/dependent variable#1 e.g. hormone, nmol/L
V2 <- Data$V2    #feature/dependent variable#2 e.g. hormone, nmol/L   
V3 <- Data$V3    #feature/dependent variable#3 e.g. hormone, nmol/L
V4 <- Data$V4    #feature/dependent variable#4 e.g. hormone, nmol/L
V5 <- Data$V5    #feature/dependent variable#5 e.g. hormone, IU/L

keep <- !is.na(Outcome)&!is.na(V1)&!is.na(V2)&!is.na(V3)&!is.na(V4)&!is.na(V5)
Outcome <- Outcome[keep]
V1 <- V1[keep]
V2 <- V2[keep]
V3 <- V3[keep]
V4 <- V4[keep]
V5 <- V5[keep]

MLdata <- data.frame(Outcome, V1, V2, V3, V4, V5)
colnames(MLdata) <- c("Outcome","Hormone1","Hormone2","Hormone3","Hormone4","Hormone5") 

#Annotate variables correctly
str(MLdata)
MLdata$Outcome <- as.factor(MLdata$Outcome)
MLdata$Hormone1 <- as.numeric(MLdata$Hormone1)



#=================================================================
# 75% TRAIN / 25% TEST DATAFRAME PARTITIONING
#=================================================================

library(caret)

indexes <- createDataPartition(MLdata$Outcome,
                               times = 1,
                               p = 0.75,
                               list = FALSE)

train.MLdata <- MLdata[indexes,]
test.MLdata <- MLdata[-indexes,]


# Examine the proportions to ensure no outcome variable over-representation!
prop.table(table(MLdata$Outcome))
prop.table(table(train.MLdata$Outcome))
prop.table(table(test.MLdata$Outcome))


#=================================================================
# Train Model
#=================================================================

#'caret' will perform 10-fold cross validation 3 + grid search for optimal hyperparamter settings

train.control <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 3,
                              search = "grid")


tune.grid <- expand.grid(eta = c(0.05, 0.075, 0.1),
                         nrounds = c(50, 75, 100),
                         max_depth = 6:8,
                         min_child_weight = c(2.0, 2.25, 2.5),
                         colsample_bytree = c(0.3, 0.4, 0.5),
                         gamma = 0,
                         subsample = 1)
View(tune.grid)

cl <- makeCluster(6, type = "SOCK") #number here is how many CPU threads your PC has
registerDoSNOW(cl)


ML.model <- train(Outcome ~ ., 
                  data = train.MLdata,
                  method = "xgbTree",
                  tuneGrid = tune.grid,
                  trControl = train.control)
stopCluster(cl)


# The resulting model for inspection
ML.model


# Make predictions on the TEST set using the optimal xgboost model 

preds.ML <- predict(ML.model, test.MLdata)


# Confusion Matrix to evaluate false positive/negative/accuracy/Kappa
confusionMatrix(preds.ML, test.MLdata$Outcome)

#save model
saveRDS(ML.model, "model.rds")
