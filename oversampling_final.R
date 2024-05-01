```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
library(tidyverse)
library(caTools)
library(FSelector)
library(ROSE)#for oversampling
library(e1071)# for SVM
library(party)#Decision Tree 1
library(caret)#confusionMatrix
library(pROC) 
```

**data preparation**
  
  ```{r}
mydata <- read.csv("customer_data.csv", stringsAsFactors = TRUE)

```
#Stage 1 tidying

```{r}
str(mydata)
summary(mydata)
```

```{r}
mydata$Customer_ID <- NULL
mydata$account <- NULL
mydata$spend<-NULL
mydata$visit <- as.factor(mydata$visit)

str(mydata)
#purchase_segment there have NAs 

mydata <- na.omit(mydata)
summary(mydata)
```

#Stage two: Partitioning
```{r}
set.seed(37) 

split <- sample.split(mydata$visit, SplitRatio = 0.7) 

training <- subset(mydata, split == TRUE) 

test <- subset(mydata, split == FALSE)

```

#Stage three: Sampling and Balancing the training data
```{r}
table(training$visit)
prop.table(table(training$visit))

oversampled_training<-ovun.sample(visit~.,data=training,method="over",p=0.5,seed = 1)$data

table(oversampled_training$visit)
prop.table(table(oversampled_training$visit))
```


**Information Gain**
  
  ```{r message=FALSE}

# Use function information.gain to compute information gain values of the attributes
visit_weights <- information.gain(visit~., oversampled_training)

# Print weights
print(visit_weights)

```

```{r}

# Let's save a copy of the weights
df <- visit_weights

# add row names as a column to keep them during ordering
df$visit <- rownames(df)

# Let's sort the weights in decreasing order of information gain values.
# We will use arrange() function 
df <- arrange(df, -attr_importance)

# Plot the weights
barplot(df$attr_importance, names = df$visit, las = 2, ylim = c(0, 0.5))
```

```{r}
#  Use cutoff.biggest.diff() 
cutoff.biggest.diff(visit_weights)

# Filter features where the information gain is not zero
filter(visit_weights, attr_importance > 0)

```

```{r  message=FALSE}

# Use cutoff.k() to find the most informative 12 attributes
filtered_attributes <- cutoff.k(visit_weights, 12)

# Print filtered attributes
print(filtered_attributes)

```

```{r  message=FALSE}

# Select a subset of the dataset by using filtered_attributes 
datamodelling <- oversampled_training[filtered_attributes]#delete the non-related variable

```

```{r  message=FALSE}

# Do not forget to add class column to the filtered dataset for modelling
datamodelling$target <- oversampled_training$visit

# View subsetdata by using head() function  
head(datamodelling,10)
```


# SVM model
```{r}
svm_radial <- svm(target ~ . , data = datamodelling, kernel = "radial", scale = TRUE,probability=TRUE)

summary(svm_radial)

svm_predict = predict(svm_radial, test)
```

**SVM evaluation**
  ```{r}
confusionMatrix(svm_predict,test$visit,positive='1',mode="prec_recall")
```
**SVM model tuning**
  
  ```{r}
# tune() function uses random numbers. Therefore, set a seed 
set.seed(4)

# Find the best cost value among the list (0.5, 1, 1.5, 5) 
tune_out = tune(svm, target ~ ., data = datamodelling, kernel= "radial", scale = TRUE, ranges = list(cost=c(0.5, 1.5)))

```

```{r}

# Save the best model as svm_best
svm_best = tune_out$best.model

```

```{r  message=FALSE}

# Predict the class of the test data 
SVM_tune_predict <- predict(svm_best, test)

# Use confusionMatrix to print the performance of SVM model
confusionMatrix(SVM_tune_predict, test$visit, positive='1', 
                mode = "prec_recall")

```
# Decision Tree

```{r}
# Load tree library
library(tree)

# Load maptree library for plotting
library(maptree)

# Build the decision tree by using tree() function
dectree <- tree(target ~., datamodelling, control = tree.control(nrow(datamodelling), mindev = 0,mincut=15))

# Display the summary of your model and print the model
summary(dectree)

decTree_predict <- predict(dectree, test, type = "class")

```

**Tree evaluation**
  ```{r}
confusionMatrix(decTree_predict,
                test$visit,
                positive='1',
                mode="prec_recall")
```

**prune Decision Tree model**
  ```{r }
# Set the seed
set.seed(6)

# Apply cv.tree function to Dtree
CVresults = cv.tree(dectree, 
                    FUN = prune.misclass)

# Plot the results
# CVresults$size :number of terminal nodes in subtree
# CVresults$dev : number of misclassified samples in subtree

# Let's plot the last 10 values
tree_size = tail(CVresults$size, 10)
misclassifiations = tail(CVresults$dev, 10)


plot(tree_size, 
     misclassifiations/nrow(datamodelling), 
     type = "b",
     xlab = "Tree Size", 
     ylab = "CV Misclassification Rate")

```

```{r }
# Prune the tree
decTree_prune = prune.misclass(dectree, best = 4)#best need to change according to the plot above 

# Check the summary of the pruned tree
summary(decTree_prune)

# Let's use this model for prediction
predict_tree_pruned <- predict(decTree_prune , 
                               test, 
                               type="class")
```

```{r }
# Confusion matrix
confusionMatrix(predict_tree_pruned, 
                test$visit, 
                positive='1', 
                mode = "prec_recall")


```

# Random Forest

```{r}
library(randomForest)

set.seed(1)

# Build Random Forest model and assign it to model_RF
model_RF <- randomForest(target ~ ., datamodelling, importance=T)
varImpPlot(model_RF, main = "Most important features in RF")

# Print model_RF
print(model_RF)

# Check the important features by using importance() function
importance(model_RF)

RF_predict <- predict(model_RF, test)

```

**RF evaluation**
  ```{r}
confusionMatrix(RF_predict,test$visit,positive='1',mode="prec_recall")
```


# Logistic Regression

```{r}
# Build a logistic regression model assign it to LogReg
LogReg <- glm(target ~ ., data = datamodelling, family = "binomial", control = list(maxit=100))

LogReg_predict <- predict(LogReg, test, type = "response")

# Predict the class 
LogReg_class <- ifelse(LogReg_predict > 0.5, 1, 0)

# Save the predictions as factor variables
LogReg_class <- as.factor(LogReg_class)
```

**LogReg evaluation**
  ```{r}
confusionMatrix(LogReg_class, test$visit, 
                positive = "1", mode = "prec_recall") 
```

# Visualise the performances of RF, SVM, SVM_tuned, DT, DT_pruned and LogReg by using ROC and Gain charts
**ROC chart**
  ```{r}
# Obtain class probabilities by using predict() and adding type = "prob" for Random Forest model_RF
prob_RF <- predict(model_RF, test, type = "prob")

# Add probability = TRUE for SVM; model_SVM
SVMpred <- predict(svm_radial, test, probability = T)
# Obtain predicted probabilities for SVM
prob_SVM <- attr(SVMpred, "probabilities")


# Add probability = TRUE for SVM; model_SVM
SVMpred_tuned <- predict(svm_best, test, probability = T)
# Obtain predicted probabilities for SVM
prob_SVM_tuned <- attr(SVMpred_tuned, "probabilities")


prob_DT <- predict(dectree, test)
prob_DT_prune <- predict(decTree_prune, test)
```

```{r}
# Obtain the ROC curve data for logistic regression
ROC_LogReg <- roc(test$visit, LogReg_predict)

#the following two the second argument of ROC formula need to change, may not be right
# SVM
ROC_SVM <- roc(test$visit, prob_SVM[, 2])

ROC_SVM_tuned <- roc(test$visit, prob_SVM_tuned[, 2])

# Random Forest
ROC_RF <- roc(test$visit, prob_RF[, 2])

ROC_DT<- roc(test$visit, prob_DT[, 2])

ROC_DT_pruned<- roc(test$visit, prob_DT_prune[, 2])

```


```{r}
# Plot the ROC curve for Random Forest and SVM
ggroc(list(LogReg = ROC_LogReg, RF = ROC_RF, SVM = ROC_SVM, SVM_tuned = ROC_SVM_tuned, DT=ROC_DT, DT_pruned=ROC_DT_pruned), 
      legacy.axes=TRUE) + 
  xlab("FPR") + ylab("TPR") +
  geom_abline(intercept = 0, slope = 1, # random baseline model
              color = "darkgrey", linetype = "dashed")

```

```{r}
# Calculate the area under the curve (AUC) for LogReg
auc(ROC_LogReg)

# Calculate the area under the curve (AUC) for Random Forest
auc(ROC_RF)

# Calculate the area under the curve (AUC) for SVM
auc(ROC_SVM)

auc(ROC_SVM_tuned)

auc(ROC_DT)

auc(ROC_DT_pruned)
```

**Gain chart**
  ```{r}
library(CustomerScoringMetrics)
GainTable_LogReg <- cumGainsTable(LogReg_predict,  # probabilities
                                  test$visit, # actual class
                                  resolution = 1/100) # percentile intervals,
# default = 1/10 (10%)
# Extract the gain values for Gain chart（Maybe the first argument of the Gain formula need to change 
GainTable_RF <- cumGainsTable(prob_RF[, 2], test$visit, 
                              resolution = 1/100)

GainTable_SVM <- cumGainsTable(prob_SVM[, 2], test$visit, 
                               resolution = 1/100)

GainTable_SVM_tuned <- cumGainsTable(prob_SVM_tuned[, 2], test$visit, 
                                     resolution = 1/100)
GainTable_DT <- cumGainsTable(prob_DT[, 2], test$visit, 
                              resolution = 1/100)
GainTable_DT_pruned <- cumGainsTable(prob_DT_prune[, 2], test$visit, 
                                     resolution = 1/100)
```

```{r}
GainTable.data <- data.frame(Percentage = GainTable_LogReg[,1], LogReg = GainTable_LogReg[,4], RF = GainTable_RF[,4], SVM = GainTable_SVM[,4], SVM_tuned = GainTable_SVM_tuned[,4], DT = GainTable_DT[,4], DT_pruned = GainTable_DT_pruned[,4])

library(reshape2)
GainTable.melt <- melt(GainTable.data, id="Percentage")
colnames(GainTable.melt) <- c("Percentage", "name", "value")

library(ggplot2)
ggplot(GainTable.melt, aes(x=Percentage, y=value, group=name, color=name)) + geom_line() + xlab("Percentage of test instances") + ylab("Percentage of Correct Positive Predictions (True Positive Rate)") + geom_abline(intercept = 0, slope = 1, color = "darkgrey", linetype = "dashed")
```