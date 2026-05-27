#This code runs Support Vector Machines on the NHANES data to predict extreme blood pressure (binary outcome).

# Load necessary packages-as a best practice label what
# packages are for. This way you have a record of what
# each package does and the role of the package in this
# code.

# -----------------Load required packages------------

# used for feature selection
library(randomForest)
# used for SVM model
library(e1071)
# used for general data manipulation
library(tidyverse)
#for data splitting and feature selection
library(caret)

# -----------------Load data------------

## Read and look at data 
data <- read_csv("C:/Users/hayden.deverill/OneDrive - West Point/Sarah Booth Paper/Data/NHANES Data/Nutrition Data from NHANES.csv")

## Check data headings
head(data)

# -----------------Preprocess data------------

## Ensure all categorical data are changed from characters to factors
data$med_hbp <- as.factor(data$med_hbp)
data$med_chol <- as.factor(data$med_chol)
data$extreme_bp <- as.factor(data$extreme_bp)
data$extreme_waist <- as.factor(data$extreme_waist)
data$extreme_hdl <- as.factor(data$extreme_hdl)
data$extreme_tri <- as.factor(data$extreme_tri)
data$extreme_glu <- as.factor(data$extreme_glu)
data$metsyn <- as.factor(data$metsyn)

## Convert 'gender' to 'sex' to a factor
data$sex <- as.factor(data$gender)

## Remove gender (gender name changed to sex)
data <- data %>% select(-gender)

## Remove any missing data
data <- na.omit(data)
# write_csv(data, "data.csv")

## Sniff check: Display the summary of the selected features
summary(data[, c("age", "bmi", "sex","carb", "fat_total", "protein", "extreme_bp")])

## Another sniff check
data

## Scale and center the numeric data (SVM requires this due to it's distance based optimization decision boundary)
# data <- data %>% mutate_if(is.numeric,scale)

# -----------------Split the data into train/test------------
## Split the data into 80% training and 20% testing
set.seed(123)  # for reproducibility
trainIndex <- createDataPartition(data$bp_sys, p = 0.8, list = FALSE)
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]

# -----------------Standardize the data------------

## Get column names for numeric and non-numeric features
numeric_features <- train_data[sapply(train_data, is.numeric)] %>% colnames()
non_numeric_features <- train_data[sapply(train_data, is.factor)] %>% colnames()

## Find mean and SD of train_data
train_mean <- colMeans(train_data[numeric_features])
train_sd <- apply(train_data[numeric_features], 2, sd)

## scale the train_data using the train_data mean and sd
train_scaled <- scale(train_data[numeric_features], center = train_mean, scale = train_sd)

## scale the test_data using the train_data mean and sd
test_scaled <- scale(test_data[numeric_features], center = train_mean, scale = train_sd)

## create final train and test datasets scaled
### have to combine the numeric and non-numeric columns together
train_data_scaled <- cbind(train_scaled, train_data[non_numeric_features])
test_data_scaled <- cbind(test_scaled, test_data[non_numeric_features])

# -----------------Feature Selection------------
## use feature selection to gain an intuition about which features may be most important
## ensure results are repeatable
set.seed(7)
## prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)
## train the model
model <- train(extreme_bp~., 
               data=data %>% select(-c(bp_sys,bp_di)), #bp_sys and bp_di are correlated to extreme_bp
               method="rf", 
               preProcess = c("center","scale"),
               trControl=control)
## view the model
model
## estimate variable importance
importance <- varImp(model, scale=FALSE)
## summarize importance
print(importance)
## plot importance
plot(importance)

# Use recursive feature elimination to gain intuition about best feature combination

## ensure the results are repeatable
set.seed(7)
## define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
## run the RFE algorithm
results <- rfe(data[,c(1:3, 6:22,24:29)], data$extreme_bp, sizes=c(1:8), rfeControl=control)
## summarize the results
print(results)
## list the chosen features
predictors(results)
## plot the results
plot(results, type=c("g", "o"))

# -----------------Train the SVM Classifier------------

# Here is what the next code chunk means:
## svm - the command from the e1071 package to train a SVM model
## extreme_bp ~ metsyn + height + weight + triglycerides + waist + protein + age
### extreme_bp - the categorical variable I am trying to predict
### height + weight + triglycerides + waist + protein + age - the input variables I am using to predict extreme_bp
##  data = train_data - I am using the training data to train the SVM model on
##  kernel = "linear"- I am using the linear SVM kernel
##  cost = 10 - I am specifying the cost parameter to be 10

## Original Model - Accuracy 55%
# svm_model <- svm(extreme_bp ~ age + bmi + carb + fat_total + protein + sex,
#                          data = train_data,
#                          kernel = "linear",
#                          cost = 10,
#                          scale = FALSE)

## New Model - Accuracy 64%
svm_model <- svm(extreme_bp ~ height + weight + triglycerides + waist + protein + age + glucose, 
                 data = train_data_scaled, 
                 kernel = "linear",
                 scale = TRUE,
                 cost = 10)

# Print the model summary
print(svm_model)

# -----------------Predict on the test data using the SVM model------------

# The next code chunk tests the model on new data. In here, we see
# how to use the model to make predictions.

# Predict on the test data
test_predictions <- predict(svm_model, test_data_scaled)

# Calculate performance metrics
confusion <- confusionMatrix(test_predictions, test_data_scaled$extreme_bp, positive="TRUE")

# Print performance metrics
print(confusion)
