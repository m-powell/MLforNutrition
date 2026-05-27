# Install and load the necessary packages

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

## create a subset of train and test data based on only using two features
train_data_scaled_select <- train_data_scaled %>% select(height,weight,extreme_bp)
test_data_scaled_select <- test_data_scaled %>% select(height,weight,extreme_bp)

# scale whole dataset with train_data for plot
data_scaled <- rbind(train_data_scaled,test_data_scaled)

# -----------------Fit the Model------------

## fit model
svm_model <- svm(extreme_bp ~ height + weight, 
                 data = train_data_scaled_select, 
                 kernel = "linear",
                 scale = TRUE,
                 cost = 10)

# Print the model summary
summary(svm_model)

# -----------------Predict on Test Set with Model------------

# Make predictions on the test set
predictions <- predict(svm_model, test_data_scaled_select)

# Calculate performance metrics
confusion <- confusionMatrix(predictions, test_data_scaled_select$extreme_bp, positive="TRUE")

# Print performance metrics
print(confusion)

# -----------------Visualize SVM on data------------

# Define a grid over the feature space for plotting
x_seq <- seq(min(train_data_scaled$height), max(train_data_scaled$height), length = 300)
y_seq <- seq(min(train_data_scaled$weight), max(train_data_scaled$weight), length = 300)
grid <- expand.grid(height = x_seq, weight = y_seq)

# Get the decision values for the grid
decision_values <- predict(svm_model, grid, decision.values = TRUE)
decision <- attr(decision_values, "decision.values")

# Plot the data points
plot(train_data_scaled$height, train_data_scaled$weight, col = as.integer(train_data_scaled$extreme_bp) + 1, pch = 19, xlab = "Height (Scaled)", ylab = "Weight (Scaled)")

# Highlight the support vectors
points(svm_model$SV, col = "blue", pch = 5, cex = 1.5)

# Plot the decision boundary and margins
contour(x_seq, y_seq, matrix(decision, 300, 300), levels = c(-1, 0, 1), 
        label = FALSE, add = TRUE, col = c("red", "black", "red"), lwd = 2, lty = c(2, 1, 2))

# Add legend to the plot
legend("topleft", legend = c("Extreme BP - FALSE", "Extreme BP - TRUE", "Support Vectors", "Decision Boundary", "Margin"), 
       col = c("red", "green", "blue", "black", "red"), pch = c(19, 19, 5, NA, NA), 
       lty = c(NA, NA, NA, 1, 2), cex = 0.8)

title("SVM Predicting Extreme Blood Pressure using Height and Weight")

# -----------------Visualize SVM on data (ggplot)------------

# https://eight2late.wordpress.com/2018/06/06/an-intuitive-introduction-to-support-vector-machines-using-r-part-1/
#build weight vector
w <- t(svm_model$coefs) %*% svm_model$SV
#calculate slope
slope_1 <- -w[1]/w[2]
slope_1
#calculate intercept
intercept_1 <- svm_model$rho/w[2]
intercept_1

p <- ggplot(data=train_data_scaled, aes(x=height,y=weight,colour=extreme_bp)) + geom_point()+ scale_colour_manual(values=c("red","blue"))
#identify support vectors in training set
df_sv <- train_data_scaled[svm_model$index,]
#add layer marking out support vectors with semi-transparent purple blobs
p <- p + geom_point(data=df_sv,aes(x=height,y=weight),colour="purple",size = 4,alpha=0.5, fill = NA, shape = 1) + scale_shape(solid = FALSE)
#display plot
#p
# augment Figure 8 with decision boundary using calculated slope and intercept
p <- p + geom_abline(slope=slope_1, intercept = intercept_1)
#p
#margins are offset 1/w[2] on either side of decision boundary
#p created in earlier code block
p <- p + geom_abline(slope=slope_1,intercept = intercept_1-1/w[2], linetype="dashed")+
  geom_abline(slope=slope_1,intercept = intercept_1+1/w[2], linetype="dashed")
#display plot
p