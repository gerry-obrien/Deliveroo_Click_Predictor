#Deliveroo Ad Click Prediction Project, Alessandro Ivashkevich, Artur Garipov, Modhura Das, Gerry O'Brien, Anirudh Susarla
#Objective: Predict click/no click for ad campaign based on consumer profile and order data

# Load all necessary libraries
# Install packages if needed (uncomment to install)
# install.packages(c(
#   "naniar", "plot.matrix", "tidyverse", "RColorBrewer", "ggpubr",
#   "caret", "skimr", "Hmisc", "ggsci", "dplyr", "gridExtra",
#   "corrplot", "rpart", "rpart.plot", "rattle", "xgboost",
#   "pscl", "pROC", "magrittr", "class", "nnet", "grid",
#   "plotly", "glmnet", "randomForest", "ISLR2", "MASS",
#   "mvtnorm", "factoextra", "cluster", "mclust", "GGally",
#   "ggplot2", "dendextend", "tree"
# ))

# Load libraries
library(naniar)
library(plot.matrix)
library(tidyverse)
library(RColorBrewer)
library(ggpubr)
library(caret)
library(skimr)
library(Hmisc)
library(ggsci)
library(dplyr)
library(gridExtra)
library(corrplot)
library(rpart)
library(rpart.plot)
library(rattle)
library(xgboost)
library(pscl)
library(pROC)
library(magrittr)
library(class)
library(nnet)
library(grid)
library(plotly)
library(glmnet)
library(randomForest)
library(ISLR2)
library(MASS)
library(mvtnorm)
library(factoextra)
library(cluster)
library(mclust)
library(GGally)
library(ggplot2)
library(dendextend)
library(tree)

# Load the dataset
load('/Users/alessandro_iva/Desktop/Marketing/DeliveryAdClick.RData')

# The dataset contains:
# - ClickTraining: 18000 observations with consumer click data
# - ClickPrediction: 2000 consumer profiles to predict click/no click

# Examine the structure
str(ClickTraining)
str(ClickPrediction)

# Check for null values in the training data
is.null <- any(is.na(ClickTraining))
cat("Has null values:", is.null, "\n")

# Count NAs per column
if(is.null){
  cat("\nNA counts per column:\n")
  print(colSums(is.na(ClickTraining)))
  
  # Remove rows with NA values
  cat("\nOriginal dataset size:", nrow(ClickTraining), "\n")
  ClickTraining <- na.omit(ClickTraining)
  cat("After removing NAs:", nrow(ClickTraining), "\n")
}

# Check for duplicates
has_duplicates <- any(duplicated(ClickTraining))
cat("Has duplicates:", has_duplicates, "\n")

# Summary statistics
summary(ClickTraining)

# Convert target variable to factor
ClickTraining$Clicks_Conversion <- as.factor(ClickTraining$Clicks_Conversion)

# Convert categorical variables to factors if not already
ClickTraining$Region <- as.factor(ClickTraining$Region)
ClickTraining$Carrier <- as.factor(ClickTraining$Carrier)
ClickTraining$Social_Network <- as.factor(ClickTraining$Social_Network)
ClickTraining$Restaurant_Type <- as.factor(ClickTraining$Restaurant_Type)

# EXPLORATORY DATA ANALYSIS

# 1. Check the click conversion rate (target variable distribution)
conversion_rate <- table(ClickTraining$Clicks_Conversion) / nrow(ClickTraining)
cat("\nConversion Rate:\n")
print(conversion_rate)

# Pie Chart: CLICK CONVERSION RATE
tab <- as.data.frame(table(ClickTraining$Clicks_Conversion))
slices <- c(tab[1,2], tab[2,2]) 
lbls <- c("No Click", "Click")
pct <- round(slices/sum(slices)*100, digits = 2)
lbls <- paste(lbls, pct)
lbls <- paste(lbls, "%", sep="")
pie(slices, labels = lbls, col=rainbow(length(lbls)), angle = 90,
    main="Percentage of Click Conversion")

# 2. Analyze numeric variables
ClickTraining_num <- select_if(ClickTraining, is.numeric)

# Compute mean, median and standard deviation
means_CT <- colMeans(ClickTraining_num)
median_CT <- apply(X = ClickTraining_num, MARGIN = 2, FUN = median)
sd_CT <- apply(X = ClickTraining_num, MARGIN = 2, FUN = sd)

cat("\nMeans:\n")
print(means_CT)
cat("\nMedians:\n")
print(median_CT)
cat("\nStandard Deviations:\n")
print(sd_CT)

# Plot distributions of numeric variables
par(mfrow = c(2,2), mar = c(4,4,3,1))
for(i in 1:ncol(ClickTraining_num)){
  hist(ClickTraining_num[,i], freq = F, main = names(ClickTraining_num)[i],
       col = rgb(.7,.7,.7), border = "white", xlab = "")
  abline(v = means_CT[i], lwd = 2)
  abline(v = median_CT[i], lwd = 2, col = rgb(.7,0,0))
  legend("topright", c("Mean", "Median"), lwd = 2, col = c(1, rgb(.7,0,0)),
         cex = .8, bty = "n")
}
dev.off()

# 3. Categorical variables analysis
# Region vs Click Conversion
table(ClickTraining$Region, ClickTraining$Clicks_Conversion)
plot(table(ClickTraining$Region, ClickTraining$Clicks_Conversion), 
     col = c("blue", "yellow"), main = "Region vs Click Conversion")

# Carrier vs Click Conversion
table(ClickTraining$Carrier, ClickTraining$Clicks_Conversion)
plot(table(ClickTraining$Carrier, ClickTraining$Clicks_Conversion), 
     col = c("blue", "yellow"), main = "Carrier vs Click Conversion")

# Social Network vs Click Conversion
table(ClickTraining$Social_Network, ClickTraining$Clicks_Conversion)
plot(table(ClickTraining$Social_Network, ClickTraining$Clicks_Conversion), 
     col = c("blue", "yellow"), main = "Social Network vs Click Conversion")

# Restaurant Type vs Click Conversion
table(ClickTraining$Restaurant_Type, ClickTraining$Clicks_Conversion)
plot(table(ClickTraining$Restaurant_Type, ClickTraining$Clicks_Conversion), 
     col = c("blue", "yellow"), main = "Restaurant Type vs Click Conversion")

# Weekday vs Click Conversion
table(ClickTraining$Weekday, ClickTraining$Clicks_Conversion)
plot(table(ClickTraining$Weekday, ClickTraining$Clicks_Conversion), 
     col = c("blue", "yellow"), main = "Weekday vs Click Conversion")

# 4. Histograms with Click/No Click comparison
p1 <- ClickTraining %>% 
  group_by(Clicks_Conversion, Region) %>% 
  tally() %>% 
  mutate(prop=n/sum(n)) %>% 
  ggplot(aes(x=Region, y=prop, fill=Clicks_Conversion)) + 
  geom_col(position="dodge") + 
  scale_fill_jama() + 
  labs(y="proportion") + 
  theme_light() + 
  theme(legend.position="bottom", axis.text.x = element_text(angle = 45, hjust = 1))

p2 <- ClickTraining %>% 
  group_by(Clicks_Conversion, Carrier) %>% 
  tally() %>% 
  mutate(prop=n/sum(n)) %>% 
  ggplot(aes(x=Carrier, y=prop, fill=Clicks_Conversion)) + 
  geom_col(position="dodge") + 
  scale_fill_jama() + 
  labs(y="proportion") + 
  theme_light() + 
  theme(legend.position="bottom")

p3 <- ClickTraining %>% 
  group_by(Clicks_Conversion, Social_Network) %>% 
  tally() %>% 
  mutate(prop=n/sum(n)) %>% 
  ggplot(aes(x=Social_Network, y=prop, fill=Clicks_Conversion)) + 
  geom_col(position="dodge") + 
  scale_fill_jama() + 
  labs(y="proportion") + 
  theme_light() + 
  theme(legend.position="bottom")

p4 <- ClickTraining %>% 
  group_by(Clicks_Conversion, Restaurant_Type) %>% 
  tally() %>% 
  mutate(prop=n/sum(n)) %>% 
  ggplot(aes(x=Restaurant_Type, y=prop, fill=Clicks_Conversion)) + 
  geom_col(position="dodge") + 
  scale_fill_jama() + 
  labs(y="proportion") + 
  theme_light() + 
  theme(legend.position="bottom")

grid.arrange(p1, p2, ncol=2, nrow=1)
grid.arrange(p3, p4, ncol=2, nrow=1)

# 5. Density plots for numeric variables
p5 <- ClickTraining %>% 
  ggplot(aes(x=Daytime, fill=Clicks_Conversion)) + 
  geom_density(alpha=0.6) + 
  scale_fill_jama() + 
  theme_light()

p6 <- ClickTraining %>% 
  ggplot(aes(x=Time_On_Previous_Website, fill=Clicks_Conversion)) + 
  geom_density(alpha=0.6) + 
  scale_fill_jama() + 
  theme_light()

p7 <- ClickTraining %>% 
  ggplot(aes(x=Number_of_Previous_Orders, fill=Clicks_Conversion)) + 
  geom_density(alpha=0.6) + 
  scale_fill_jama() + 
  theme_light()

grid.arrange(p5, p6, p7, ncol=1, nrow=3)

# 6. Box plots for numeric variables
boxplot(ClickTraining$Daytime ~ ClickTraining$Clicks_Conversion, 
        data = ClickTraining, col = "red",
        xlab = "Click Conversion", ylab = "Daytime")

boxplot(ClickTraining$Time_On_Previous_Website ~ ClickTraining$Clicks_Conversion, 
        data = ClickTraining, col = "red",
        xlab = "Click Conversion", ylab = "Time On Previous Website")

boxplot(ClickTraining$Number_of_Previous_Orders ~ ClickTraining$Clicks_Conversion, 
        data = ClickTraining, col = "red",
        xlab = "Click Conversion", ylab = "Number of Previous Orders")

dev.off()

# CORRELATION ANALYSIS
# Convert categorical variables to numeric for correlation analysis
ClickTraining_corr <- ClickTraining
ClickTraining_corr$Region <- as.numeric(ClickTraining_corr$Region)
ClickTraining_corr$Carrier <- as.numeric(ClickTraining_corr$Carrier)
ClickTraining_corr$Weekday <- as.numeric(ClickTraining_corr$Weekday)
ClickTraining_corr$Social_Network <- as.numeric(ClickTraining_corr$Social_Network)
ClickTraining_corr$Restaurant_Type <- as.numeric(ClickTraining_corr$Restaurant_Type)
ClickTraining_corr$Clicks_Conversion <- as.numeric(ClickTraining_corr$Clicks_Conversion)

# Compute correlation matrix
res_corr <- cor(ClickTraining_corr)
corrplot(res_corr, type="lower", tl.col="#636363", tl.cex=0.7)

# DATA PREPROCESSING FOR MODELING

# Create dummy variables for categorical features (one-hot encoding)
# This is needed for many ML algorithms

# For now, let's create a processed dataset
ClickTraining_processed <- ClickTraining

# Convert Clicks_Conversion back to 0/1 numeric for some models
ClickTraining_processed$Clicks_Conversion_numeric <- as.numeric(as.character(ClickTraining$Clicks_Conversion))

# Split into training and testing sets (70/30 split)
set.seed(123)
trainIndex <- createDataPartition(ClickTraining_processed$Clicks_Conversion, p=0.70, list=FALSE)
train_data <- ClickTraining_processed[trainIndex, ]
test_data <- ClickTraining_processed[-trainIndex, ]

cat("\nTraining set size:", nrow(train_data), "\n")
cat("Testing set size:", nrow(test_data), "\n")

#########################################
# MODEL 1: K-NEAREST NEIGHBORS (KNN)
#########################################

cat("\n=== KNN MODEL ===\n")

# KNN with 10-fold cross validation
set.seed(1)
fit.knn <- train(Clicks_Conversion ~ Daytime + Time_On_Previous_Website + 
                   Number_of_Previous_Orders + Region + Carrier + 
                   Social_Network + Restaurant_Type + Weekday,
                 data=train_data, 
                 method="knn",
                 tuneGrid = data.frame(k = 1:50),
                 metric="Accuracy",
                 trControl = trainControl(method="cv", number=10))

knn.k_best <- fit.knn$bestTune
cat("Best k:", knn.k_best$k, "\n")
print(fit.knn)
plot(fit.knn)

# Predictions on test set
prediction_knn <- predict(fit.knn, newdata = test_data)
cf_knn <- confusionMatrix(prediction_knn, test_data$Clicks_Conversion)
print(cf_knn)

# ROC curve for KNN
roc_knn <- pROC::roc(test_data$Clicks_Conversion,
                     as.numeric(prediction_knn),
                     plot = TRUE,
                     col = "midnightblue",
                     lwd = 3,
                     auc.polygon = TRUE,
                     auc.polygon.col = "lightblue",
                     print.auc = TRUE,
                     main = "ROC Curve - KNN")

cat("KNN AUC:", roc_knn$auc, "\n")

#########################################
# MODEL 2: LOGISTIC REGRESSION
#########################################

cat("\n=== LOGISTIC REGRESSION MODEL ===\n")

# Fit logistic regression model
logit.model <- glm(Clicks_Conversion ~ Region + Daytime + Carrier + 
                     Time_On_Previous_Website + Weekday + Social_Network + 
                     Number_of_Previous_Orders + Restaurant_Type, 
                   family = "binomial", data = train_data)

summary(logit.model)

# Find optimal threshold
best_t <- 0
best_score <- 0

for(t in seq(0.01, 0.98, by = 0.01)){
  probs <- predict(logit.model, train_data, type = "response")
  logit.pred <- rep(0, length(train_data$Clicks_Conversion))
  logit.pred[probs > t] <- 1
  logit.pred <- as.factor(logit.pred)
  
  cm <- confusionMatrix(logit.pred, train_data$Clicks_Conversion)
  sum_values <- cm$overall['Accuracy'] + cm$byClass['Sensitivity'] + cm$byClass['Specificity']
  
  if(sum_values > best_score){
    best_score <- sum_values
    best_t <- t
  }
}

cat("Best threshold:", best_t, "\n")

# Predictions on test set with optimal threshold
probs_test <- predict(logit.model, test_data, type = "response")
logit.pred_test <- rep(0, length(test_data$Clicks_Conversion))
logit.pred_test[probs_test > best_t] <- 1
logit.pred_test <- as.factor(logit.pred_test)

# Confusion matrix
cm_logit <- confusionMatrix(logit.pred_test, test_data$Clicks_Conversion)
print(cm_logit)

# ROC curve
roc_logit <- pROC::roc(train_data$Clicks_Conversion,
                       logit.model$fitted.values,
                       plot = TRUE,
                       col = "midnightblue",
                       lwd = 3,
                       auc.polygon = TRUE,
                       auc.polygon.col = "lightblue",
                       print.auc = TRUE,
                       main = "ROC Curve - Logistic Regression")

cat("Logistic Regression AUC:", roc_logit$auc, "\n")

#########################################
# MODEL 3: AIC AND BIC STEP SELECTION
#########################################

cat("\n=== AIC/BIC STEP SELECTION ===\n")

# Full model
logit_fit_full <- glm(Clicks_Conversion ~ Region + Daytime + Carrier + 
                        Time_On_Previous_Website + Weekday + Social_Network + 
                        Number_of_Previous_Orders + Restaurant_Type,
                      family = "binomial",
                      data = train_data)

# AIC - Forward selection
logit_fit_aic_forward <- step(glm(Clicks_Conversion ~ 1,
                                  family = "binomial",
                                  data = train_data),
                               scope = formula(logit_fit_full),
                               direction = "forward",
                               trace = 0)

# AIC - Backward selection
logit_fit_aic_backward <- step(logit_fit_full,
                                direction = "backward",
                                trace = 0)

# BIC - Forward selection
logit_fit_bic_forward <- step(glm(Clicks_Conversion ~ 1,
                                  family = "binomial",
                                  data = train_data),
                               scope = formula(logit_fit_full),
                               direction = "forward",
                               k = log(nrow(train_data)),
                               trace = 0)

# BIC - Backward selection
logit_fit_bic_backward <- step(logit_fit_full,
                                direction = "backward",
                                k = log(nrow(train_data)),
                                trace = 0)

cat("\nAIC Forward - Selected variables:\n")
print(names(coef(logit_fit_aic_forward)))

cat("\nBIC Forward - Selected variables:\n")
print(names(coef(logit_fit_bic_forward)))

# Predictions with AIC model
tt <- 0.5
prob_out_aic <- predict(logit_fit_aic_forward,
                        newdata = test_data,
                        type = "response")
pred_out_aic <- as.factor(ifelse(prob_out_aic > tt, 1, 0))

# Predictions with BIC model
prob_out_bic <- predict(logit_fit_bic_forward,
                        newdata = test_data,
                        type = "response")
pred_out_bic <- as.factor(ifelse(prob_out_bic > tt, 1, 0))

# Confusion matrices
cm_aic <- confusionMatrix(pred_out_aic, test_data$Clicks_Conversion)
cm_bic <- confusionMatrix(pred_out_bic, test_data$Clicks_Conversion)

cat("\nAIC Model Accuracy:", cm_aic$overall['Accuracy'], "\n")
cat("BIC Model Accuracy:", cm_bic$overall['Accuracy'], "\n")

# ROC curves
roc_aic <- pROC::roc(test_data$Clicks_Conversion,
                     prob_out_aic,
                     plot = TRUE,
                     col = "midnightblue",
                     lwd = 3,
                     auc.polygon = TRUE,
                     auc.polygon.col = "lightblue",
                     print.auc = TRUE,
                     main = "ROC Curve (AIC)")

roc_bic <- pROC::roc(test_data$Clicks_Conversion,
                     prob_out_bic,
                     plot = TRUE,
                     col = "midnightblue",
                     lwd = 3,
                     auc.polygon = TRUE,
                     auc.polygon.col = "lightblue",
                     print.auc = TRUE,
                     main = "ROC Curve (BIC)")

cat("AIC AUC:", roc_aic$auc, "\n")
cat("BIC AUC:", roc_bic$auc, "\n")

#########################################
# MODEL 4: LASSO REGRESSION
#########################################

cat("\n=== LASSO REGRESSION ===\n")

# Prepare data for LASSO
# Create model matrix (handles categorical variables automatically)
x_train <- model.matrix(Clicks_Conversion ~ Region + Daytime + Carrier + 
                          Time_On_Previous_Website + Weekday + Social_Network + 
                          Number_of_Previous_Orders + Restaurant_Type, 
                        data = train_data)[, -1]

y_train <- as.numeric(as.character(train_data$Clicks_Conversion))

x_test <- model.matrix(Clicks_Conversion ~ Region + Daytime + Carrier + 
                         Time_On_Previous_Website + Weekday + Social_Network + 
                         Number_of_Previous_Orders + Restaurant_Type, 
                       data = test_data)[, -1]

y_test <- as.numeric(as.character(test_data$Clicks_Conversion))

# Perform k-fold cross-validation to find optimal lambda
cv_lasso <- cv.glmnet(x_train, y_train, alpha = 1, family = "binomial")

# Find optimal lambda
best_lambda <- cv_lasso$lambda.min
cat("Best lambda:", best_lambda, "\n")

# Plot lambda vs MSE
plot(cv_lasso)

# Fit best model
best_lasso <- glmnet(x_train, y_train, alpha = 1, lambda = best_lambda, family = "binomial")

# Predictions
lasso_predictions <- predict(best_lasso, newx = x_test, type = "response")

# ROC curve
roc_lasso <- pROC::roc(y_test,
                       as.numeric(lasso_predictions),
                       plot = TRUE,
                       col = "midnightblue",
                       lwd = 3,
                       auc.polygon = TRUE,
                       auc.polygon.col = "lightblue",
                       print.auc = TRUE,
                       main = "ROC Curve - LASSO")

cat("LASSO AUC:", roc_lasso$auc, "\n")

#########################################
# MODEL 5: DECISION TREE
#########################################

cat("\n=== DECISION TREE MODEL ===\n")

# Fit decision tree
tree_model <- tree(Clicks_Conversion ~ Region + Daytime + Carrier + 
                     Time_On_Previous_Website + Weekday + Social_Network + 
                     Number_of_Previous_Orders + Restaurant_Type, 
                   data = train_data)

summary(tree_model)

# Plot tree
plot(tree_model)
text(tree_model, pretty=0)

# Predictions on test set
predict_tree <- predict(tree_model, test_data, type = "class")
cm_tree <- confusionMatrix(predict_tree, test_data$Clicks_Conversion)
print(cm_tree)

cat("Decision Tree Accuracy:", mean(predict_tree == test_data$Clicks_Conversion), "\n")

# Cross-validation to find optimal tree size
set.seed(1000)
cv_tree <- cv.tree(object = tree_model, FUN = prune.misclass)
plot(x = cv_tree$size, y = cv_tree$dev, type = "b",
     main = "Cross-validation for Tree Size",
     xlab = "Tree Size", ylab = "Deviance")

# Prune tree if beneficial
optimal_size <- cv_tree$size[which.min(cv_tree$dev)]
cat("Optimal tree size:", optimal_size, "\n")

if(optimal_size < length(tree_model$frame$var)){
  tree_model_pruned <- prune.misclass(tree_model, best = optimal_size)
  plot(tree_model_pruned)
  text(tree_model_pruned, pretty=0)
  
  predict_tree_pruned <- predict(tree_model_pruned, test_data, type = "class")
  cm_tree_pruned <- confusionMatrix(predict_tree_pruned, test_data$Clicks_Conversion)
  print(cm_tree_pruned)
}

# ROC curve
roc_tree <- pROC::roc(response = test_data$Clicks_Conversion,
                      predictor = as.numeric(predict_tree),
                      plot = TRUE,
                      col = "midnightblue",
                      lwd = 3,
                      auc.polygon = TRUE,
                      auc.polygon.col = "lightblue",
                      print.auc = TRUE,
                      main = "ROC Curve - Decision Tree")

cat("Decision Tree AUC:", roc_tree$auc, "\n")

#########################################
# MODEL 6: RANDOM FOREST
#########################################

cat("\n=== RANDOM FOREST MODEL ===\n")

# Fit random forest
set.seed(123)
tr_cont <- trainControl(method="cv", number=10)
rf_model <- randomForest(Clicks_Conversion ~ Region + Daytime + Carrier + 
                           Time_On_Previous_Website + Weekday + Social_Network + 
                           Number_of_Previous_Orders + Restaurant_Type,
                         data = train_data,
                         importance = TRUE,
                         ntree = 500)

# Variable importance
importance_rf <- importance(rf_model)
importance_sorted <- importance_rf[order(importance_rf[, "MeanDecreaseGini"], decreasing = TRUE), ]
print(importance_sorted)

# Plot variable importance
barplot(importance_sorted[, "MeanDecreaseGini"],
        horiz = TRUE,
        names.arg = rownames(importance_sorted),
        main = "Variable Importance Plot - Random Forest",
        xlab = "Mean Decrease Gini",
        ylab = "Variables",
        col = "blue",
        las = 1,
        cex.names = 0.7)

# Predictions
pred_rf <- predict(rf_model, test_data, type = "prob")

# OOB error rate
oob_error_rate <- mean(rf_model$err.rate[, "OOB"])
cat("OOB Error Rate:", oob_error_rate, "\n")

# ROC curve
roc_rf <- pROC::roc(test_data$Clicks_Conversion, pred_rf[, 2])
cat("Random Forest AUC:", auc(roc_rf), "\n")

plot(roc_rf, col = "midnightblue", main = "ROC Curve - Random Forest", 
     auc.polygon = TRUE, auc.polygon.col = "lightblue", 
     print.auc = TRUE, lwd = 3)

#########################################
# MODEL 7: XGBOOST
#########################################

cat("\n=== XGBOOST MODEL ===\n")

# Prepare data matrices
xgb_train <- xgb.DMatrix(data = x_train, label = y_train)
xgb_test <- xgb.DMatrix(data = x_test, label = y_test)
watchlist <- list(train = xgb_train, test = xgb_test)

# Find optimal number of rounds
xgb_cv <- xgb.train(data = xgb_train, 
                    max.depth = 3, 
                    watchlist = watchlist, 
                    nrounds = 200,
                    objective = "binary:logistic",
                    verbose = 0)

# Fit final XGBoost model
# Use fewer rounds to prevent overfitting
params <- list(
  max_depth = 3,
  objective = "binary:logistic",
  eval_metric = "auc"
)

final_xgb <- xgb.train(params = params,
                       data = xgb_train, 
                       nrounds = 100, 
                       verbose = 0)

# Predictions
pred_y_train_xgb <- predict(final_xgb, xgb_train)
pred_y_test_xgb <- predict(final_xgb, xgb_test)

# ROC curve
roc_xgb <- pROC::roc(y_test,
                     pred_y_test_xgb,
                     plot = TRUE,
                     col = "midnightblue",
                     lwd = 3,
                     auc.polygon = TRUE,
                     auc.polygon.col = "lightblue",
                     print.auc = TRUE,
                     main = "ROC Curve - XGBoost")

cat("XGBoost AUC:", roc_xgb$auc, "\n")

# Feature importance for XGBoost
importance_matrix <- xgb.importance(feature_names = colnames(x_train), model = final_xgb)
cat("\nXGBoost Feature Importance:\n")
print(importance_matrix)

# Plot feature importance
if(nrow(importance_matrix) > 0){
  xgb.plot.importance(importance_matrix, top_n = min(10, nrow(importance_matrix)))
}

#########################################
# MODEL COMPARISON
#########################################

cat("\n=== MODEL COMPARISON ===\n")
cat("KNN AUC:", roc_knn$auc, "\n")
cat("Logistic Regression AUC:", roc_logit$auc, "\n")
cat("AIC Model AUC:", roc_aic$auc, "\n")
cat("BIC Model AUC:", roc_bic$auc, "\n")
cat("LASSO AUC:", roc_lasso$auc, "\n")
cat("Decision Tree AUC:", roc_tree$auc, "\n")
cat("Random Forest AUC:", auc(roc_rf), "\n")
cat("XGBoost AUC:", roc_xgb$auc, "\n")








#########################################
# CLUSTERING ANALYSIS
#########################################

cat("\n=== CLUSTERING ANALYSIS ===\n")

# Prepare data for clustering (remove target variable)
ClickTraining_cluster <- ClickTraining_corr[, -which(names(ClickTraining_corr) == "Clicks_Conversion")]

# Scale the data
ClickTraining_scaled <- scale(ClickTraining_cluster)
ClickTraining_scaled <- data.frame(ClickTraining_scaled)

# Find optimal number of clusters using WSS method
cat("Finding optimal number of clusters...\n")
fviz_nbclust(ClickTraining_scaled[sample(1:nrow(ClickTraining_scaled), 1000), ], 
             kmeans, method = 'wss', k.max = 10)

# Find optimal number using Silhouette method
fviz_nbclust(ClickTraining_scaled[sample(1:nrow(ClickTraining_scaled), 1000), ], 
             kmeans, method = 'silhouette', k.max = 10)

# Perform K-means clustering with optimal k (let's try k=3)
set.seed(123)
km_model <- kmeans(ClickTraining_scaled, centers = 3, nstart = 100, iter.max = 100)
clusters_km <- km_model$cluster

cat("K-means clustering completed. Cluster sizes:\n")
print(table(clusters_km))

# Visualize clusters
fviz_cluster(km_model, ClickTraining_scaled, 
             geom = c("point"), 
             ellipse.type = "norm", 
             pointsize = 0.5) +
  theme_minimal() +
  labs(title = "K-means Clustering")

# Add cluster assignments to original data
ClickTraining$cluster_km <- factor(km_model$cluster)

# See how click conversion is distributed among clusters
ggplot(ClickTraining, aes(fill = Clicks_Conversion, x = cluster_km)) +
  geom_bar(position = "dodge") +
  ggtitle("Click Conversion Distribution Among K-means Clusters") +
  xlab("Cluster") +
  ylab("Number of Consumers") +
  theme_classic()

table(ClickTraining$cluster_km, ClickTraining$Clicks_Conversion)

# Hierarchical Clustering
cat("\n=== HIERARCHICAL CLUSTERING ===\n")

# Use a sample for hierarchical clustering (faster)
sample_size <- min(2000, nrow(ClickTraining_scaled))
sample_idx <- sample(1:nrow(ClickTraining_scaled), sample_size)
ClickTraining_sample <- ClickTraining_scaled[sample_idx, ]

# Euclidean distance
dist_eucl <- factoextra::get_dist(ClickTraining_sample, method = "euclidean")

# Find optimal k using WSS
fviz_nbclust(x = ClickTraining_sample, 
             FUNcluster = factoextra::hcut,
             diss = dist_eucl,
             method = "wss",
             k.max = 10)

# Find optimal k using Silhouette
fviz_nbclust(x = ClickTraining_sample, 
             FUNcluster = factoextra::hcut,
             diss = dist_eucl,
             method = "silhouette",
             k.max = 10)

# Hierarchical clustering with Ward method
hc_ward <- factoextra::hcut(x = dist_eucl, 
                             k = 3,
                             hc_method = "ward.D2")

# Dendrogram
fviz_dend(x = hc_ward, main = "Hierarchical Clustering Dendrogram (Ward)")

# Add hierarchical cluster assignments
ClickTraining$hc_ward <- NA
ClickTraining$hc_ward[sample_idx] <- as.integer(cutree(hc_ward, k = 3))

# Visualize cluster sizes
ggplot(ClickTraining[!is.na(ClickTraining$hc_ward), ], 
       aes(x = hc_ward)) + 
  geom_bar(fill = "#00bfff") +
  xlab("Cluster") +
  ggtitle("Hierarchical Clustering - Cluster Sizes") +
  ylab("Number of Consumers") +
  theme_classic()

# See how click conversion is distributed
ggplot(ClickTraining[!is.na(ClickTraining$hc_ward), ], 
       aes(fill = Clicks_Conversion, x = hc_ward)) +
  geom_bar(position = "dodge") +
  ggtitle("Click Conversion Distribution Among Hierarchical Clusters") +
  xlab("Cluster") +
  ylab("Number of Consumers") +
  theme_classic()

table(ClickTraining$hc_ward[!is.na(ClickTraining$hc_ward)], 
      ClickTraining$Clicks_Conversion[!is.na(ClickTraining$hc_ward)])

# Calculate silhouette scores for comparison
cat("\n=== CLUSTERING EVALUATION ===\n")

# Use a sample for silhouette calculation (for speed)
sample_size_sil <- min(1000, nrow(ClickTraining_scaled))
sample_idx_sil <- sample(1:nrow(ClickTraining_scaled), sample_size_sil)
sil_km <- silhouette(clusters_km[sample_idx_sil], dist(ClickTraining_scaled[sample_idx_sil, ]))
avg_sil_km <- mean(sil_km[, 3])
cat("Average Silhouette Score (K-means):", avg_sil_km, "\n")

# Adjusted Rand Index
ari_km <- adjustedRandIndex(ClickTraining$cluster_km, ClickTraining$Clicks_Conversion)
cat("Adjusted Rand Index (K-means vs Click Conversion):", ari_km, "\n")

#########################################
# PREDICTIONS ON NEW DATA
#########################################

cat("\n=== PREDICTIONS ON CLICKPREDICTION DATASET ===\n")

# Check for NAs in prediction dataset
if(any(is.na(ClickPrediction))){
  cat("NAs found in ClickPrediction dataset:\n")
  print(colSums(is.na(ClickPrediction)))
  cat("\nRemoving rows with NAs...\n")
  ClickPrediction <- na.omit(ClickPrediction)
}

# Prepare ClickPrediction data with same factor levels as training data
ClickPrediction$Region <- factor(ClickPrediction$Region, levels = levels(train_data$Region))
ClickPrediction$Carrier <- factor(ClickPrediction$Carrier, levels = levels(train_data$Carrier))
ClickPrediction$Social_Network <- factor(ClickPrediction$Social_Network, levels = levels(train_data$Social_Network))
ClickPrediction$Restaurant_Type <- factor(ClickPrediction$Restaurant_Type, levels = levels(train_data$Restaurant_Type))
ClickPrediction$Weekday <- factor(ClickPrediction$Weekday, levels = levels(train_data$Weekday))

# Remove any rows with NA factor levels (categories not seen in training)
cat("Rows before factor level filtering:", nrow(ClickPrediction), "\n")
ClickPrediction <- ClickPrediction[complete.cases(ClickPrediction), ]
cat("Rows after factor level filtering:", nrow(ClickPrediction), "\n")

# Make predictions using the best model (e.g., XGBoost or Random Forest)
# Using Random Forest for predictions
# Prepare matrix for XGBoost prediction
# We use model.matrix to create the same one-hot encoded columns as in training
x_prediction <- model.matrix(~ Region + Daytime + Carrier + 
                               Time_On_Previous_Website + Weekday + Social_Network + 
                               Number_of_Previous_Orders + Restaurant_Type, 
                             data = ClickPrediction)[, -1]

# Predict with XGBoost
# XGBoost outputs probabilities, so we apply a 0.5 threshold to get binary class labels
predictions_final <- ifelse(predict(final_xgb, xgb.DMatrix(data = x_prediction)) > 0.5, 1, 0)
# Create results dataframe
results <- data.frame(
  Observation = 1:nrow(ClickPrediction),
  Predicted_Click = predictions_final
)

cat("\nFirst 10 predictions:\n")
print(head(results, 10))

# Summary of predictions
cat("\nPrediction Summary:\n")
print(table(results$Predicted_Click))
cat("Predicted Click Rate:", mean(results$Predicted_Click == 1), "\n")



