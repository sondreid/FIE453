# Group Presentation
# Student numbers: s164142, s174858

# Theme: Classification using SVM (Support Vector Machine)

# Set WD
#setwd("~/OneDrive - Norges Handelshøyskole/MASTER/FIE453/Group presentation")
#setwd("Desktop/Repositories/FIE453/data")

# Libraries
    require(tidyverse)
    require(magrittr)
    require(caret)
    require(kableExtra)
    require(yardstick)

# Load data
load("merged.Rdata")

# Looking at the first 10 appearence of companies
permno <- merged %>% 
    select(PERMNO) %>% 
    unique() %>% 
    head(10)

# Filtering the data frame containing only the 10 first companies
df <- merged %>% 
    filter(PERMNO %in% permno$PERMNO)

# Replacing the NA's with 0
df[is.na(df)] <- 0

# Retrieving all the columns without all the columns being 0
columns.without.all.zero <- 
    df %>% 
    apply(MARGIN = 2, FUN = function(x) all(x == 0)) %>% 
    as.data.frame() %>% 
    filter(. == FALSE) %>% 
    rownames()

# Data frame only containing useful columns
df %<>% 
    select(all_of(columns.without.all.zero)) 

# Retrieving all the columns with duplicates
columns.with.all.duplicates <- 
    df %>% 
    apply(MARGIN = 2, FUN = function(x) if(length(c(unique(x))) == 1) return(TRUE) else return(FALSE)) %>% 
    as.data.frame() %>% 
    filter(. == TRUE) %>% 
    rownames()

# Data frame only containing useful columns
df %<>% 
    select(!columns.with.all.duplicates) 


### Variable selection ####
# Removing irrelevant or almost perfect correlated columns

non_numeric <- c("RETX",
                 "date",
                 "datadate",
                 "fyearq",
                 "fqtr",
                 "datacqtr",
                 "datafqtr")

non_numeric_strict <- 
               c("RETX",
                 "date",
                 "datadate",
                 "fyearq",
                 "fqtr",
                 "datacqtr",
                 "datafqtr",
                 "aol2q",
                 "tfvaq",
                 "txpq",
                 "doq",
                 "xidoq",
                 "derlltq",
                 "fyr")


df %<>% 
    select(-one_of(non_numeric))


### Add binary label

df %<>% 
    mutate(positive_return = case_when(RET > 0 ~ 1, TRUE ~ 0),
           positive_return = as.factor(positive_return)) %>% 
    select(-RET) 



### Class Balance of positive returns ###
# Is accuracy a good performance measure

df %>% ggplot(aes(x = positive_return)) +
    geom_bar() +
    scale_colour_manual(values = c("black", "orange")) +
    theme_bw() +
    labs(title = "Distribution of positive returns labels",
         x = "Positive Returns 0/1",
         y = "Count")



    
# SVM --------------------------------------------------------------------------
# Size of the train-test split and number of observations in data frame


### Split train and test data
size <- 0.8
n <- nrow(df)
train_indices <- sample(1:n, size = floor(n*size))
train_df <- df[train_indices, ]
test_df  <- df[-train_indices, ]




#
# Performing SVM using the Caret package
#


perform_svm <- function() {
    set.seed(123)
    train_control <- trainControl(method="repeatedcv", number=10, repeats=3, 
                                  savePredictions = T)
    svm_model_all_pca <- caret::train(positive_return~.,
                            method = "svmPoly",
                            data = train_df,
                            trControl  = train_control, 
                            #preProcess = c("pca"),
                            preProcess = c("center", "scale", "pca"),
                            tunelength = 4,
                            allowParallel=TRUE)



    svm_model_all_scaled <- caret::train(positive_return~.,
                            method = "svmPoly",
                            data = train_df,
                            trControl  = train_control, 
                            preProcess = c("center", "scale"),
                            tunelength = 4,
                            allowParallel=TRUE)


    ###### Modelling only few features #####
    subset_train_df <- train_df %>%
        select(c(positive_return, PRC, VOL, vwretd))


    svm_model_subset_pca <- caret::train(positive_return~.,
                            method = "svmPoly",
                            data = subset_train_df,
                            trControl  = train_control, 
                            #preProcess = c("pca"),
                            preProcess = c("center", "scale", "pca"),
                            tunelength = 4)

    svm_model_subset_scaled <- caret::train(positive_return~.,
                            method = "svmPoly",
                            data = subset_train_df,
                            trControl  = train_control, 
                            preProcess = c("center", "scale"),
                            tunelength = 4)

    save(svm_model_all_pca, svm_model_all_scaled, svm_model_subset_pca, svm_model_subset_scaled, file = "svm_model.Rdata")
}
#perform_svm()

load(file = "svm_model.Rdata")



# Predicting with all the variables
preds_svm_model_all_pca <- predict(svm_model_all_pca, newdata = test_df)
preds_svm_model_all_scaled <- predict(svm_model_all_scaled, newdata = test_df)
preds_svm_model_subset_pca <- predict(svm_model_subset_pca, newdata = test_df)
preds_svm_model_subset_scaled <- predict(svm_model_subset_scaled, newdata = test_df)





plot_confusion_matrix <- function(obs, preds) {
    #' Function that plots confusion matrix
    conf_data <- tibble(obs = obs, preds =preds )
    confusion_matrix <- conf_mat(conf_data, truth = "obs", estimate = "preds")
    autoplot(confusion_matrix, type = "heatmap") +
    theme(text = element_text(size = 25))    
}




### Make tables
make_table <- function(obs, preds, model_name) {
    #' Function that produces a table
    table_df <- tibble("Accuracy" = confusionMatrix(preds, obs)[[3]][1], 
                       "Recall" = caret::recall(table(obs, preds)),
                       "Preciscion" = caret::precision(table(obs, preds)))  %>% 
    t()  
    colnames(table_df) <- "Measure"
    table_df  %<>% 
            kbl(caption = model_name)  %>% 
            kable_classic(full_width = F, html_font = "Times New Roman")
    return (table_df)
}

## ALL PCA
plot_confusion_matrix(test_df$positive_return, preds_svm_model_all_pca)
make_table(test_df$positive_return, preds_svm_model_all_pca, "All variables PCA reduced")

## Subset PCA
plot_confusion_matrix(test_df$positive_return, preds_svm_model_subset_pca)
make_table(test_df$positive_return, preds_svm_model_subset_pca, "Subset of variables PCA reduced")


## All scaled
plot_confusion_matrix(test_df$positive_return, preds_svm_model_all_scaled)
make_table(test_df$positive_return, preds_svm_model_all_scaled, "All variables scaled")


## Subset scaled
plot_confusion_matrix(test_df$positive_return, preds_svm_model_subset_scaled)
make_table(test_df$positive_return, preds_svm_model_subset_scaled, "All variables scaled")



library(pROC)
# Select a parameter setting
selectedIndices <- svm_model_all_pca$pred$mtry == 2
# Plot:
plot.roc(svm_model_all_pca$pred$obs[selectedIndices],
         svm_model_all_pca$pred$M[selectedIndices])


selectedIndices <- svm_model_all_pca$pred$mtry == 2

library(ggplot2)
library(plotROC)
ggplot(svm_model_all_pca$pred[selectedIndices, ], 
       aes(m = M, d = factor(obs, levels = c("R", "M")))) + 
    geom_roc(hjust = -0.4, vjust = 1.5) + coord_equal()




roc_svm_test <- roc(response = svm_model_all_pca$trainingData, predictor =as.numeric(preds_svm_model_all_pca))
plot(roc_svm_test, add = TRUE,col = "red", print.auc=TRUE, print.auc.x = 0.5, print.auc.y = 0.3)
legend





gc_ctrl1 <- trainControl(method = "repeatedcv",
                         number = 5,
                         repeats = 5,
                         classProbs = TRUE,
                         summaryFunction = twoClassSummary,
                         savePredictions = TRUE)


levels(train_df$positive_return)=c("Yes","No")

gc_train1 <- train(positive_return~.,
                   data = train_df,
                   method = c("svmRadial", "svmPoly"),
                   # train() use its default method of calculating an analytically derived estimate for sigma
                   tuneLength = 5,# 5 arbitrary values for C and sigma = 25 models
                   trControl = gc_ctrl1,
                   preProc = c("center", "scale"),
                   metric = "ROC",
                   verbose = FALSE)

max(gc_train1$results[,"ROC"])




# ROC using pROC
gc_prob <- predict(gc_train1, newdata = test_df %>% select(-positive_return), type = "prob")
gc_pROC <- roc(response = test_df$positive_return, predictor = gc_prob[, "Yes"])
plot(gc_pROC)
gc_pROC$auc
# Area under the curve: 0.8376


ROC_df <- tibble(x = gc_pROC$specificities  , y = gc_pROC$sensitivities) %>% 
    mutate(fpr = 1- x)


ROC_df <- tibble(fpr =  1- gc_pROC$specificities  , Specificity = gc_pROC$sensitivities)

ROC_df %>% ggplot() + 
  geom_line(aes(x = fpr, y = Specificity, color = "Specificity/fpr")) + 
  geom_abline(slope =  1) +
  theme_classic()



produce_roc <- function(model, title,  train_df, test_df) {
  #'
  #'
  gc_prob <- predict(model, newdata = test_df, type = "prob")
  gc_pROC <- roc(response = test_df$positive_return, predictor = gc_prob[, "Yes"])
  ROC_df <- tibble(fpr =  1- gc_pROC$specificities  , tpr = gc_pROC$sensitivities)
  ROC_df %>% ggplot() + 
    geom_line(aes(x = fpr, y = tpr, colour = "specificity/fpr")) + 
    geom_abline(slope =  1, colour = "random prediction") +
    theme_classic() +
    labs(title = title, x = "False Positive Rate", y = "True Positive Rate")
  
  
}
produce_roc(gc_train1, "ROC", train_df, test_df)





# ROC using plotROC (ggplot2 extension)
gc_prob_ex <- extractProb(list(gc_train1), test_df %>% select(-positive_return))
gc_ggROC <- ggplot::plotROC(gc_prob_ex, aes(d=obs, m=Good)) + geom_roc() 
gc_ggROC_styled <- gc_ggROC +  annotate("text", x = .75, y = .25, 
                                        label = paste("AUC =", round(calc_auc(gc_ggROC)$AUC, 2)))
gc_ggROC_styled
