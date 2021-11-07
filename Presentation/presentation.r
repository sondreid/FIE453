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
    require(ggplot2)
    require(plotROC)
    require(pROC)

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

# Remove non-numeric variables
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


# Convert class labels

levels(df$positive_return)=c("Yes","No")

    
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
    #'Perform SVM on a predetermined depdendent variable, and datasets.
    set.seed(123)
    train_control <- trainControl(method="repeatedcv", number=10, repeats=3, 
                                  savePredictions = T,
                                  summaryFunction = twoClassSummary,
                                  classProbs = T)
    
    svm_model_all_pca <- caret::train(positive_return~.,
                            method = "svmPoly",
                            data = train_df,
                            trControl  = train_control, 
                            #preProcess = c("pca"),
                            preProcess = c("center", "scale", "pca"),
                            tunelength = 4,
                            metric = "ROC",
                            allowParallel=TRUE)



    svm_model_all_scaled <- caret::train(positive_return~.,
                            method = "svmPoly",
                            data = train_df,
                            trControl  = train_control, 
                            preProcess = c("center", "scale"),
                            tunelength = 4,
                            metric = "ROC",
                            allowParallel=TRUE)
    
    
    svm_radial_model_all_pca <- caret::train(positive_return~.,
                                      method = "svmRadial",
                                      data = train_df,
                                      trControl  = train_control, 
                                      #preProcess = c("pca"),
                                      preProcess = c("center", "scale", "pca"),
                                      tunelength = 4,
                                      metric = "ROC",
                                      allowParallel=TRUE)
    
    
    
    svm_radial_model_all_scaled <- caret::train(positive_return~.,
                                         method = "svmRadial",
                                         data = train_df,
                                         trControl  = train_control, 
                                         preProcess = c("center", "scale"),
                                         tunelength = 4,
                                         metric = "ROC",
                                         allowParallel=TRUE)


    ###### Modelling only few features #####
    subset_train_df <- train_df %>%
        select(c(positive_return, PRC, VOL, vwretd))


    svm_model_subset_scaled <- caret::train(positive_return~.,
                            method = "svmPoly",
                            data = subset_train_df,
                            trControl  = train_control, 
                            preProcess = c("center", "scale"),
                            tunelength = 4, 
                            metric = "ROC",
                            allowParallel=TRUE)
    
    svm_radial_model_subset_scaled <- caret::train(positive_return~.,
                                            method = "svmRadial",
                                            data = subset_train_df,
                                            trControl  = train_control, 
                                            preProcess = c("center", "scale"),
                                            tunelength = 4, 
                                            metric = "ROC",
                                            allowParallel=TRUE)

    save(svm_model_all_pca, svm_model_all_scaled, svm_radial_model_all_pca,
         svm_radial_model_all_scaled,  svm_model_subset_scaled, svm_radial_model_subset_scaled, file = "svm_model.Rdata")
}
#perform_svm()

load(file = "svm_model.Rdata")




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
plot_confusion_matrix(test_df$positive_return, predict(svm_model_all_pca, test_df))
make_table(test_df$positive_return, predict(svm_model_all_pca, test_df), "All variables PCA reduced using polynomial kernel")

## ALL PCA Radial kernel
plot_confusion_matrix(test_df$positive_return, predict(svm_radial_model_all_pca,test_df))
make_table(test_df$positive_return, predict(svm_radial_model_all_pca, test_df), "All variables PCA reduced using radial kernel")  %>% 
  save_kable(file = "images/PCA_radial_all_variables.png", density = 2000, zoom = 3)



## All scaled
plot_confusion_matrix(test_df$positive_return, predict(svm_model_all_scaled, test_df))
make_table(test_df$positive_return, predict(svm_model_all_scaled, test_df), "All variables scaled using polynomial kernel")  %>% 
  save_kable(file = "images/scaled_all_variables.png", density = 2000, zoom = 3)



## All scaled radial
plot_confusion_matrix(test_df$positive_return, predict(svm_radial_model_all_scaled, test_df))
make_table(test_df$positive_return, predict(svm_radial_model_all_scaled, test_df), "All variables scaled")


## Subset scaled
plot_confusion_matrix(test_df$positive_return, predict(svm_model_subset_scaled, test_df))
make_table(test_df$positive_return, predict(svm_model_subset_scaled, test_df), "Subset of variables scaled using polynomial")


## Subset scaled using radial kernel
plot_confusion_matrix(test_df$positive_return, predict(svm_radial_model_subset_scaled, test_df))
make_table(test_df$positive_return, predict(svm_radial_model_subset_scaled, test_df), "Subset of variables scaled using radial kernel") %>% 
  save_kable(file = "images/scaled_subset_radial.png", density = 2000, zoom = 3)






produce_roc <- function(model, title, test_df) {
  #' Function which produces a ROC plot based on the input model, title and
  #' datasets.
  gc_prob <- predict(model, newdata = test_df, type = "prob")
  gc_pROC <- roc(response = test_df$positive_return, predictor = gc_prob[, "Yes"])
  ROC_df <- tibble(fpr =  1- gc_pROC$specificities  , tpr = gc_pROC$sensitivities)
  ROC_df %>% ggplot() + 
    geom_line(aes(x = fpr, y = tpr, colour = "specificity/fpr")) + 
    geom_abline(slope =  1) +
    theme_classic() +
    labs(title = title, x = "False Positive Rate", y = "True Positive Rate")
  
  
}


produce_roc(svm_model_all_pca, "ROC", test_df)
produce_roc(svm_model_all_scaled, "ROC scaled and centered. All variables", test_df)


# Radial PCA all variables
produce_roc(svm_radial_model_all_pca, "SVM model using radial kernel PCA reduced", test_df)

# Poly PCA all variables
produce_roc(svm_model_all_pca, "ROC poly kernel PCA reduced", test_df)


# Radial
produce_roc(svm_radial_model_subset_scaled, "ROC Radial on poly", test_df)
# Poly
produce_roc(svm_model_subset_scaled, "ROC", test_df)
