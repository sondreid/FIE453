################################################################################
########################### Pre-processing #####################################
################################################################################

# Libraries --------------------------------------------------------------------
library(tidyverse)
library(magrittr)
library(tidymodels)
library(randomForest)
library(caret)
library(doParallel)



# Set WD -----------------------------------------------------------------------
#setwd("~/OneDrive - Norges Handelshøyskole/MASTER/FIE453/FinalExam/FIE453/Final Paper")




# Load and read data -----------------------------------------------------------
load("data/merged.Rdata")
#company_names_df <- read.csv(file = "data/names.csv")
feature_names_df <- read.delim(file = "descriptions/compustat-fields.txt")    
#company_names_df %<>% rename_with(tolower)
#feature_names_df %<>% rename_with(tolower) 
merged %<>% rename_with(tolower)




# Subset of 100 companies ------------------------------------------------------
permno_top <- (merged %>% 
    select(permno) %>% 
    unique() %>% 
    head(1000))$permno

df_reduced <- merged %>% filter(permno %in% permno_top)




# Feature selection functions --------------------------------------------------
remove_cols_only_zero_and_NA <- function(df, print_removed_cols = F) {
    
    #'@description Function that removes columns containing only zeros and NAs
    #'
    #'@param df    Passing a data frame
    #'@param print_removed_cols True if user want to print removed columns
    #'@return      Data frame without columns that containing only zeros and NAs

    cols <- df %>% apply(MARGIN = 2, function(x) (sum(x==0, na.rm = T) + sum(is.na(x)))/length(x))
    cols <- cols[cols == 1] %>% as.data.frame() %>% rownames() 
    
    if(print_removed_cols) cat("Columns removed: ", cols, "\n\n")
    
    return (df %>% select(-cols))
}




remove_NA <- function(df, ratio, print_removed_cols = F){
    
    #'@description Function that removes columns containing NAs beyond a given
    #'             ratio
    #'             
    #'@param df    Passing a data frame
    #'@param ratio Passing a upper limit NA ratio
    #'@param print_removed_cols True if user want to print removed columns
    #'@return      Data frame without columns containing NAs beyond given ratio
    
    cols <- df %>% apply(MARGIN = 2, function(x) sum(is.na(x))/length(x))
    cols <- cols[cols >= ratio] %>% as.data.frame() %>% rownames() 
    
    if(print_removed_cols) cat("Columns removed: ", cols, "\n\n")
    
    return(df %>% select(-cols))
}




remove_nzv <- function(df, print_removed_cols = F){
    
    #'@description Function that removes near zero variance columns
    #'             
    #'@param df    Passing a data frame
    #'@param print_removed_cols True if user want to print removed columns
    #'@return      Data frame without columns near zero variance columns
    
    rec <- recipe(retx ~ ., 
                  data = df)
    
    cols <- (rec %>% 
                step_nzv(all_predictors()) %>% 
                prep(df) %>% 
                tidy(number = 1))$terms
    
    if(print_removed_cols) cat("Columns removed: ", cols, "\n\n")
    
    return(df %>% select(-cols))
}



remove_hcv <- function(df, threshold = 0.9, print_removed_cols = F){
    
    #'@description Function that removes highly correlated features
    #'             
    #'@param df    Passing a data frame
    #'@param treshold Correlation beneath this treshold
    #'@param print_removed_cols True if user want to print removed columns
    #'@return      Data frame without highly correlated features
    
    
    
    numeric_cols <- df %>% lapply(is.numeric) %>% unlist()
    
    rec <- recipe(retx ~ ., 
                  data = df[numeric_cols])
    
    cols <- (rec %>% 
                 step_corr(all_predictors(),
                           threshold = threshold) %>% 
                 prep(df[numeric_cols]) %>% 
                 tidy(number = 1))$terms
    
    if(print_removed_cols) cat("Columns removed: ", cols, "\n\n")
    
    return(df %>% select(-cols))
}




replace_NA_with_mean <- function(df, print_replaced_cols = F){
    
    #'@description Function that replaces NA with column means
    #'             
    #'@param df    Passing a data frame
    #'@param print_replaced_cols True if user want to print replaced columns
    #'@return      Data frame NA-replaced column means
    
    na_cols <- df %>% apply(MARGIN = 2, function(x) any(is.na(x)))
    numeric_cols <- df[na_cols] %>% lapply(is.numeric) %>% unlist() 
    col_means <- df[na_cols] %>% colMeans(na.rm = T)
    col_names <- col_means %>% names()
    
    for (col in col_names){
        df[col] <- df[col][[1]] %>% replace_na(col_means[col])
    }
    
    if(print_replaced_cols) cat("Columns replaced: ", col_names, "\n\n")
    
    return(df)
}




#### Functions for model analysis ##############################################



plot_confusion_matrix <- function(obs, preds) {
    #' @description: Function that plots confusion matrix 
    #' @obs: Observations in vector format
    #' @preds: predictions in vector format
    conf_data <- tibble(obs = obs, preds =preds )
    confusion_matrix <- conf_mat(conf_data, truth = "obs", estimate = "preds")
    autoplot(confusion_matrix, type = "heatmap") +
        theme(text = element_text(size = 25))    
}




make_table <- function(obs, preds, model_name) {
    #' @description: Function that produces a table of performance metrics: Accuracy, preciscion and recall
    #' @obs: Observations in vector format
    #' @preds: predictions in vector format
    #' @return: A dataframe 
    obs <- as.factor(obs)
    preds <- as.factor(preds)
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

 #####################

# Testing ----------------------------------------------------------------------
df_reduced %<>% 
    remove_cols_only_zero_and_NA(print_removed_cols = T) %>% 
    remove_NA(0.2, print_removed_cols = T) %>% 
    remove_nzv(print_removed_cols = T) %>% 
    remove_hcv(0.9, print_removed_cols = T) %>% 
    replace_NA_with_mean(print_replaced_cols = T)





perform_train_test_split <- function(df, train_ratio = 0.8) {
    #' @Description: Ensures an equal amount of companies in each set
    #' 
    #' @df:    The dataframe to be split
    #' @ratio: Ratio of training data, (validation and test set to equal length)
    #' @return: A list of three dataframes: training, validation and test sets
    set.seed(123)
    all_companies <- df$permno %>% unique()
    
    train_indices <- sample(1:length(all_companies), floor(length(all_companies)* train_ratio))
    train_companies <- all_companies[train_indices]
    test_companies <- all_companies[-train_indices]
    
    train_sample <- df %>% filter(permno %in% train_companies)
    test_sample  <- df %>% filter(permno %in% test_companies)
    return ( 
        list(train_sample, test_sample)
    )
}



find_company_observations <- function(df, minimum_obserations) {
    #' 
    #' @description: Finds companies that have less than a minimum amount of observations
    all_companies <- df$permno %>% unique()
    
    df %<>% group_by(permno) %>% 
        summarise(count = n()) %>% 
        ungroup() %>% 
        filter(count < minimum_obserations) %>% 
        arrange(desc(count))
    
    return(df)
}

low_observation_count_companies <- find_company_observations(df_reduced, 50)

df_reduced <- df_reduced %>% anti_join(low_observation_count_companies) # Cut companies with fewer than 50 observations (they cannot be reliably predicted)



# Train and test split

train_test_reduced <- perform_train_test_split(df_reduced, train_ratio = 0.8)
train_df_reduced <- train_test_reduced[[1]]
test_df_reduced <- train_test_reduced[[2]]

# Check for similar rows
test_df_reduced %>% inner_join(train_df_reduced, by = "permno") %>% nrow()




cl <- makePSOCKcluster(5)
registerDoParallel(cl)



#y <- train_df_reduced$retx
#x <- train_df_reduced %>% select(-retx)

train_control <- trainControl(method = "cv",
                        number = 5,
                        verboseIter = TRUE,
                        classProbs = TRUE, 
                        summaryFunction = twoClassSummary)

# Random Forest ----------------------------------------------------------------
set.seed(1)

mtry <- round(sqrt(ncol(x)))

tunegrid_rf <- expand.grid(.mtry = 2)


start_time <- Sys.time()
rf <- train(x, y,
            method = "rf",
            importance = TRUE,
            tuneGrid = tunegrid_rf,
            trControl = train_control)
end_time <- Sys.time()

# Most important features
varImp(rf)

# KNN ----------------------------------------------------------------
# Should be far faster than RF, SVM, etc


tunegrid_knn <- expand.grid(k = 5:10)


knn <- train(retx~,
             data = train_df_reduced,
             method = "knn",
             tunegrid = tunegrid_knn,
             trControl = train_control,
             preProcess = c("center","scale"),
             allowParalell=TRUE)


knn
summary(knn)



# SVM ----------------------------------------------------------------


tunegrid_svm

svm_model_all_pca <- caret::train(retx~,
                                  data = train_df_reduced,
                                  method = "svmRadial",
                                  data = train_df,
                                  trControl  = train_control, 
                                  preProcess = c("center", "scale", "pca"),
                                  allowParallel=TRUE)





# Stop cluster
stopCluster(cl)









