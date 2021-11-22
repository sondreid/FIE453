################################################################################
########################### Pre-processing #####################################
################################################################################

# Libraries --------------------------------------------------------------------
library(tidyverse)
library(magrittr)
library(tidymodels)



# Set WD -----------------------------------------------------------------------
setwd("~/OneDrive - Norges Handelshøyskole/MASTER/FIE453/FinalExam/FIE453/Final Paper")




# Load and read data -----------------------------------------------------------
load("data/merged.Rdata")
company_names_df <- read.csv(file = "data/names.csv")
feature_names_df <- read.delim(file = "data/compustat-fields.txt")    
company_names_df %<>% rename_with(tolower)
feature_names_df %<>% rename_with(tolower) 
merged %<>% rename_with(tolower)




# Subset of 100 companies ------------------------------------------------------
permno_top <- (merged %>% 
    select(permno) %>% 
    unique() %>% 
    head(1000))$permno

df <- merged %>% filter(permno %in% permno_top)




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
    #'@param print_removed_cols True if user want to print removed columns
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


# Testing ----------------------------------------------------------------------
df %<>% 
    remove_cols_only_zero_and_NA(print_removed_cols = T) %>% 
    remove_NA(0.2, print_removed_cols = T) %>% 
    remove_nzv(print_removed_cols = T) %>% 
    remove_hcv(0.9, print_removed_cols = T) %>% 
    replace_NA_with_mean(print_replaced_cols = T)


# Random Forest ----------------------------------------------------------------
library(randomForest)
library(caret)
y <- df$retx
x <- df %>% select(-retx)

control <- trainControl(method = "repeatedcv",
                        number = 5,
                        repeats = 3,
                        verboseIter = TRUE)

set.seed(1)

mtry <- sqrt(ncol(x))

tunegrid <- expand.grid(.mtry = mtry)


start_time <- Sys.time()
rf <- train(x, y,
            method = "rf",
            importance = TRUE,
            tuneGrid = tunegrid,
            trControl = control)
end_time <- Sys.time()

