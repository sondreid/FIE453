################################################################################
########################### Pre-processing #####################################
################################################################################
# Candidates: 
## 8, 15, 54



# Libraries --------------------------------------------------------------------
library(tidyverse)
library(magrittr)
library(tidymodels)
library(randomForest)
library(caret)
library(doParallel)
library(MLmetrics)
library(gbm)
library(PerformanceAnalytics)
library(kableExtra)
library(knitr)
library(monomvn)
library(kableExtra)
library(lubridate)
library(kknn)
library(nnet)


set.seed(1)


# Load and read data -----------------------------------------------------------
load("data/merged.Rdata")
company_names_df <- read.csv(file = "descriptions/names.csv") 
feature_names_df <- read.delim(file = "descriptions/compustat-fields.txt")    
company_names_df %<>% rename_with(tolower) %>% mutate(date = lubridate::ymd(date))
feature_names_df %<>% rename_with(tolower) 
merged %<>% rename_with(tolower) %>% mutate(date = lubridate::ymd(date))




######################## DATA PROCESSING FUNCTIONS  ##########################





# Data reduction ---------------------------------------------------------------
get_subset_of_companies <-function(df, number_of_companies) {

    #' @description:              To reduce run time, we want to reduce the 
    #'                            number of companies,(for variable selection 
    #'                            purposes)
    #' 
    #' @param df                  The dataframe to be split
    #' @param number_of_companies The number of speakers to be retained
    #' @return                    A dataframe of fewer companies
    
    set.seed(123)
    companies <- df$permno %>% unique()
    subset_of_companies <- companies %>% 
        sample(x = ., size = number_of_companies)
    return(df %>% filter(permno %in% subset_of_companies))
    
}




get_subset_of_companies_ratio <-function(df, ratio) {

    #' @Description:              To reduce run time, we want to reduce the 
    #'                            number of companies, (for variable selection 
    #'                            purposes)
    #' 
    #' @param df                  The dataframe to be split
    #' @param number_of_companies The number of speakers to be retained
    #' @return:                   A dataframe of fewer companies
    
    set.seed(123)
    companies <- df$permno %>% unique()
    number_of_companies <- companies %>% length()
    subset_of_companies <- companies %>% 
        sample(x = ., size = as.integer(number_of_companies*ratio))
    return(df %>% filter(permno %in% subset_of_companies))
    
}



# Feature selection functions --------------------------------------------------
remove_cols_only_zero_and_NA <- function(df, print_removed_cols = F) {
    
    #' @description              Function that removes columns containing only 
    #'                           zeros and NAs
    #'
    #' @param df                 Passing a data frame
    #' @param print_removed_cols True if user want to print removed columns
    #' @return                   Data frame without columns that containing only 
    #'                           zeros and NAs

    cols <- df %>% 
        apply(MARGIN = 2, 
              function(x) (sum(x==0, na.rm = T) + sum(is.na(x)))/length(x))
    
    cols <- cols[cols == 1] %>% 
        as.data.frame() %>% 
        rownames() 
    
    if(print_removed_cols) cat("Columns removed: ", cols, "\n\n")
    
    return (df %>% dplyr::select(-cols))
}




remove_NA <- function(df, ratio, print_removed_cols = F){
    
    #' @description              Function that removes columns containing NAs 
    #'                           beyond a given ratio
    #'             
    #' @param df                 Passing a data frame
    #' @param ratio              Passing a upper limit NA ratio
    #' @param print_removed_cols True if user want to print removed columns
    #' @return                   Data frame without columns containing NAs 
    #'                           beyond given ratio
    
    cols <- df %>% 
        apply(MARGIN = 2, 
              function(x) sum(is.na(x))/length(x))
    
    cols <- cols[cols >= ratio] %>% 
        as.data.frame() %>% 
        rownames() 
    
    if(print_removed_cols) cat("Columns removed: ", cols, "\n\n")
    
    return(df %>% dplyr::select(-cols))
}




remove_nzv <- function(df, print_removed_cols = F){
    
    #' @description              Function that removes near zero variance 
    #'                           columns
    #'             
    #' @param df                 Passing a data frame
    #' @param print_removed_cols True if user want to print removed columns
    #' @return                   Data frame without columns near zero variance 
    #'                           columns
    
    rec <- recipe(retx ~ ., 
                  data = df)
    
    cols <- (rec %>% 
                step_nzv(all_predictors()) %>% 
                prep(df) %>% 
                tidy(number = 1))$terms
    
    if(print_removed_cols) cat("Columns removed: ", cols, "\n\n")
    
    return(df %>% dplyr::select(-cols))
    
}



remove_hcv <- function(df, threshold = 0.8, print_removed_cols = F){
    
    #' @description              Function that removes highly correlated 
    #'                           features
    #'             
    #' @param df                 Passing a data frame
    #' @param treshold           Correlation beneath this threshold
    #' @param print_removed_cols True if user want to print removed columns
    #' @return                   Data frame without highly correlated features
    
    numeric_cols <- df %>% 
        lapply(is.numeric) %>% 
        unlist()
    
    rec <- recipe(retx ~ ., 
                  data = df[numeric_cols])
    
    cols <- (rec %>% 
                 step_corr(all_predictors(),
                           threshold = threshold) %>% 
                 prep(df[numeric_cols]) %>% 
                 tidy(number = 1))$terms
    
    if(print_removed_cols) cat("Columns removed: ", cols, "\n\n")
    
    return(df %>% dplyr::select(-cols))
    
}




replace_NA_with_mean <- function(df, print_replaced_cols = F){
    
    #' @description               Function that replaces NA with column means
    #'             
    #' @param df                  Passing a data frame
    #' @param print_replaced_cols True if user want to print replaced columns
    #' @return                    Data frame NA-replaced column means
    
    na_cols <- df %>% 
        apply(MARGIN = 2, 
              function(x) any(is.na(x)))
    
    numeric_cols <- df[na_cols] %>% 
        lapply(is.numeric) %>% 
        unlist() 
    
    col_means <- df[na_cols] %>% 
        colMeans(na.rm = T)
    
    col_names <- col_means %>% 
        names()
    
    for (col in col_names){
        df[col] <- df[col][[1]] %>% 
            replace_na(col_means[col])
    }
    
    if(print_replaced_cols) cat("Columns replaced: ", col_names, "\n\n")
    
    return(df)
}





remove_NA_rows <- function(df) {
    #' @description Function that removes any rows with one or more NA's
    #'
    #' @param df    Passing a data frame
    #' @return      Data frame NA filtered rows
    
    return(df %>% filter(across(everything(), ~ !is.na(.x))) )
    
}





perform_train_test_split <- function(df, train_ratio = 0.8) {
    
    #' @description Ensures an equal amount of companies in each set
    #' 
    #' @param df    The dataframe to be split
    #' @param ratio Ratio of training data, (validation and test set to equal 
    #'              length)
    #' @return      A list of three data frames: training, validation and 
    #'              test sets
    
    set.seed(123)
    
    all_companies <- df$permno %>% unique()
    
    train_indices <- sample(1:length(all_companies), 
                            floor(length(all_companies) * train_ratio))
    train_companies <- all_companies[train_indices]
    test_companies <- all_companies[-train_indices]
    
    train_sample <- df %>% filter(permno %in% train_companies)
    test_sample  <- df %>% filter(permno %in% test_companies)
    
    return (list(train_sample, test_sample))
    
}



find_company_observations <- function(df, minimum_observations) {
    
    #' @description                 Finds companies that have less than a 
    #'                              minimum amount of observations
    #' 
    #' @param df                    Passing a data frame
    #' @param minimum_observations  Passing minimum observation limit
    #' @return                      A data frame
    
    all_companies <- df$permno %>% 
        unique()
    
    df %<>% group_by(permno) %>% 
        summarise(count = n()) %>% 
        ungroup() %>% 
        filter(count < minimum_observations) %>% 
        arrange(desc(count))
    
    return(df)
}





# Applying preprocessing steps
## Apply variance and correlation filter


merged %<>% remove_cols_only_zero_and_NA(print_removed_cols = T) %>% 
    remove_NA(0.2, print_removed_cols = T) %>% 
    remove_nzv(print_removed_cols = T) %>% 
    remove_hcv(0.9, print_removed_cols = T) %>% 
    remove_NA_rows() %>%  # Remove rows with NA's       
    transform(vol = as.numeric(vol),
              shrout = as.numeric(shrout)) 


# Irrelevant features ----------------------------------------------------------
# Variables that cannot be included with dependent variable RETX
excluded_variables <- c("ret", 
                        "prc",         # Price should maybe be allowed
                        "datafqtr",# Remove all date related variables
                        "fyearq",
                        "datacqtr",
                        "gvkey", # Company identifier
                        "fyr",
                        "fqtr",
                        "datacqtr") 

merged %<>%  dplyr::select(-excluded_variables)





######################### Time splitting


selection_data <- merged %>% 
    filter(year(date) < "2018")


evaluation_data <- merged %>% 
    filter(year(date) >= "2018")

selection_data %<>% dplyr::select(-date)




################### TRAIN AND TEST SPLITS ### 




# Train-Test-Split
train_test <- perform_train_test_split(selection_data, 
                                       train_ratio = 0.8)                       # Split into train and test set with seperate sets of companies
train_df <- train_test[[1]]
test_df <- train_test[[2]]

test_df %>% 
    inner_join(train_df, by = "permno") %>% 
    nrow()

train_df %<>% dplyr::select(-permno) # Remove company numbers from training

low_observation_count_companies <- find_company_observations(test_df, 50)
test_df %<>% anti_join(low_observation_count_companies)                        # Cut companies with fewer than 50 observations (they cannot be reliably predicted)




### REDUCED DATA SET FOR TESTING

selection_data_reduced <- get_subset_of_companies_ratio(selection_data, 0.1) 

# Train-Test-Split
train_test_reduced <- perform_train_test_split(selection_data_reduced, 
                                       train_ratio = 0.8)                       # Split into train and test set with seperate sets of companies
train_df_reduced <- train_test_reduced[[1]]
test_df_reduced <- train_test_reduced[[2]]

train_df_reduced %<>% dplyr::select(-permno) # Remove company numbers from training

low_observation_count_companies <- find_company_observations(test_df_reduced, 50)
test_df_reduced %<>% anti_join(low_observation_count_companies)                        # Cut companies with fewer than 50 observations (they cannot be reliably predicted)



### SAVE datasets

save(train_df, selection_data, test_df, train_df_reduced, test_df_reduced, 
     company_names_df, evaluation_data, feature_names_df, file = "cached_data/train_test.Rdata") 



rm(merged, df_selection, df_selection_reduced) # Remove large datasets from memory









