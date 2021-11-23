############################ ENE452 final paper ##############################
##############################################################################

########################### DATA EXPLORATION SCRIPT ###########################

## Libaries
library(tidyverse)
library(magrittr)
library(caret)
library(tsibble)
library(Keras)
library(readr)
library(kableExtra)
library(leaps)


load("data/merged.Rdata")
read_compustat_crsp <- function() {
  crsp_df <- read_csv("data/crsp.csv")
  compustat_df <- read_csv("data/compustat.csv")
  
  compustat_columns <- colnames(compustat_df) %>% as_tibble()
  crsp_columns <- colnames(crsp_df) %>% as_tibble()
  
  save(compustat_columns, crsp_columns, file = "cached_data/column_names.Rdata")
}
#read_compustat_crsp()
load(file = "cached_data/column_names.Rdata")

# Subset of 100 companies
permno <- merged %>% 
  select(PERMNO) %>% 
  unique() %>% 
  head(1000)


remove_zero_and_NA <- function(df, ratio) {
  #'
  #'@ratio: ratio of NA's which a column of data cannot exceed
  
  out_df <- df 
  df[is.na(df)] <- 0
  cols <- df %>% apply(MARGIN = 2, function(x) sum(x==0, na.rm = T)/length(x)) 
  cols <- cols[cols < ratio] %>% as.data.frame() %>% rownames() 
  
  return (
    out_df %>% select(cols)
  )
}

###### EITHER FULLY MERGED SET OR A REDUCED DATASET

# Filtering the data frame containing only the 100 first companies
df_reduced <- merged %>% 
  filter(PERMNO %in% permno$PERMNO) %>% 
  remove_zero_and_NA(0.3) %>% 
  filter(!is.na(RETX)) 





# This has to be done piecewise to preserve memory if we intend to try to use all rows
#merged %<>% 
 # remove_all_zero_columns() %>%
  #remove_all_duplicates()



### Subset selection



#subset_of_variabels <-  regsubsets(RETX~., df_reduced )
#summary(subset_of_variabels)




######################################## TRAIN, VALIDATION AND TEST SPLITS ################################## 
perform_train_validate_split <- function(df, train_ratio = 0.6, test_ratio = 0.5) {
  #' @Description: Ensures an equal amount of companies in each set
  #' 
  #' @df:    The dataframe to be split
  #' @ratio: Ratio of training data, (validation and test set to equal length)
  #' @return: A list of three dataframes: training, validation and test sets
  set.seed(13231)
  all_companies <- df$PERMNO %>% unique()
  
  train_indices <- sample(1:length(all_companies), floor(length(all_companies)* train_ratio))
  train_companies <- all_companies[train_indices]
  val_test_companies <- all_companies[-train_indices]
  
  val_indices <- sample(1:length(val_test_companies), floor(length(val_test_companies)* test_ratio))
  val_companies <- all_companies[val_indices]
  

  test_companies <- all_companies[-c(val_indices, train_indices)]
  train_sample <- df %>% filter(PERMNO %in% train_companies)
  val_sample <- df %>% filter(PERMNO %in% val_companies)
  test_sample <- df %>% filter(PERMNO %in% test_companies)
  return ( 
    list(train_sample, val_sample, test_sample)
  )
}


perform_train_test_split <- function(df, train_ratio = 0.8) {
  #' @Description: Ensures an equal amount of companies in each set
  #' 
  #' @df:    The dataframe to be split
  #' @ratio: Ratio of training data, (validation and test set to equal length)
  #' @return: A list of three dataframes: training, validation and test sets
  set.seed(123)
  all_companies <- df$PERMNO %>% unique()
  
  train_indices <- sample(1:length(all_companies), floor(length(all_companies)* train_ratio))
  train_companies <- all_companies[train_indices]
  test_companies <- all_companies[-train_indices]

  train_sample <- df %>% filter(PERMNO %in% train_companies)
  test_sample  <- df %>% filter(PERMNO %in% test_companies)
  return ( 
    list(train_sample, test_sample)
  )
}



find_company_observations <- function(df, minimum_obserations) {
  #' 
  #' @description: Finds companies that have less than a minimum amount of observations
  all_companies <- df$PERMNO %>% unique()
  
  df %<>% group_by(PERMNO) %>% 
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
test_df_reduced %>% inner_join(train_df_reduced, by = "PERMNO") %>% nrow()
