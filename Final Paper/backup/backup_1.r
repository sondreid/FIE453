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
  head(100)


remove_all_zero_columns <- function(df) {
  df[is.na(df)] <- 0
  columns <- df %>% apply(MARGIN = 2, FUN = function(x) all(x == 0)) %>% 
    as.data.frame() %>% 
    filter(. == FALSE) %>% 
    rownames()
  return (df %>% select(all_of(columns)))
}

remove_all_duplicates <- function(df) {
  columns <- df %>% 
    apply(MARGIN = 2, FUN = function(x) if(length(c(unique(x))) == 1) return(FALSE) else return(TRUE)) %>% 
    as.data.frame() %>% 
    filter(. == TRUE) %>% 
    rownames()
  
  return (df %>% select(all_of(columns)))
}


remove_zero_and_NA <- function(df, ratio) {
  #'
  #'@ratio: ``
  cols <- df %>% apply(MARGIN = 2, function(x) sum(is.na(x) | x == 0, na.rm = T)/length(df)) 
  cols <- cols[cols < ratio] %>% as.data.frame() %>% rownames() 
  
  return (
    df %>% select(cols)
  )
}




# Filtering the data frame containing only the 10 first companies
df_reduced <- merged %>% 
  filter(PERMNO %in% permno$PERMNO)


###### EITHER FULLY MERGED SET OR A REDUCED DATASET

df_reduced %>% 
  remove_all_zero_columns() %>%
  remove_all_duplicates() %>% 
  remove_zero_and_NA(0.7)




# This has to be done piecewise to preserve memory if we intend to try to use all rows
merged %<>% 
  remove_all_zero_columns() %>%
  remove_all_duplicates()





remove_zero_and_NA(df_reduced, 0.5)
cols <- df_reduced %>% apply(MARGIN = 2, function(x) sum((is.na(x) | x == 0), na.rm = F)/length(x))
cols <- cols[cols < 0.5] %>% as.data.frame() %>% rownames() 
cols  %>% length()

test <- df_reduced %>% select(cols)
test <- merged %>% group_by() %>% summarise(across(fns = function(x) x %>% is.na() %>% sum(na.rm = T)))


# Reduce columns
columns <- c("PRC",
             "PERMNO",
             "RET")

merged_subset <- merged %>% select(columns)

merged_subset %<>% 
  remove_all_zero_columns() %>%
  remove_all_duplicates() %>% 
  mutate(return = difference(PRC, 1)) # Add returns 



# Add returns
#merged %>% mutate(return = difference(PRC, 1))






