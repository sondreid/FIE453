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
test_df <- df %>% remove_zero_and_NA(0.5)

###### EITHER FULLY MERGED SET OR A REDUCED DATASET

# Filtering the data frame containing only the 100 first companies
df_reduced <- merged %>% 
  filter(PERMNO %in% permno$PERMNO) %>% 
  remove_zero_and_NA(0.3) %>% 
  filter(!is.na(RETX)) %>% 
  select_if(negate(is.character)) # Remove factors and character variables





# This has to be done piecewise to preserve memory if we intend to try to use all rows
merged %<>% 
  remove_all_zero_columns() %>%
  remove_all_duplicates()



### Subset selection



subset_of_variabels <-  regsubsets(RETX~., df_reduced )
summary(subset_of_variabels)







