################################################################################
########################### Pre-processing #####################################
################################################################################

# Libraries --------------------------------------------------------------------
library(tidyverse)
library(magrittr)




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
    head(100))$permno

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
    
    if(print_removed_cols) cat("Columns removed: ", cols, "\n")
    
    return (out_df %>% select(-cols))
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
    
    if(print_removed_cols) cat("Columns removed: ", cols, "\n")
    
    return(df %>% select(-cols))
}





test_df <- df %>% remove_cols_only_zero_and_NA(T) %>% remove_NA(0.2, T)






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




