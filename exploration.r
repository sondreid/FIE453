######################################################################################################################################
################################################ FIE453        #######################################################################
######################################################################################################################################


#### Libraries

library(tidyverse)
library(magrittr)
library(dplyr)
library(kableExtra)



### Load data


crispr <- read_csv("Data/crispr.csv") %>% as_tibble() %>% 
  rename("company_identifier" = "PERMNO")

crispr[1:100,] %>% tibble::view()

compustat <- read_csv("Data/compustat.csv") %>% as_tibble() %>% 
  rename("company_identifier" = "GV_KEY")
compustat[1:100,] %>% tibble::view()

comp_colnames <- colnames(compustat)

"SUFF" %in% comp_colnames


joined_data_set <- compustat %>% left_join(crspr)