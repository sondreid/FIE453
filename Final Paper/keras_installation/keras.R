### KERAS ####


## First installation
library(reticulate)
conda_version()
conda_list()
conda_python(envname = "r-reticulate")    
conda_install(envname = "r-reticulate",packages="r-reticulate")
conda_install(envname = "r-reticulate",packages="r-tensorflow")
conda_install(envname = "r-reticulate",packages="r-keras")


# Restart session
## AFTER RESTART

library(tensorflow)
library(keras)
library(reticulate)
conda_python(envname = "r-reticulate")
tensorflow::use_condaenv("r-reticulate")
install_tensorflow(version = "gpu", method = "conda", envname = "r-reticulate")
install_keras(method = "conda" ,envname = "r-reticulate", tensorflow = "gpu")



    
    
    
### TEST 

library(ggplot2)
library(reshape2)
library(tensorflow)
library(keras)
library(reticulate)

tensorflow::use_condaenv("r-reticulate")

k = backend()     # Check if keras loads


