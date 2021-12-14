########################### Keras installation ###

" 
This file contains the needed code to install a miniconda enviroment and install python packages Tensorflow and Keras
"

## First installation
library(reticulate)
conda_version()
conda_list()
conda_python(envname = "r-reticulate")    
conda_install(envname = "r-reticulate",packages="r-reticulate")
conda_install(envname = "r-reticulate",packages="r-tensorflow")
conda_install(envname = "r-reticulate",packages="r-tfdatasets")
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



    
    
    
### Verify correctness of keras and tensorflow installation

library(ggplot2)
library(reshape2)
library(tensorflow)
library(keras)
library(reticulate)

tensorflow::use_condaenv("r-reticulate")

k = backend()     # Check if keras loads


