################################################################################
################################ Analysis ######################################
################################################################################
# Candidates: FILL IN CANDIDATE NUMBERS



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


### Load keras and tensorflow ###

library(tensorflow)
library(keras)
library(tfdatasets)
library(reticulate)
#set_random_seed (42, disable_gpu = FALSE) # Set seed for reproducability, both tensorflow and R native seed
set.seed(1)
conda_python(envname = "r-reticulate") # Create miniconda enviroment (if not already done)
tensorflow::use_condaenv("r-reticulate") # Specify enviroment to tensorflow





######### Dataframe with all companies using only variance and correlation filter#############
# Load all 
load(file = "cached_data/train_test.Rdata")
# Or run the code in preprocessing


##### Scale numeric features ####


scale_py <- function(x) {
  #' R implementation which corresponds to python standard_scaler()
  n <- nrow(x)
  scaled_df <- scale(x, scale= apply(x, 2, sd) * sqrt(n-1/n)) 
  return(scaled_df)
  
}


train_df_scaled <- train_df %>% dplyr::select(-costat,  -retx) %>% 
  scale_py() %>% 
  as_tibble() %>% 
  mutate(costat = train_df$costat) %>% 
  mutate(retx = train_df$retx) # Add in retx without scaling 


test_df_scaled <- test_df %>% dplyr::select(-costat,  -retx) %>% 
  scale_py() %>% 
  as_tibble() %>% 
  mutate(costat = test_df$costat) %>% 
  mutate(retx = test_df$retx) # Add in retx without scaling




train_df_reduced_scaled <- train_df_reduced %>% dplyr::select(-costat, -retx) %>% 
  scale_py() %>% 
  as_tibble() %>% 
  mutate(costat = train_df_reduced$costat) %>% 
  mutate(retx = train_df_reduced$retx)


test_df_reduced_scaled <- test_df_reduced %>% dplyr::select(-costat,  -retx) %>% 
  scale_py() %>% 
  as_tibble() %>% 
  mutate(costat = test_df_reduced$costat) %>% 
  mutate(retx = test_df_reduced$retx)





################################################################################
######################### Load or run models ###################################
################################################################################

# In order to save run time one can choose to load the model results
load(file = "models/models.Rdata")



load_models <- function() {
  
  load(file = "models/models.Rdata") # Load knn, gbm, ridge regression
  ## Load NN models
  
  load(file = "models/1_nn_layer_model_history.Rdata")
  best_model_nn_1_layer_all <- list( load_model_hdf5("models/1_layer_nn_model.hdf5"), load(file = "models/1_nn_layer_model_history.Rdata"))
  load(file = "models/2_nn_layer_model_all.Rdata")
  best_model_nn_2_layer_all <- load_model_hdf5("models/2_layer_nn_model.hdf5")
 
  
}





make_0_benchmark <- function(selected_test_df) {
  #' Makes a zero benchmark to compare models
  #' @
  benchmark_0 <- postResample(rep(0, nrow(selected_test_df)), selected_test_df$retx)
  
  return (benchmark_0)
  
}






# Neural network using specified number of layers  ---------------------------------------




print_dot_callback <- callback_lambda(
  #' Simplified callback, showing dots instead of full loss/validation error plots
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
) 



build_nn_model_5_layers <- function(selected_train_df, selected_spec, batch_normalization, dropout_rate) {
  input <- layer_input_from_dataset(selected_train_df %>% dplyr::select(-retx))
  
  output <- input %>% 
    layer_dense_features(dense_features(selected_spec)) %>% 
    layer_dense(units = 32, activation = "relu") %>%
    layer_dropout(dropout_rate)
  
  if (batch_normalization == T) {output %<>% layer_batch_normalization()}
  
  output  %<>% 
    layer_dense(units = 16, activation = "relu") %>%
    layer_dropout(dropout_rate)
  
  if (batch_normalization == T) {output %<>% layer_batch_normalization()}
  output  %<>% 
    layer_dense(units = 8, activation = "relu") %>%
    layer_dropout(dropout_rate)
  
  
  if (batch_normalization == T) {output %<>% layer_batch_normalization()}
  output  %<>% 
    layer_dense(units = 4, activation = "relu") %>%
    layer_dropout(dropout_rate) 
  
  if (batch_normalization == T) {output %<>% layer_batch_normalization()}
  
  output %<>%   layer_dense(units = 2) 
  
  
  
  
  model <- keras_model(input, output)
  
  return (model)
}

build_nn_model_4_layers <- function(selected_train_df, selected_spec, batch_normalization, dropout_rate) {
  input <- layer_input_from_dataset(selected_train_df %>% dplyr::select(-retx))
  
  output <- input %>% 
    layer_dense_features(dense_features(selected_spec)) %>% 
    layer_dense(units = 32, activation = "relu") %>%
    layer_dropout(dropout_rate)
  
  if (batch_normalization == T) {output %<>% layer_batch_normalization()}
  
  output  %<>% 
    layer_dense(units = 16, activation = "relu") %>%
    layer_dropout(dropout_rate)
  
  
  if (batch_normalization == T) {output %<>% layer_batch_normalization()}
  output  %<>% 
    layer_dense(units = 8, activation = "relu") %>%
    layer_dropout(dropout_rate)
  
  if (batch_normalization == T) {output %<>% layer_batch_normalization()}
    
  output %<>% layer_dense(units = 4) 
  
  
  
  model <- keras_model(input, output)
  
  return (model)
}




build_nn_model_3_layers <- function(selected_train_df, selected_spec, batch_normalization, dropout_rate) {
  input <- layer_input_from_dataset(selected_train_df %>% dplyr::select(-retx))
  
  output <- input %>% 
    layer_dense_features(dense_features(selected_spec)) %>% 
    layer_dense(units = 32, activation = "relu") %>%
    layer_dropout(dropout_rate)
  
    if (batch_normalization == T) {output %<>% layer_batch_normalization()}
  
    output  %<>% 
      layer_dense(units = 16, activation = "relu") %>%
      layer_dropout(dropout_rate)
    
    if (batch_normalization == T) {output %<>% layer_batch_normalization()}
    output %<>% 
      layer_dense(units = 8) 

    
  
  model <- keras_model(input, output)
  return(model)
  
}





build_nn_model_2_layers <- function(selected_train_df, selected_spec, batch_normalization, dropout_rate) {
  



  input <- layer_input_from_dataset(selected_train_df %>% dplyr::select(-retx))
  
  output <- input %>% 
    layer_dense_features(dense_features(selected_spec)) %>% 
    layer_dense(units = 32, activation = "relu") %>%
    layer_dropout(dropout_rate)
  
  if (batch_normalization == T) {output %<>% layer_batch_normalization()}
  
  output  %<>% 
    layer_dense(units = 16) 
  
  
  model <- keras_model(input, output)

  return(model)
  
}



build_nn_model_1_layer <- function(selected_train_df, selected_spec) {
  #'
  #'@description: 
  #'
  input <- layer_input_from_dataset(selected_train_df %>% dplyr::select(-retx))
  
  output <- input %>% 
    layer_dense_features(dense_features(selected_spec)) %>% 
    layer_dense(units = 32) 
  
  model <- keras_model(input, output)
  return(model)
  
}


build_model <- function(selected_train_df, selected_spec, num_layers, batch_normalization, dropout_rate) {
  #'
  #'@description: wrapper function which calls on the adequate nn-model building function based on the number of layers
  #'wanted.
  #'
  #'
  if      (num_layers == 1) {
    output_model = build_nn_model_1_layer(selected_train_df, selected_spec)
  }
  
  else if (num_layers == 2) {
    output_model = build_nn_model_2_layers(selected_train_df, selected_spec, batch_normalization, dropout_rate)
    
  }
  else if (num_layers == 3) {
    output_model = build_nn_model_3_layers(selected_train_df, selected_spec, batch_normalization, dropout_rate)
   
  }
  else if (num_layers == 4) {
    output_model = build_nn_model_4_layers(selected_train_df, selected_spec, batch_normalization, dropout_rate)
  }
  
  else if (num_layers == 5) {
    output_model = build_nn_model_5_layers(selected_train_df, selected_spec, batch_normalization, dropout_rate)
    
  }
  

  return (output_model)
}







grid_search_nn_model_generaL_optimizer <- function(model_train_df, dropout_rates, num_layers, 
                                                   learning_rates, epochs, batch_sizes, patience_list,  verbose) {
  
  selected_spec <- feature_spec(model_train_df, retx ~ . ) %>% 
    step_numeric_column(all_numeric(), -costat) %>% # Scale numeric features
    step_categorical_column_with_vocabulary_list(costat) %>%  # non-numeric variables
    fit()
  
  
  best_MAE <- Inf
  best_model <- NA
  best_history <- NA
  batch_normalizations <- list(T, F)
  if (num_layers == 1) {batch_normalizations <- list(F)}
  for (dropout_rate in dropout_rates) {
    for (learn_rate in learning_rates) {
      for (batch_normalization in batch_normalizations) {
        for (batch_size in batch_sizes) {
          for (patience in patience_list) {
            selected_model = build_model(model_train_df, selected_spec, num_layers, batch_normalization, dropout_rate)
            reduce_lr = callback_reduce_lr_on_plateau(monitor = "val_loss", patience = patience)
            new_early_stop = callback_early_stopping(monitor = "val_loss", patience = patience + 10)
            
            
            
            optimizer = optimizer_adam(learning_rate = learn_rate) # Adam optimizer at different starting learning rates
            new_model = selected_model %>% 
              compile(
                loss = "mse", 
                optimizer = optimizer,
                metrics = list("mean_absolute_error"))
            
            
            new_history = new_model %>% 
              fit(
                x = model_train_df %>% dplyr::select(-retx),
                y = model_train_df$retx,
                epochs = epochs,
                batch_size = batch_size,
                validation_split = 0.2,
                verbose = verbose,
                use_multiprocessing = T, 
                callbacks = list(reduce_lr, new_early_stop) #Print simplified dots, and stop learning when validation improvements stalls
              )
            mae_list_length =  new_history$metrics$val_mean_absolute_error %>% length()
            new_mae =  new_history$metrics$val_mean_absolute_error[[mae_list_length]]
            print(paste("> MAE of new model", new_mae))
            if (new_mae < best_MAE) {
              best_history <- new_history
              best_model <- new_model
              best_MAE <- new_mae
              print(paste(">New best model. MAE of new model", best_MAE))
              print(paste(">Batch size", batch_size))
              print(paste(">dropout_rate", dropout_rate))
              print(paste("> Learning rate", learn_rate))
              print(paste("> Patience", patience))
              print(paste(">Batch normalization", batch_normalization))
            }
          }
          
        }
      }
    }
  }
  return (list(best_model, best_history))
  
}





############################## TESTING #################################




## Single layer

best_model_nn_1_layer_test <- grid_search_nn_model_generaL_optimizer(train_df_reduced_scaled, 
                                                                     dropout_rates = list(0),
                                                                     num_layers = 1,
                                                                     batch_sizes = list(300, 500, 1000, 7000),
                                                                     epochs = 200,
                                                                     learning_rates = list(0.005, 0.1, 0.4, 0.6),
                                                                     patience_list = list(1, 5, 10, 20),
                                                                     verbose = 0
                                                                     
)




predictions_1_nn_model <- best_model_nn_1_layer_test[[1]] %>% predict(test_df_reduced_scaled %>% dplyr::select(-retx, -permno))
predictions_1_nn_model[ , 1]

postResample(predictions_1_nn_model[ , 1], test_df_reduced$retx)


# Two layers

## Single layer

best_model_nn_2_layers_test <- grid_search_nn_model_generaL_optimizer(train_df_reduced_scaled, 
                                                                     dropout_rates = list(0, 0.3, 0.4),
                                                                     num_layers = 2,
                                                                     batch_sizes = list(300, 500, 1000, 7000),
                                                                     epochs = 200,
                                                                     learning_rates = list(0.005, 0.1, 0.4),
                                                                     patience_list = list(1, 2,10, 5, 20,25),
                                                                     verbose = 0
                                                                     
)


predictions_2_nn_model <- best_model_nn_2_layers_test[[1]] %>% predict(test_df_reduced_scaled %>% dplyr::select(-retx, -permno))
predictions_2_nn_model[ , 1]

postResample(predictions_2_nn_model[ , 1], test_df_reduced$retx)




## Three layer

best_model_nn_3_layer_test <- grid_search_nn_model_generaL_optimizer(train_df_reduced_scaled, 
                                                                     dropout_rates = list(0,0.3, 0.4),
                                                                     num_layers = 3,
                                                                     batch_sizes = list(500, 1000, 7000),
                                                                     epochs = 200,
                                                                     learning_rates = list(0.005, 0.1, 0.4),
                                                                     patience_list = list(1, 5, 10, 20),
                                                                     verbose = 0
                                                                     
)

                                                                     



predictions_3_nn_model <- best_model_nn_3_layer_test[[1]] %>% predict(test_df_reduced_scaled %>% dplyr::select(-retx, -permno))
predictions_3_nn_model[ , 1]

postResample(predictions_3_nn_model[ , 1], test_df_reduced$retx)



## Five layer
best_model_nn_5_layer_test <- grid_search_nn_model_generaL_optimizer(train_df_reduced, 
                                                                     dropout_rates = list(0, 0.1, 0.3, 0.4),
                                                                     num_layers = 5,
                                                                     batch_sizes = list(500, 1000, 7000),
                                                                     epochs = 200,
                                                                     optimizer = sgd_opt,
                                                                     patience_list = list(1, 2, 5, 20,25, 40, 50),
                                                                     verbose = 0
                                                                     
)


predictions_5_nn_model <- best_model_nn_5_layer_adam[[1]] %>% predict(test_df_reduced %>% dplyr::select(-retx))
predictions_5_nn_model[ , 1]

postResample(predictions_5_nn_model[ , 1], test_df_reduced_scaled$retx)


#### Test save





best_model_nn_1_layer_test <- grid_search_nn_model_generaL_optimizer(train_df_reduced_scaled, 
                                                                     dropout_rates = list(0),
                                                                     num_layers = 1,
                                                                     batch_sizes = list(500, 1000, 7000),
                                                                     epochs = 200,
                                                                     learning_rates = list(0.001, 0.1, 0.9),
                                                                     patience_list = list(1, 5, 20,25),
                                                                     verbose = 0
)




predictions_1_nn_model <- best_model_nn_1_layer_test[[1]] %>% predict(test_df_reduced_scaled %>% dplyr::select(-retx, -permno))
predictions_1_nn_model[ , 1]

postResample(predictions_1_nn_model[ , 1], test_df_reduced$retx)






#########################################################

# Better than 0 benchmark?
make_0_benchmark(test_df) 





#### RUN ON entire dataset




best_model_nn_1_layer_all  <- grid_search_nn_model_generaL_optimizer(train_df_scaled, 
                                                                    dropout_rates = list(0),
                                                                    num_layers = 1,
                                                                    batch_sizes = list(300, 500, 1000),
                                                                    epochs = 200,
                                                                    learning_rates = list(0.005, 0.1, 0.4, 0.6),
                                                                    patience_list = list(1, 2,10, 5, 20),
                                                                    verbose = 0
                                                                                                
)





best_model_nn_2_layers_all <-  grid_search_nn_model_generaL_optimizer(train_df_scaled, 
                                                                       dropout_rates = list(0, 0.25, 0.4),
                                                                       num_layers = 2,
                                                                       batch_sizes = list(500, 1000, 7000),
                                                                       epochs = 200,
                                                                       learning_rates = list(0.005, 0.1, 0.4),
                                                                       patience_list = list(1, 5,10, 20),
                                                                       verbose = 0
                                                                       
)




best_model_nn_3_layers_all <- grid_search_nn_model_generaL_optimizer(train_df_scaled, 
                                                                     dropout_rates = list(0, 0.25, 0.4),
                                                                     num_layers = 3,
                                                                     batch_sizes = list(500, 1000, 7000),
                                                                     epochs = 200,
                                                                     learning_rates = list(0.005, 0.1, 0.4),
                                                                     patience_list = list(1, 5, 20),
                                                                     verbose = 0
                                                                     
)

best_model_nn_4_layers_all <- grid_search_nn_model_generaL_optimizer(train_df_scaled, 
                                                                     dropout_rates = list(0,0.25, 0.4),
                                                                     num_layers = 4,
                                                                     batch_sizes = list(500, 1000, 7000),
                                                                     epochs = 200,
                                                                     learning_rates = list(0.005, 0.1, 0.4),
                                                                     patience_list = list(1, 5, 20),
                                                                     verbose = 0
                                                                     
)





best_model_nn_5_layers_all <- grid_search_nn_model_generaL_optimizer(train_df_scaled, 
                                                                     dropout_rates = list(0,  0.25, 0.4),
                                                                     num_layers = 5,
                                                                     batch_sizes = list(500, 1000, 7000),
                                                                     epochs = 200,
                                                                     learning_rates = list(0.005, 0.1, 0.4),
                                                                     patience_list = list(1, 5, 20),
                                                                     verbose = 0
                                                                     
)






############## Model predictions

## 1 NN


predictions_1_nn_model <- best_model_nn_1_layer_all[[1]] %>% predict(test_df_scaled %>% dplyr::select(-retx, -permno))
predictions_1_nn_model[ , 1]

postResample(predictions_1_nn_model[ , 1], test_df_scaled$retx)


## 2 NN
predictions_2_nn_model <- best_model_nn_2_layers_all[[1]] %>% predict(test_df_scaled %>% dplyr::select(-retx, -permno))
predictions_2_nn_model[ , 1]

postResample(predictions_2_nn_model[ , 1], test_df_scaled$retx)

## 3 NN
predictions_3_nn_model <- best_model_nn_3_layers_all[[1]] %>% predict(test_df_scaled %>% dplyr::select(-retx, -permno))
predictions_3_nn_model[ , 1]

postResample(predictions_3_nn_model[ , 1], test_df_scaled$retx)


## 4 NN
predictions_4_nn_model <- best_model_nn_4_layers_all[[1]] %>% predict(test_df_scaled %>% dplyr::select(-retx, -permno))
predictions_4_nn_model[ , 1]

postResample(predictions_4_nn_model[ , 1], test_df_scaled$retx)


################# Saving models #########################

## 1 NN

best_model_nn_1_layer_history <- best_model_nn_1_layer_all[[2]]
save(best_model_nn_1_layer_history, file = "models/1_nn_layer_model_history.Rdata") # Save model history
best_model_nn_1_layer_all[[1]]  %>% save_model_hdf5("models/1_layer_nn_model.hdf5") # Save model

## 2 NN

best_model_nn_2_layers_history <- best_model_nn_2_layers_all[[2]]
save(best_model_nn_2_layers_history, file = "models/2_nn_layers_model_history.Rdata") # Save model history
best_model_nn_2_layers_all[[1]]  %>% save_model_hdf5("models/2_layer_nn_model.hdf5") # Save model


## 3 NN
best_model_nn_3_layers_history <- best_model_nn_3_layers_all[[2]]
save(best_model_nn_3_layers_history, file = "models/3_nn_layers_model_history.Rdata") # Save model history
best_model_nn_3_layers_all[[1]]  %>% save_model_hdf5("models/3_layer_nn_model.hdf5") # Save model


## 4 NN
best_model_nn_4_layers_history <- best_model_nn_4_layers_all[[2]]
save(best_model_nn_4_layers_history, file = "models/4_nn_layer_model_history.Rdata") # Save model history
best_model_nn_4_layers_all[[1]]  %>% save_model_hdf5("models/4_layers_nn_model.hdf5") # Save model









# Train Control 

parts <- createDataPartition(train_df$retx, times = 1, p = 0.2) # 20 % of training data is used for validation (i.e, hyperparameter selection)

train_control <- trainControl(method = "cv", #Method does not matter as parts dictate 20 % validation of training set
                              index = parts, 
                              savePredictions = T)




# Enable parallel processing
num_cores <- detectCores() - 3
cl <- makePSOCKcluster(num_cores) # Use most cores, or specify
registerDoParallel(cl)





# KNN --------------------------------------------------------------------------
# Should be far faster than RF, SVM, etc



# Tune grid of KNN
tunegrid_knn <- expand.grid(k = 5:25)


# Training the KNN model and evaluating with MAE (Mean Average Error)
knn_model <- caret::train(retx ~ .,
                   data          = train_df_scaled,
                   trControl     = train_control,
                   method        = "knn",
                   metric        = "MAE",                             
                   tuneLength      = 20,
                   allowParalell = TRUE)


# Looking at the KNN-model
knn_model


# Finding the KNN-model that minimizes MAE
knn_model$results$MAE %>% min() # Validation accuracy


# Summary of KNN-model
summary(knn_model)









# Bayesian ridge regression ----------------------------------------------------
# Training the Ridge-model
bayesian_ridge_model <- caret::train(retx ~ ., 
                              data       = train_df_scaled, 
                              trControl  = train_control, 
                              metric     = "MAE",
                              method     = "bridge")


# Looking at the Ridge model
bayesian_ridge_model


# Finding the Ridge-model that minimizes MAE
bayesian_ridge_model$results$MAE %>% min() # Validation accuracy


bayesian_ridge_preds <- predict(bayesian_ridge_model, test_df_scaled)
postResample(bayesian_ridge_preds, test_df$retx)


# Generalized additive model ---------------------------------------------------


#### NOT USED ###
# Tune grid of GAM-model
tunegrid_gam <-  expand.grid(method = c("GCV", "REML"),
                             select = list(T, F))


# Training the GAM-model
gam_model <- caret::train(retx ~ ., 
                   data       = train_df_scaled, 
                   trControl  = train_control, 
                   tuneLength   = 10,
                   metric     = "MAE",
                   method     = "gam")


gam_preds <- predict(gam_model, test_df_scaled)
postResample(gam_preds, test_df$retx)



# GBM --------------------------------------------------------------------------
# Tune grid of GBM-model
tunegrid_gbm <-  expand.grid(interaction.depth = c(1, 5, 9), 
                             n.trees           = c(50, 200, 700, 1300), 
                             shrinkage         = 0.1,
                             n.minobsinnode    = 20)

# Training the GBM-model
gbm_model <- caret::train(retx ~ .,
                   data       = train_df_scaled,
                   method     = "gbm",
                   metric     = "MAE",                                             
                   tuneLength   = 25,
                   trControl  = train_control)

gbm_preds <- predict(gbm_model, test_df_scaled)
postResample(gbm_preds, test_df$retx)

# Saving the models ------------------------------------------------------------
save(knn_model, bayesian_ridge_model,gbm_model, file = "models/models.Rdata")



# Stop cluster
stopCluster(cl)

