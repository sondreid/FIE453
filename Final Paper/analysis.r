################################################################################
################################ Analysis ######################################
################################################################################
# Candidates:



# Importing data from preprocessing script
source("preprocessing.R")
load(file = "model_results/features.Rdata")


### Load keras and tensorflow ###

library(tensorflow)
library(keras)
library(tfdatasets)
library(reticulate)
#set_random_seed (42, disable_gpu = FALSE) # Set seed for reproducability, both tensorflow and R native seed
set.seed(1)
conda_python(envname = "r-reticulate") # Create miniconda enviroment (if not already done)
tensorflow::use_condaenv("r-reticulate") # Specify enviroment to tensorflow





################################################################################
########################## Train and Test Split ################################
################################################################################

######### Dataframe with all companies using only variance and correlation filter#############
# Load all 
load(file = "cached_data/train_test.Rdata")
# Or run the code in preprocessing





################################################################################
######################### Load or run models ###################################
################################################################################

# In order to save run time one can choose to load the model results
load(file = "models/models.Rdata")





make_0_benchmark <- function(selected_test_df) {
  #' Makes a zero benchmark to compare models
  #' @
  benchmark_0 <- postResample(rep(0, nrow(selected_test_df)), selected_test_df$retx)
  
  return (benchmark_0)
  
}







# Neural network using specified number of layers  ---------------------------------------

spec <- feature_spec(train_df, retx ~ . ) %>% 
  step_numeric_column(all_numeric(), -costat, normalizer_fn = scaler_standard()) %>% # Scale numeric features
  step_categorical_column_with_vocabulary_list(costat) %>%  # non-numeric variables
  fit()


spec_reduced <- feature_spec(train_df_reduced, retx ~ . ) %>% 
    step_numeric_column(all_numeric(), -costat, normalizer_fn = scaler_standard()) %>% # Scale numeric features
    step_categorical_column_with_vocabulary_list(costat) %>%  # non-numeric variables
  fit()
  



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
    layer_dropout(dropout_rate) %>% 
    layer_dense(units = 2) 
  
  
  
  
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
  input <- layer_input_from_dataset(selected_train_df %>% dplyr::select(-retx))
  
  output <- input %>% 
    layer_dense_features(dense_features(selected_spec)) %>% 
    layer_dense(units = 32) 
  
  model <- keras_model(input, output)
  return(model)
  
}


build_model <- function(selected_train_df, selected_spec, num_layers, batch_normalization, dropout_rate) {
  if (num_layers == 1) {
    output_model = build_nn_model_1_layer(selected_train_df, selected_spec)
    
  }
  
  else if (num_layers == 2) {
    output_model = build_nn_model_2_layers(selected_train_df, selected_spec, batch_normalization, dropout_rate)
    
  }
  else if (num_layers == 3) {
    output_model = build_nn_model_3_layers(selected_train_df, selected_spec, batch_normalization, dropout_rate)
   
  }
  else if (num_layers == 5) {
    output_model = build_nn_model_5_layers(selected_train_df, selected_spec, batch_normalization, dropout_rate)
    
  }
  return (output_model)
}






grid_search_nn_model_generaL_optimizer <- function(model_train_df, dropout_rates, num_layers, 
                                                   epochs, batch_sizes, optimizer, patience_list,  verbose) {
  
  selected_spec <- feature_spec(model_train_df, retx ~ . ) %>% 
    step_numeric_column(all_numeric(), -costat, normalizer_fn = scaler_standard()) %>% # Scale numeric features
    step_categorical_column_with_vocabulary_list(costat) %>%  # non-numeric variables
    fit()
  
  
  best_MAE <- Inf
  best_model <- NA
  best_history <- NA
  batch_normalizations <- list(T, F)
  for (dropout_rate in dropout_rates) {
    for (batch_normalization in batch_normalizations) {
    for (batch_size in batch_sizes) {
      for (patience in patience_list) {
        selected_model = build_model(model_train_df, selected_spec, num_layers, batch_normalization, dropout_rate)
        
        reduce_lr = callback_reduce_lr_on_plateau(monitor = "val_loss", patience = patience)
        new_early_stop = callback_early_stopping(monitor = "val_loss", patience = patience + 10)
        
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
            callbacks = list(print_dot_callback, reduce_lr, new_early_stop) #Print simplified dots, and stop learning when validation improvements stalls
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
          print(paste("> Patience", patience))
          print(paste(">Batch normalization", batch_normalization))
        }
      }
      
     }
    }
  }
  return (list(best_model, best_history))
  
}




nn_model_1_layer_reduced <- build_nn_model_1_layer(train_df_reduced, spec_reduced)
nn_model_2_layers_reduced        <- build_nn_model_2_layers(train_df_reduced, spec_reduced)
nn_model_3_layers_reduced <- build_nn_model_3_layers(train_df_reduced, spec_reduced)
nn_model_5_layers_reduced <- build_nn_model_5_layers(train_df_reduced, spec_reduced)



nn_model_1_layer <- build_nn_model_1_layer(train_df, spec)
nn_model_2_layers <- build_nn_model_2_layers(train_df, spec)
nn_model_3_layers <- build_nn_model_3_layers(train_df, spec)
nn_model_5_layers <- build_nn_model_5_layers(train_df, spec)




#load nn models



load("models/3_nn_layer_model_history.Rdata")


### ADAM optimizer

adam_opt = optimizer_adam()
sgd_opt = optimizer_sgd(learning_rate = 0.8)



############################## TESTING #################################

## Single layer

best_model_nn_1_layer_test <- grid_search_nn_model_generaL_optimizer(train_df_reduced, 
                                                                      dropout_rates = list(0, 0.1, 0.3, 0.4),
                                                                      num_layers = 1,
                                                                      batch_sizes = list(500, 1000, 7000),
                                                                      epochs = 200,
                                                                      optimizer = sgd_opt,
                                                                      patience_list = list(1, 2, 5, 20,25, 40, 50),
                                                                      verbose = 0
                                                                      
)



predictions_1_nn_model <- best_model_nn_1_layer_test[[1]] %>% predict(test_df_reduced %>% dplyr::select(-retx, -permno))
predictions_1_nn_model[ , 1]

postResample(predictions_1_nn_model[ , 1], test_df_reduced$retx)


# Two layers

## Single layer

best_model_nn_2_layers_test <- grid_search_nn_model_generaL_optimizer(train_df_reduced, 
                                                                     dropout_rates = list(0, 0.1, 0.3, 0.4),
                                                                     num_layers = 2,
                                                                     batch_sizes = list(500, 1000, 7000),
                                                                     epochs = 200,
                                                                     optimizer = sgd_opt,
                                                                     patience_list = list(1, 2, 5, 20,25, 40, 50),
                                                                     verbose = 0
                                                                     
)


predictions_2_nn_model <- best_model_nn_2_layers_test[[1]] %>% predict(test_df_reduced %>% dplyr::select(-retx, -permno))
predictions_2_nn_model[ , 1]

postResample(predictions_2_nn_model[ , 1], test_df_reduced$retx)




## Three layer

best_model_nn_3_layer_test <- grid_search_nn_model_generaL_optimizer(train_df_reduced, 
                                                                     dropout_rates = list(0, 0.1, 0.3, 0.4),
                                                                     num_layers = 3,
                                                                     batch_sizes = list(500, 1000, 7000),
                                                                     epochs = 200,
                                                                     optimizer = sgd_opt,
                                                                     patience_list = list(1, 2, 5, 20,25, 40, 50),
                                                                     verbose = 0
                                                                     
)


predictions_3_nn_model <- best_model_nn_3_layer_test[[1]] %>% predict(test_df_reduced %>% dplyr::select(-retx, -permno))
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

postResample(predictions_5_nn_model[ , 1], test_df_reduced$retx)


#########################################################

# Better than 0 benchmark?
make_0_benchmark(test_df_reduced) 



#### RUN ON entire dataset




best_model_nn_1_layer_all  <- grid_search_nn_model_generaL_optimizer(train_df_reduced, 
                                                                    dropout_rates = list(0, 0.1, 0.3, 0.4),
                                                                    num_layers = 1,
                                                                    batch_sizes = list(300, 500, 1000, 7000),
                                                                    epochs = 200,
                                                                    optimizer = sgd_opt,
                                                                    patience_list = list(5, 20,25, 40, 50),
                                                                    verbose = 0
                                                                                                
)


predictions_1_nn_model <- best_model_nn_1_layer_all[[1]] %>% predict(test_df %>% dplyr::select(-retx))
predictions_1_nn_model[ , 1]

postResample(predictions_1_nn_model[ , 1], test_df$retx)

## Do not save unless certain
#save(best_model_nn_1_layer_all, file = "models/1_nn_layer_model_all.Rdata")
#best_model_nn_1_layer_all[[1]]  %>% save_model_tf("models/1_layer_nn_model") # Save model


## two layers


best_model_nn_2_layer_all <-  grid_search_nn_model_generaL_optimizer(train_df_reduced, 
                                                                       dropout_rates = list(0, 0.1, 0.3, 0.4),
                                                                       num_layers = 2,
                                                                       batch_sizes = list(300, 500, 1000, 7000),
                                                                       epochs = 200,
                                                                       optimizer = sgd_opt,
                                                                       patience_list = list(5, 20,25, 40, 50),
                                                                       verbose = 0
                                                                       
)




best_model_nn_3_layers_all <- grid_search_nn_model_generaL_optimizer(train_df_reduced, 
                                                                     dropout_rates = list(0, 0.1, 0.3, 0.4),
                                                                     num_layers = 3,
                                                                     batch_sizes = list(300, 500, 1000, 7000),
                                                                     epochs = 200,
                                                                     optimizer = sgd_opt,
                                                                     patience_list = list(5, 20,25, 40, 50),
                                                                     verbose = 0
                                                                     
)


predictions_3_nn_model <- best_model_nn_3_layers_all[[1]] %>% predict(test_df %>% dplyr::select(-retx))
predictions_3_nn_model[ , 1]

postResample(predictions_3_nn_model[ , 1], test_df$retx)


save(best_model_nn_3_layers_all, file = "models/3_nn_layer_model_history.Rdata") # Save model history
best_model_nn_3_layers_all[[1]]  %>% save_model_tf("models/3_layer_nn_model") # Save model


best_model_nn_5_layers_all <- grid_search_nn_model_generaL_optimizer(train_df_reduced, 
                                                                     dropout_rates = list(0, 0.1, 0.3, 0.4),
                                                                     num_layers = 5,
                                                                     batch_sizes = list(300, 500, 1000, 7000),
                                                                     epochs = 200,
                                                                     optimizer = sgd_opt,
                                                                     patience_list = list(5, 20,25, 40, 50),
                                                                     verbose = 0
                                                                     
)

predictions_5_nn_model <- best_model_nn_5_layers_all[[1]] %>% predict(test_df %>% dplyr::select(-retx))
predictions_5_nn_model[ , 1]

postResample(predictions_5_nn_model[ , 1], test_df$retx)



save(best_model_nn_5_layers_all, file = "models/5_nn_layer_model_all.Rdata")






## Save models

best_model_nn_3_layer_adam[[1]]  %>% save_model_tf("models/3_layer_nn_model")
best_model_nn_3_layer_adam[[2]]  %>% save_model_hdf5("models/3_layer_nn_model")

save(best_model_nn_3_layer_adam, file = "models/3_nn_layer_model_history.Rdata")





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
                   data          = train_df,
                   trControl     = train_control,
                   method        = "knn",
                   metric        = "MAE",                             
                   tuneLength      = 10,
                   preProcess    = c("center", "scale"),
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
                              data       = train_df, 
                              preProcess = c("center", "scale"),
                              trControl  = train_control, 
                              tuneLength = 10,
                              metric     = "MAE",
                              method     = "bridge")


# Looking at the Ridge model
bayesian_ridge_model


# Finding the Ridge-model that minimizes MAE
bayesian_ridge_model$results$MAE %>% min() # Validation accuracy





# Generalized additive model ---------------------------------------------------
# Tune grid of GAM-model
tunegrid_gam <-  expand.grid(method = c("GCV", "REML"),
                             select = list(T, F))


# Training the GAM-model
gam_model <- caret::train(retx ~ ., 
                   data       = train_df, 
                   preProcess = c("center", "scale"),
                   trControl  = train_control, 
                   tuneLength   = 10,
                   metric     = "MAE",
                   method     = "gam")


gam_preds <- predict(gam_model, test_df)
postResample(gam_preds, test_df$retx)



# GBM --------------------------------------------------------------------------
# Tune grid of GBM-model
tunegrid_gbm <-  expand.grid(interaction.depth = c(1, 5, 9), 
                             n.trees           = (1:30)*50, 
                             shrinkage         = 0.1,
                             n.minobsinnode    = 20)

# Training the GBM-model
gbm_model <- caret::train(retx ~ .,
                   data       = train_df,
                   method     = "gbm",
                   preProcess = c("center","scale"),
                   metric     = "MAE",                                             
                   tuneLength   = 10,
                   trControl  = train_control)

gbm_preds <- predict(gbm_model, test_df)
postResample(gbm_preds, test_df$retx)

# Saving the models ------------------------------------------------------------
save(knn_model, bayesian_ridge_model, gam_model,gbm_model, file = "models/models.Rdata")



# Stop cluster
stopCluster(cl)


################################################################################
########################### MODEL SELECTION ####################################
################################################################################
# Based on model test performance metrics





evaluate_models <- function(modelList, train_df, test_df) {
  
  #' @description     Function that evaluates the model both on the training set
  #'                  and the test set by returning RMSE and MAE
  #' 
  #' @param modelList Passing a list with fitted models
  #' @param train_df    Dataframe of training data
  #' #' @param test_df   Passing test data frame
  #' @return          Returns a tibble of train and test metrics
  
  model_performance <- tibble()
  
  for (model in modelList) {
    test_predictions          <- predict(model, newdata = test_df)
    train_predictions         <- predict(model, newdata = train_df)
    test_performance_metrics  <- postResample(pred = test_predictions, 
                                              obs = test_df$retx)
    train_performance_metrics <- postResample(pred = train_predictions, 
                                              obs = train_df$retx)
    
    model_performance %<>% bind_rows(
      tibble(
             "Model name"    =  model$method,
             "Training RMSE" = train_performance_metrics[[1]],
             "Training MAE"  = train_performance_metrics[[3]],
             "Test RMSE"     = test_performance_metrics[[1]],
             "Test MAE"      = test_performance_metrics[[3]]
             )
      )
  }
  
  return (model_performance)
  
}

modelList <- list(knn_model, multi_hidden_layer_model, nn_model, gam_model, bayesian_ridge_model)   # List of all models
"Uncomment to perform model evaluation"
#model_evaluation <- evaluate_models(modelList, train_df,  test_df)  %>%  arrange(`Test MAE`)
#save(model_evaluation, file = "model_results/model_evalution.Rdata")

load(file = "model_results/model_evalution.Rdata")

# Printing the model evaluation results with kable extra
model_evaluation %>% 
  arrange(`Test MAE`) %>% 
  kable(caption = "Performance metrics of tested models", 
        digits  = 4) %>% 
  kable_classic(full_width = F, 
                html_font  = "Times New Roman")  %>% 
  save_kable("images/evaluation_metrics_all_models.png", 
             zoom = 1.5, 
             density = 1500)



# Statically typing names for models
model_evaluation %>% 
  arrange(`Test MAE`) %>% 
  mutate("Model name" = c("Neural Net 10 neurons", "Bayesian Ridge Regression", "Neural Net 5 neurons", "Generalized Additive Models", "K-Nearest Neighbors")) %>% 
  kable(caption = "Performance metrics of tested models", 
        digits  = 4) %>% 
  kable_classic(full_width = F, 
                html_font  = "Times New Roman")  %>% 
  save_kable("images/evaluation_metrics_all_models.png", 
             zoom = 1.5, 
             density = 1500)



################### Compared to a benchmark model ########################################




###### A benchmark model which only predicts 0 in returns


make_0_benchmarK(test_df ) %>% 
tibble("Test MAE" = benchmark_0_results[[3]],
       "Model" = "0 return prediction") %>% 
  kable(caption = "Performance of 0-prediction model", 
        digits  = 4) %>% 
  kable_classic(full_width = F, 
                html_font  = "Times New Roman") %>% 
  save_kable("images/0_return_prediction.png", 
             zoom = 3, 
             density = 1500)

### On reduced set

make_0_benchmarK(test_df_reduced ) %>% 
  tibble("Test MAE" = benchmark_0_results_all[[3]],
       "Model" = "0 return prediction") %>% 
  kable(caption = "Performance of 0-prediction model", 
        digits  = 4) %>% 
  kable_classic(full_width = F, 
                html_font  = "Times New Roman")



################################################################################
###################### Select stocks based on predictability ###################
################################################################################



select_stocks <- function(test_df, selected_model) {
  
  #' @description:         Function that selects stocks based on predictability
  #'                       with their performance metrics
  #' 
  #' @param test_df        Passing a test data frame 
  #' @param selected_model Passing a selected model
  #' @return               Companies with highest predictability
  
  companies <- test_df$permno %>% unique()
  
  
  company_predictability <- tibble()
  
  for (company in companies) {
    company_data <- test_df %>% 
      filter(permno == company)
    
    company_predictions <- predict(selected_model, company_data)
    company_performance_metrics <- postResample(pred = company_predictions, 
                                                obs = company_data$retx)
    
    company_predictability %<>% bind_rows(
      tibble("Company name"       = get_company_name(company_data$permno[1]),
             "Company identifier" = company_data$permno[1],
             "Test RMSE"          = company_performance_metrics[[1]],
             "Test MAE"           = company_performance_metrics[[3]])
    ) 
  }
  
  return (company_predictability)
  
}

select_stocks_always_0 <- function(test_df) {
  
  #' @description:         Function that selects stocks based on predictability
  #'                       with their performance metrics
  #' 
  #' @param test_df        Passing a test data frame 
  #' @return               Companies with highest predictability
  
  companies <- test_df$permno %>% unique()
  
  
  company_predictability <- tibble()
  
  for (company in companies) {
    company_data <- test_df %>% 
      filter(permno == company)
    

    company_performance_metrics <- postResample(pred = rep(0, nrow(company_data)), 
                                                obs = company_data$retx)
    
    company_predictability %<>% bind_rows(
      tibble("Company name"       = get_company_name(company_data$permno[1]),
             "Company identifier" = company_data$permno[1],
             "Test RMSE"          = company_performance_metrics[[1]],
             "Test MAE"           = company_performance_metrics[[3]])
    ) 
  }
  
  return (company_predictability)
  
}

always_0_stocks <- select_stocks_always_0(test_df)






selected_stock_company_info <- function(selected_stocks, test_df,  n) {
  #' @description:    Makes a summary of feature means of selcted stocks (companies)
  #' 
  #' @sele
  #' @test_df: orginal test set from which the selcted companies are drawn
  #' @n: number of included companies
  selected_stocks <- selected_stocks %>% head(n)
  company_info <- tibble()
  for (i in 1:nrow(selected_stocks)) {

    mean_marketcap <- test_df %>%
      filter(permno == selected_stocks[i, ]$`Company identifier`) %>% 
      summarise(mean_marketcap = mean(marketcap))
    
    mean_volume <- test_df %>%
      filter(permno == selected_stocks[i, ]$`Company identifier`) %>% 
      summarise(mean_volume = mean(vol))
    
    mean_cash <- test_df %>%
      filter(permno == selected_stocks[i, ]$`Company identifier`) %>% 
      summarise(mean_cash = mean(chq))
    
    
    mean_operating_income <- test_df %>%
      filter(permno == selected_stocks[i, ]$`Company identifier`) %>% 
      summarise(mean_operating_income = mean(oiadpq))
    
    company_info %<>% bind_rows(
      tibble("Company name" = stringr::str_to_title(selected_stocks[i, ]$`Company name`),
             "Mean market cap" = mean_marketcap$mean_marketcap,
             "Mean volume" = mean_volume$mean_volume,
             "Mean cash" = mean_cash$mean_cash,
             "Mean operating income" = mean_operating_income$mean_operating_income )
    )
  }
  return (company_info)
  
}

selected_model <- multi_hidden_layer_model 

selected_stocks <- select_stocks(test_df, selected_model) %>%  arrange(`Test MAE`)





# Printing the most predictable stocks
selected_stock_company_info(selected_stocks, test_df, 10) %>% 
  kable(caption = "10 stocks of highest predictability", 
        digits  = 2)  %>% 
  kable_classic(full_width = F, 
                html_font = "Times New Roman") %>% 
  save_kable("images/predictable_stocks.png", 
             zoom = 1.5, 
             density = 1900)



## Summaries for all companies in test sets
mean_marketcap <- test_df %>%
  summarise(mean_marketcap = mean(marketcap))

mean_volume <- test_df %>%
  summarise(mean_volume = mean(vol))

mean_cash <- test_df %>%
  summarise(mean_cash = mean(chq))


mean_operating_income <- test_df %>%
  summarise(mean_operating_income = mean(oiadpq))


all_companies_summary <- 
  tibble("Mean market cap" = mean_marketcap$mean_marketcap,
         "Mean volume" = mean_volume$mean_volume,
         "Mean cash" = mean_cash$mean_cash,
         "Mean operating income" = mean_operating_income$mean_operating_income )


all_companies_summary %>% 
kable(caption = "Company mean metrics of all companies in test set", 
      digits  = 2)  %>% 
  kable_classic(full_width = F, 
                html_font = "Times New Roman") %>% 
  save_kable("images/all_company_summary.png", 
             zoom = 1.5, 
             density = 1900)






