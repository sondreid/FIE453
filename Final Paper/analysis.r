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
set_random_seed (42, disable_gpu = FALSE) # Set seed for reproducability, both tensorflow and R native seed
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
mlp_grid<-expand.grid(layer1=32,
                      layer2=16,
                      layer3=9)


multi_hidden_layer_model <- caret::train(retx ~ ., 
                  data       = train_df_all, 
                  preProcess = c("center", "scale"),
                  trControl  = train_control, 
                  tuneGrid   = mlp_grid,
                  metric     = "MAE",
                  verbose = T,
                  allowParalell = T,
                  method     = "mlpML")

multi_hidden_preds <- predict(multi_hidden_layer_model, test_df_all)
postResample(multi_hidden_preds, test_df_all$retx)



### Keras multilayer

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

early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)



build_nn_model_5_layers <- function(selected_train_df, selected_spec) {
  input <- layer_input_from_dataset(selected_train_df %>% dplyr::select(-retx))
  
  output <- input %>% 
    layer_dense_features(dense_features(selected_spec)) %>% 
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 8, activation = "relu") %>%
    layer_dense(units = 4,  activation = "relu") %>%
    layer_dense(units = 2,  activation = "relu") 
  
  model <- keras_model(input, output)
  
  return (model)
}


build_nn_model_3_layers <- function(selected_train_df, selected_spec) {
  input <- layer_input_from_dataset(selected_train_df %>% dplyr::select(-retx))
  
  output <- input %>% 
    layer_dense_features(dense_features(selected_spec)) %>% 
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 8,  activation = "relu") 
  
  model <- keras_model(input, output)
  return(model)
  
}

build_nn_model_1_layer <- function(selected_train_df, selected_spec) {
  input <- layer_input_from_dataset(selected_train_df %>% dplyr::select(-retx))
  
  output <- input %>% 
    layer_dense_features(dense_features(selected_spec)) %>% 
    layer_dense(units = 32, activation = "relu") 
    
    model <- keras_model(input, output)
  return(model)
  
}

nn_model_1_layer_reduced <- build_nn_model_1_layer(train_df_reduced, spec_reduced)
nn_model_3_layers_reduced <- build_nn_model_3_layers(train_df_reduced, spec_reduced)
nn_model_5_layers_reduced <- build_nn_model_5_layers(train_df_reduced, spec_reduced)


grid_search_nn_model <- function(selected_model, model_train_df, model_test_df, learning_rates, momentums,
                                 epochs, batch_sizes, verbose) {
  
  best_MAE <- Inf
  best_model <- NA
  best_history <- NA
  for (lr in learning_rates) {
    for (momentum in momentums) {
      for (epoch in epochs) {
        for (batch_size in batch_sizes) {
            new_model <- selected_model %>% 
              compile(
                loss = "mse", 
                optimizer = optimizer_sgd(
                  learning_rate =lr,
                  momentum = momentum),
                metrics = list("mean_absolute_error"))
            
            
            history <- new_model %>% 
                  fit(
                  x = model_train_df %>% dplyr::select(-retx),
                  y = model_train_df$retx,
                  epochs = epoch,
                  batch_size = batch_size,
                  validation_split = 0.2,
                  verbose = verbose,
                  callbacks = list(print_dot_callback, early_stop) #Print simplified dots, and stop learning when validation improvements stalls
              )
          c(loss, mae) %<-% (new_model %>% evaluate(model_test_df %>% dplyr::select(-retx), model_test_df$retx, verbose = 0))
          if (mae < best_MAE) {
            best_history <- history
            best_model <- new_model
            best_MAE <- mae
          }
       }
      }
    }
  }
  return (best_model)

}


grid_search_nn_model_ada_delta <- function(model, model_train_df, model_test_df,
                                         epochs, batch_sizes, verbose) {
  
  best_MAE <- Inf
  best_model <- NA
  best_history <- NA
    for (epoch in epochs) {
        for (batch_size in batch_sizes) {
          new_model <- model %>% 
            compile(
              loss = "mse", 
              optimizer = optimizer_adadelta(),
              metrics = list("mean_absolute_error"))
          
          
          history <- new_model %>% 
            fit(
              x = model_train_df %>% dplyr::select(-retx),
              y = model_train_df$retx,
              epochs = epoch,
              batch_size = batch_size,
              validation_split = 0.2,
              verbose = verbose,
              callbacks = list(print_dot_callback, early_stop) #Print simplified dots, and stop learning when validation improvements stalls
            )
          c(loss, mae) %<-% (new_model %>% evaluate(model_test_df %>% dplyr::select(-retx), model_test_df$retx, verbose = 0))
          if (mae < best_MAE) {
            best_history <- history
            best_model <- new_model
            best_MAE <- mae
          }

    }
  }
  return (list(best_model, best_history))
  
}

grid_search_nn_model_rmsprop <- function(model, model_train_df, model_test_df, learning_rates, momentums,
                                 epochs, batch_sizes, verbose) {
  
  best_MAE <- Inf
  best_model <- NA
  for (lr in learning_rates) {
    for (momentum in momentums) {
      for (epoch in epochs) {
        for (batch_size in batch_sizes) {
          new_model <- model %>% 
            compile(
              loss = "mse", 
              optimizer = optimizer_rmsprop(learning_rate = lr),
              metrics = list("mean_absolute_error"))
          
          
          history <- new_model %>% 
            fit(
              x = model_train_df %>% dplyr::select(-retx),
              y = model_train_df$retx,
              epochs = epoch,
              batch_size = batch_size,
              validation_split = 0.2,
              verbose = verbose,
              callbacks = list(print_dot_callback, early_stop) #Print simplified dots, and stop learning when validation improvements stalls
            )
          c(loss, mae) %<-% (new_model %>% evaluate(model_test_df %>% dplyr::select(-retx), model_test_df$retx, verbose = 0))
          if (mae < best_MAE) {
            best_model <- history
            best_MAE <- mae
          }
        }
      }
    }
  }
  return (best_model)
  
}

best_model_rms_prop  <- grid_search_nn_model_rmsprop(nn_model_5_layers_reduced, train_df_reduced, test_df_reduced, 
                                             learning_rates = list(0.001, 0.01),
                                             momentums = list(0),
                                             batch_sizes = list(50, 80, 400, 1000, 10000),
                                             epochs = list(20, 50, 70, 300),
                                             verbose = 0
                                             
)

best_model_nn_5_layer <- grid_search_nn_model(nn_model_5_layers_reduced, train_df_reduced, test_df_reduced, 
                                   learning_rates = list(0.01, 0.0001),
                                   momentums = list(0, 0.001),
                                   batch_sizes = list(100, 500, 1500),
                                   epochs = list(30, 100, 200, 500),
                                   verbose = 0
                                   
                                   )


predictions_5_nn_model <- best_model_nn_5_layer %>% predict(test_df_reduced %>% dplyr::select(-retx))
predictions_5_nn_model[ , 1]

postResample(predictions_5_nn_model[ , 1], test_df_reduced$retx)


# Better than 0 benchmark?
make_0_benchmark(test_df_reduced) 
make_0_benchmark(test_df_reduced)[[3]] > postResample(predictions_5_nn_model[ , 1], test_df_reduced$retx)[[3]]


#rms prop
predictions_5_nn_model_rms_prop <- best_model_rms_prop %>% predict(test_df_reduced %>% dplyr::select(-retx))
predictions_5_nn_model_rms_prop[ , 1]

postResample(predictions_5_nn_model[ , 1], test_df_reduced$retx)




## Ada delta

best_model_ada_delta <- grid_search_nn_model_ada_delta(nn_model_5_layers_reduced, train_df_reduced, test_df_reduced, 
                                                       batch_sizes = list(50, 80, 400, 1000, 10000),
                                                       epochs = list(20, 50, 70, 300),
                                                       verbose = 0)



predictions_5_nn_model_ada_delta <- best_model_ada_delta[[1]] %>% predict(test_df_reduced %>% dplyr::select(-retx))
predictions_5_nn_model_ada_delta[ , 1]

postResample(predictions_5_nn_model_ada_delta[ , 1], test_df_reduced$retx)




summary(nn_model_5_layers)
nn_model_5_layers  %>% save_model_tf(file = "models/5_layer_nn_model")

c(loss, mae) %<-% (nn_model_5_layers %>% evaluate(test_df_all %>% dplyr::select(-retx), test_df_all$retx, verbose = 0))

paste0("> Mean absolute error on test set Three layer model: ", sprintf("%.4f", mae))

# Predict 
predictions_5_nn_model <- nn_model_5_layers %>% predict(test_df_all %>% dplyr::select(-retx))
predictions_5_nn_model[ , 1]

postResample(predictions_5_nn_model[ , 1], test_df_all$retx)



### Three layers

best_model_nn_3_layer <- grid_search_nn_model(nn_model_3_layers_reduced, train_df_reduced, test_df_reduced, 
                                   learning_rates = list(0.01, 0.0001),
                                   momentums = list(0, 0.001),
                                   batch_sizes = list(100, 500, 1500),
                                   epochs = list(30, 100, 200, 500),
                                   verbose = 0
                                   
)


predictions_3_nn_model <- best_model %>% predict(test_df_reduced %>% dplyr::select(-retx))
predictions_3_nn_model[ , 1]

postResample(predictions_3_nn_model[ , 1], test_df_reduced$retx)



### One layers

best_model_nn_1_layer <- grid_search_nn_model(nn_model_1_layer_reduced, train_df_reduced, test_df_reduced, 
                                   learning_rates = list(0.01, 0.0001),
                                   momentums = list(0, 0.001),
                                   batch_sizes = list(100, 500, 1500),
                                   epochs = list(30, 100, 200, 500),
                                   verbose = 0
                                   
)


predictions_1_nn_model <- best_model_nn_1_layer %>% predict(test_df_reduced %>% dplyr::select(-retx))
predictions_1_nn_model[ , 1]

postResample(predictions_1_nn_model[ , 1], test_df_reduced$retx)








# Train Control 

parts <- createDataPartition(train_df$retx, times = 1, p = 0.2) # 20 % of training data is used for validation (i.e, hyperparameter selection)

train_control <- trainControl(method = "cv", #Method does not matter as parts dictate 20 % validation of training set
                              index = parts, 
                              number = 10,
                              verboseIter = T,
                              savePredictions = T,
                              summaryFunction = defaultSummary)




# Enable parallel processing
num_cores <- detectCores() - 3
cl <- makePSOCKcluster(num_cores) # Use most cores, or specify
registerDoParallel(cl)





# KNN --------------------------------------------------------------------------
# Should be far faster than RF, SVM, etc



# Tune grid of KNN
tunegrid_knn <- expand.grid(k = 5:25)


# Training the KNN model and evaluating with MAE (Mean Average Error)
knn_model <- train(retx ~ .,
                   data          = train_df_reduced,
                   trControl     = train_control,
                   method        = "knn",
                   metric        = "MAE",                                       # Which metric makes the most sense to use RMSE or MAE. Leaning towards MAE
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
bayesian_ridge_model <- train(retx ~ ., 
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
gam_model <- train(retx ~ ., 
                   data       = train_df, 
                   preProcess = c("center", "scale"),
                   trControl  = train_control, 
                   tunegrid   = tunegrid_gam,
                   metric     = "MAE",
                   method     = "gam")






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
                   allowParalell = T,
                   trControl  = train_control)

gbm_preds <- predict(gbm_model, test_df_all)
postResample(gbm_preds, test_df_all$retx)

# Saving the models ------------------------------------------------------------
save(knn_model, gbm_model, bayesian_ridge_model, file = "models/models.Rdata")



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





# Stop cluster
stopCluster(cl)

