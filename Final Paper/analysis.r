################################################################################
################################ Analysis ######################################
################################################################################
# Candidates:



# Importing data from preprocessing script
source("preprocessing.R")
load(file = "model_results/features.Rdata")




################################################################################
########################## Train and Test Split ################################
################################################################################


top_5_most_important_features <- most_important_features$features[1:5]



df_large <- 
  get_subset_of_companies_ratio(merged, 0.6) %>% 
  dplyr::select(retx, permno, top_5_most_important_features) %>% 
  remove_NA_rows() # Remove NA rows



low_observation_count_companies <- find_company_observations(df_large, 60)
df_large %<>% anti_join(low_observation_count_companies)                        # Cut companies with fewer than 50 observations (they cannot be reliably predicted)


# Train-Test-Split
train_test <- perform_train_test_split(df_large, 
                                       train_ratio = 0.8)                       # Split into train and test set with seperate sets of companies
train_df <- train_test[[1]]
test_df <- train_test[[2]]


# Check for similar rows
train_df %>% inner_join(test_df, by = "permno") %>% nrow()
train_df %<>% dplyr::select(-permno) # Remove company numbers from training



################################################################################
######################### Load or run models ###################################
################################################################################

# In order to save run time one can choose to load the model results
load(file = "models/models.Rdata")





# KNN --------------------------------------------------------------------------
# Should be far faster than RF, SVM, etc


set.seed(1) # Set seed for reproducability

# Tune grid of KNN
tunegrid_knn <- expand.grid(k = 5:25)


# Training the KNN model and evaluating with MAE (Mean Average Error)
knn_model <- train(retx ~ .,
                   data          = train_df,
                   trControl     = train_control,
                   method        = "knn",
                   metric        = "MAE",                                       # Which metric makes the most sense to use RMSE or MAE. Leaning towards MAE
                   tuneGrid      = tunegrid_knn,
                   preProcess    = c("center", "scale"),
                   allowParalell = TRUE)


# Looking at the KNN-model
knn_model


# Finding the KNN-model that minimizes MAE
knn_model$results$MAE %>% min() # Validation accuracy


# Summary of KNN-model
summary(knn_model)



# Weighted KNN -----------------------------------------------------------------

# Tune grid of Weighted KNN with maximum 5, 9 or 15 nearest neighbor
# using different kernels
tunegrid_knn_weighted <- expand.grid(kmax = c(5, 9, 15), 
                                     distance = seq(5,20), 
                                     kernel = c('gaussian',
                                                'triangular',
                                                'epanechnikov',
                                                'optimal')
                                     )

# Training the Weighted KNN-model
knn_weighted_model <- train(retx ~ .,
                            data          = train_df,
                            trControl     = train_control,
                            method        = "kknn",
                            metric        = "MAE",                              # Which metric makes the most sense to use RMSE or MAE. Leaning towards MAE
                            tuneGrid      = tunegrid_knn_weighted,
                            preProcess    = c("center", "scale"),
                            allowParalell = TRUE)


# Looking at the Weighted KNN-model
knn_weighted_model


# Finding the Weighted KNN-model that minimizes MAE
knn_weighted_model$results$MAE %>% min() # Validation accuracy


# Summary of KNN-model
summary(knn_weighted_model)




#### Which models are computationally efficient and yield good results?



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




# Neural network using specified number of layers  ---------------------------------------
mlp_grid<-expand.grid(layer1=10,
                      layer2=10,
                      layer3=10)


multi_hidden_layer_model <- train(retx ~ ., 
                  data       = train_df, 
                  preProcess = c("center", "scale"),
                  trControl  = train_control, 
                  tuneGrid   = mlp_grid,
                  metric     = "MAE",
                  verbose = T,
                  allowParalell = T,
                  method     = "mlpML")




# Neural network using specified number of layers with weight decay  ---------------------------------------
mlp_grid<-expand.grid(layer1=15,
                      layer2=15,
                      layer3=15,
                      decay = 0.001)


mlp_weight_decay_model <- caret::train(retx ~ ., 
                                  data       = train_df, 
                                  preProcess = c("center", "scale"),
                                  trControl  = train_control, 
                                  tuneGrid   = mlp_grid,
                                  metric     = "MAE",
                                  verbose = T,
                                  allowParalell = T,
                                  method     = "mlpWeightDecayML")

mlp_weight_decay_model_preds <- predict(mlp_weight_decay_model, test_df)
postResample(mlp_weight_decay_model_preds, test_df$retx)



# Neural network  ---------------------------------------
# Tune grid of NN
tunegrid_nn <-  expand.grid(size  = c(5, 7, 15),
                            decay = c(0.001, 0.005, 0.05))


# Training the NN-model
nn_model <- train(retx ~ ., 
                   data       = train_df, 
                   preProcess = c("center", "scale"),
                   trControl  = train_control, 
                   tuneGrid   = tunegrid_nn,
                   metric     = "MAE",
                   verbose = T,
                   allowParalell = T,
                   method     = "nnet")



# Neural network with feature extraction ---------------------------------------
# Tune grid of NN
tunegrid_pca_nn <-  expand.grid(size  = c(5, 10, 15),
                            decay = c(0.001, 0.05))


# Training the NN-model
pca_nn_model <- train(retx ~ ., 
                  data       = train_df, 
                  preProcess = c("center", "scale"),
                  trControl  = train_control, 
                  tuneGrid   = tunegrid_pca_nn,
                  metric     = "MAE",
                  verbose = T,
                  allowParalell = T,
                  method     = "pcaNNet")




# SVM --------------------------------------------------------------------------
# Tune grid of SVM
tunegrid_svm <- expand.grid(C = seq(0, 2, length = 20)) # Try variations of margin C


# Training the SVM-model
svm_model <- caret::train(retx ~ .,
                          data          = train_df,
                          method        = "svmRadial",
                          metric        = "MAE",                                # Which metric makes the most sense to use RMSE or MAE. Leaning towards MAE
                          trControl     = train_control, 
                          #tunegrid     = tunegrid_svm,
                          tuneLength    = 4,
                          preProcess    = c("center", "scale", "pca"),
                          allowParallel = TRUE)


# Looking at the SVM-model
svm_model


# Finding the SVM-model that minimizes MAE
svm_model$results$MAE %>% min() # Validation accuracy





# GBM --------------------------------------------------------------------------
# Tune grid of GBM-model
tunegrid_gbm <-  expand.grid(interaction.depth = c(1, 5, 9), 
                             n.trees           = (1:30)*50, 
                             shrinkage         = 0.1,
                             n.minobsinnode    = 20)

# Training the GBM-model
gbm_model <- train(retx ~ .,
                   data       = train_df %>% head(50000),
                   method     = "gbm",
                   preProcess = c("center","scale"),
                   metric     = "MAE",                                              # Which metric makes the most sense to use RMSE or MAE. Leaning towards MAE
                  #tuneGrid   = tunegrid_gbm,
                   trControl  = train_control)



# Saving the models ------------------------------------------------------------
save(knn_model, pca_nn_model, multi_hidden_layer_model , nn_model, gam_model, bayesian_ridge_model, file = "models/models.Rdata")



################################################################################
########################### MODEL SELECTION ####################################
################################################################################
# Based on model test performance metrics





evaluate_models <- function(modelList, train_df, test_df) {
  
  #' @description     Function that evaluates the model both on the training set
  #'                  and the test set by returning RMSE and MAE
  #' 
  #' @param modelList Passing a list with fitted models
  #' @param test_df   Passing test data frame
  #' @return          Returns a tibble of test and validation metrics
  
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
model_evaluation <- evaluate_models(modelList, train_df,  test_df)  %>%  arrange(`Test MAE`)
save(model_evaluation, file = "model_results/model_evalution.Rdata")


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



#### Compared to a benchmark model

# A benchmark model which only predicts 0 in returns

benchmark_0_results <- postResample(rep(0, nrow(test_df)), test_df$retx)

tibble("Test MAE" = benchmark_0_results[[3]],
       "Model" = "0 return prediction") %>% 
  kable(caption = "Performance of 0-prediction model", 
        digits  = 4) %>% 
  kable_classic(full_width = F, 
                html_font  = "Times New Roman") %>% 
  save_kable("images/0_return_prediction.png", 
             zoom = 1.5, 
             density = 1900)




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

