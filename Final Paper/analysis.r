################################################################################
################################ Analysis ######################################
################################################################################
# Candidates:



# Importing data from preprocessing script
source("preprocessing.R")
load(file = "model_results/features.Rdata")


# Libraries
library(keras)
library(kknn)



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
load(file = "model_results/models.Rdata")





# KNN --------------------------------------------------------------------------
# Should be far faster than RF, SVM, etc


# Tune grid of KNN
tunegrid_knn <- expand.grid(k = 5:25)


# Training the KNN model and evaluating with MAE (Mean Average Error)
knn_model <- train(retx ~ .,
                   data          = train_df,
                   trControl     = train_control,
                   method        = "knn",
                   metric        = "MAE",                                       # Which metric makes the most sense to use RMSE or MAE. Leaning towards MAE
                   tunegrid      = tunegrid_knn,
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
                                     distance = seq(1,20), 
                                     kernel = c('gaussian',
                                                'triangular',
                                                'rectangular',
                                                'epanechnikov',
                                                'optimal')
                                     )

# Training the Weighted KNN-model
knn_weighted_model <- train(retx ~ .,
                            data          = train_df,
                            trControl     = train_control,
                            method        = "kknn",
                            metric        = "MAE",                              # Which metric makes the most sense to use RMSE or MAE. Leaning towards MAE
                            tunegrid      = tunegrid_knn_weighted,
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
bayesian_ridge_model <- train(retx~., 
                              data = train_df %>% head(50000), 
                              preProcess = c("center", "scale"),
                              trControl  = train_control, 
                              tuneLength = 10,
                              metric = "MAE",
                              method  = "bridge")


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


# Looking at the GAM-model
gam_model


# Finding the GAM-model that minimizes MAE
gam_model$results$MAE %>% min() # Validation accuracy




# Neural network with feature extraction ---------------------------------------
# Tune grid of NN
tunegrid_nn <-  expand.grid(size  = c(5, 20, 70),
                            decay = c(0.001, 0.1, 0.2))


# Training the NN-model
nn_model <- train(retx ~ ., 
                   data       = train_df, 
                   preProcess = c("center", "scale"),
                   trControl  = train_control, 
                   tunegrid   = tunegrid_nn,
                   metric     = "MAE",
                   method     = "nnet")



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
#save(knn_model, svm_model, gbm_model, file = "model_results/models.Rdata")
save(knn_model, bayesian_ridge_model, file = "model_results/models.Rdata")



################################################################################
########################### MODEL SELECTION ####################################
################################################################################
# Based on model test performance metrics

# List of all models
modelList <- list(knn_model, gam_model) 



evaluate_models <- function(modelList, test_df) {
  
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


# Running the model evaluation function
model_evaluation <- evaluate_models(list(knn_model), test_df)


# Printing the model evaluation results with kable extra
model_evaluation %>% 
  kable(caption = "Performance metrics of tested models", 
        digits  = 3) %>% 
  kable_classic(full_width = F, 
                html_font  = "Times New Roman")  %>% 
  save_kable("images/evaluation_metrics_all_models.png", 
             zoom = 1.5, 
             density = 1000)




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
  
  #test_df  %<>% left_join(company_names_df, by = "permno") # merge with company names
  
  company_predictability <- tibble()
  
  for (company in companies) {
    company_data <- test_df %>% 
      filter(permno == company)
    
    company_predictions <- predict(selected_model, company_data)
    company_performance_metrics <- postResample(pred = company_predictions, 
                                                obs = test_df$retx)
    
    company_predictability %<>% bind_rows(
      tibble("Company name"       = get_company_name(company_data$permno[1]),
             "Company identifier" = company_data$permno[1],
             "Test RMSE"          = company_performance_metrics[[1]],
             "Test MAE"           = company_performance_metrics[[3]])
    ) 
  }
  
  return (company_predictability)
  
}


# Running the stock selection function 
selected_stocks <- select_stocks(test_df, knn_model)


# Printing the selected stocks with kable extra
selected_stocks %>% 
  arrange(desc("Test MAE")) %>% 
  kable(caption = "10 stocks of highest predictability", 
        digits  = 3)  %>% 
  kable_classic(full_width = F, 
                html_font = "Times New Roman") %>% 
  save_kable("images/predictable_stocks.png", 
             zoom = 1.5, 
             density = 1000)


# Stop cluster
stopCluster(cl)

