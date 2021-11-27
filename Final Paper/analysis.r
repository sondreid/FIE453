################################################################################
################################ Analysis #####################################
################################################################################
source("preprocessing.R")
# Libraries
library(keras)






######### TRYING TO TRAIN AN ENTIRE MODEL #########




######## Train and test split #####################################




df_large <- get_subset_of_companies_ratio(merged, 0.6) %>% dplyr::select(retx, permno,  top_5_most_important_features)

df_large %<>% 
  remove_NA_rows() # Remove NA rows

low_observation_count_companies <- find_company_observations(df_large, 60)

df_large %<>% anti_join(low_observation_count_companies) # Cut companies with fewer than 50 observations (they cannot be reliably predicted)


train_test <- perform_train_test_split(df_large, 
                                       train_ratio = 0.8) # Split into train and test set with seperate sets of companies
train_df <- train_test[[1]]
test_df <- train_test[[2]]



# Check for similar rows
train_df %>% inner_join(test_df, by = "permno") %>% nrow()
train_df %<>% dplyr::select(-permno) # Remove company numbers from training

####################################
###### load or run models

load(file = "model_results/models.Rdata")

# KNN ----------------------------------------------------------------
# Should be far faster than RF, SVM, etc



tunegrid_knn <- expand.grid(k = 5:25)


knn_model <- train(retx~.,
                   data = train_df,
                   trControl  = train_control,
                   method = "knn",
                   metric = "MAE", # Which metric makes the most sense to use RMSE or MAE. Leaning towards MAE
                   tunegrid = tunegrid_knn,
                   preProcess = c("center","scale"),
                   allowParalell=TRUE)

knn_model


knn_model$results$MAE %>% min() # Validation accuracy
summary(knn_model)



# Weighted KNN ----------------------------------------------------------------




tunegrid_knn_weighted <- expand.grid(kmax = c(5, 9, 15), 
                                     distance = seq(1,20), 
                                      kernel = c('gaussian',
                                                 'triangular',
                                                 'rectangular',
                                                 'epanechnikov',
                                                 'optimal'))

knn_weighted_model <- train(retx~.,
             data = train_df,
             trControl  = train_control,
             method = "kknn",
             metric = "MAE", # Which metric makes the most sense to use RMSE or MAE. Leaning towards MAE
             tunegrid = tunegrid_knn_weighted,
             preProcess = c("center","scale"),
             allowParalell=TRUE)

knn_weighted_model


knn_weighted_model$results$MAE %>% min() # Validation accuracy
summary(knn_weighted_model)




#### Which models are computationally efficient and yield good results?



# Bayesian ridge regression ----------------------------------------------------------------


bayesian_ridge_model <- train(retx~., 
                              data = train_df %>% head(50000), 
                              preProcess = c("center", "scale"),
                              trControl  = train_control, 
                              tuneLength = 10,
                              metric = "MAE",
                              method  = "bridge")

bayesian_ridge_model
bayesian_ridge_model$results$MAE %>% min() # Validation accuracy







# Generalized additive model ----------------------------------------------------------------

tunegrid_gam <-  expand.grid(method = c("GCV", "REML"),
                            select = list(T, F))


gam_model <- train(retx~., 
                   data = train_df, 
                   preProcess = c("center", "scale"),
                   trControl  = train_control, 
                   tunegrid = tunegrid_gam,
                   metric = "MAE",
                   method  = "gam")



gam_model
gam_model$results$MAE %>% min() # Validation accuracy







# SVM ----------------------------------------------------------------


tunegrid_svm <- expand.grid(C = seq(0, 2, length = 20)) # Try variations of margin C

svm_model                    <- caret::train(retx~.,
                                       data = train_df,
                                       method = "svmRadial",
                                       metric = "MAE", # Which metric makes the most sense to use RMSE or MAE. Leaning towards MAE
                                       trControl  = train_control, 
                                       #tunegrid = tunegrid_svm,
                                       tuneLength = 4,
                                       preProcess = c("center", "scale", "pca"),
                                       allowParallel=TRUE)



svm_model$results$MAE %>% min() # Validation accuracy



# GBM ----------------------------------------------------------------


tunegrid_gbm <-  expand.grid(interaction.depth = c(1, 5, 9), 
                             n.trees = (1:30)*50, 
                             shrinkage = 0.1,
                             n.minobsinnode = 20)

gbm_model <- train(retx~.,
             data = train_df %>% head(50000),
             method = "gbm",
             preProcess = c("center","scale"),
             metric = "MAE", # Which metric makes the most sense to use RMSE or MAE. Leaning towards MAE
             #tuneGrid = tunegrid_gbm,
             trControl = train_control)




#save(knn_model, svm_model, gbm_model, file = "model_results/models.Rdata")
save(knn_model, bayesian_ridge_model, file = "model_results/models.Rdata")




########## MODEL SELECTION
# Based on model test performance metrics


modelList <- list(knn_model, gam_model) # List of all models


evaluate_models <- function(modelList, test_df) {
  
  #' Outputs a tibble of test and validation metrics
  
  
  model_performance <- tibble()
  for (model in modelList) {
    test_predictions <- predict(model, newdata = test_df)
    train_predictions <- predict(model, newdata = train_df)
    test_performance_metrics <- postResample(pred = test_predictions, obs = test_df$retx)
    train_performance_metrics <- postResample(pred = train_predictions, obs = train_df$retx)
    model_performance %<>% bind_rows(
      tibble(       "Model name" =  model$method,
                    "Training RMSE" = train_performance_metrics[[1]],
                    "Training MAE" = train_performance_metrics[[3]],
                    "Test RMSE" = test_performance_metrics[[1]],
                    "Test MAE" = test_performance_metrics[[3]]))
    
    
  }
  return (model_performance)
}


model_evaluation <- evaluate_models(list(knn_model), test_df)

model_evaluation %>% 
  kable(caption = "Performance metrics of tested models", digits=3)  %>% 
  kable_classic(full_width = F, html_font = "Times New Roman")  %>% 
  save_kable("images/evaluation_metrics_all_models.png",   zoom = 1.5, density = 1000)









##################### ##


### Select stocks based on predictability




get_company_name <- function(permno) {
  #'
  #'@description: Returns the name of a company based on its company identification number
  company_name <- company_names_df %>% 
    filter(permno == permno) 
   return( company_name$comnam[0])
}



select_stocks <- function(test_df, selected_model) {
  #' @description: Selects stocks based on predictability.
  ## TODO: NOT FINISHED
  companies <- test_df$permno %>% unique()
  #test_df  %<>% left_join(company_names_df, by = "permno") # merge with company names
  company_predictability <- tibble()
  for (company in companies) {
    company_data <- test_df %>% 
      filter(permno == company)
    company_predictions <- predict(selected_model, company_data)
    company_performance_metrics <- postResample(pred = prediction, obs = test_df$retx)
    company_predictability %<>% bind_rows(
      tibble("Company name" = get_company_name(company_data$permno[1]),
             "Test RMSE" = company_performance_metrics[[1]],
             "Test MAE" = company_performance_metrics[[3]])
    ) 
    
    
  }
  return (company_predictability)
  
}


selected_stocks <- select_stocks(test_df, knn_model)

# Stop cluster
stopCluster(cl)

