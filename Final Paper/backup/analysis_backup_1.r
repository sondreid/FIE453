################################################################################
################################ Analysis #####################################
################################################################################
source("preprocessing.R")
# Libraries
library(keras)






######### TRYING TO TRAIN AN ENTIRE MODEL #########




######## Train and test split

most_important_variables_list <- most_important_variables$features[1:5] # Select only most important variables for predicting RETX


df_full <- merged %>% select(retx, permno,  most_important_variables_list) %>% 
  remove_cols_only_zero_and_NA(print_removed_cols = T) %>% 
  remove_NA(0.2, print_removed_cols = T) %>% 
  remove_nzv(print_removed_cols = T) %>% 
  remove_hcv(0.9, print_removed_cols = T)

df_full %<>% 
  remove_NA_rows() # Remove NA rows

train_test <- perform_train_test_split(df_full, 
                                       train_ratio = 0.8)
train_df <- train_test[[1]]
test_df <- train_test[[2]]

# Check for similar rows
train_df %>% inner_join(test_df, by = "permno") %>% nrow()




# KNN ----------------------------------------------------------------
# Should be far faster than RF, SVM, etc


tunegrid_knn <- expand.grid(k = 5:25)


knn <- train(retx~.,
             data = train_df,
             trControl  = train_control,
             method = "knn",
             metric = "MAE", # Which metric makes the most sense to use RMSE or MAE. Leaning towards MAE
             tunegrid = tunegrid_knn,
             preProcess = c("center","scale", "pca"),
             allowParalell=TRUE)


knn

knn$results$MAE %>% min() # Validation accuracy
summary(knn)



# SVM ----------------------------------------------------------------


tunegrid_svm <- expand.grid(C = seq(0, 2, length = 20)) # Try variations of margin C

svm                    <- caret::train(retx~.,
                                       data = train_df,
                                       method = "svmRadial",
                                       metric = "MAE", # Which metric makes the most sense to use RMSE or MAE. Leaning towards MAE
                                       trControl  = train_control, 
                                       tunegrid = tunegrid_svm,
                                       preProcess = c("center", "scale", "pca"),
                                       allowParallel=TRUE)



svm$results$MAE %>% min() # Validation accuracy



# GBM ----------------------------------------------------------------


tunegrid_gbm <-  expand.grid(interaction.depth = c(1, 5, 9), 
                             n.trees = (1:30)*50, 
                             shrinkage = 0.1,
                             n.minobsinnode = 20)

gbm <- train(retx~.,
             data = train_df,
             method = "gbm",
             preProcess = c("center","scale"),
             tuneGrid = tunegrid_gbm,
             trControl = train_control)





########## MODEL SELECTION
# Based on model test performance metrics


modelList <- list(svm, gbm, knn) # List of all models


evaluate_models <- function(modelList, test_df) {
  
  #' Outputs a tibble of test and validation metrics


  model_performance <- tibble()
  for (model in modelList) {
    prediction <- predict(model, newdata = test_df)
    test_performance_metrics <- postResample(pred = prediction, obs = test_df$retx)
    validation_accuracy %<>% bind_rows(
      tibble(       "Model name" =  model$method,
                    "Validation RMSE" = model$results$RMSE %>% min(),
                    "Validation MAE" = model$results$MAE %>% min(),
                    "Test RMSE" = test_performance_metrics[[1]],
                    "Test MAE" = test_performance_metrics[[3]]))
                                       
    
  }
  return (model_performance)
}


evaluate_models(list(knn), test_df)




# Stop cluster
stopCluster(cl)







##################### ##


### Select stocks based on predictability



# merge with company names

test_df <- test_df %>% left_join(company_names_df, by = "permno")



select_stocks <- function(test_df, selected_model) {
  ## TODO: NOT FINISHED
  companies <- test_df$permno %>% unique()
  company_predictability <- tibble()
  for (company in companies) {
    company_data <- test_df %>% 
      filter(permno == company)
    company_predictions <- predict(selected_model, company_data)
    company_performance_metrics <- postResample(pred = prediction, obs = test_df$retx)
    company_predictability %<>% bind_rows(
      tibble("Company name" = company_data$comnam %>% unique() )
    ) 
    
    
  }
  
  
}
