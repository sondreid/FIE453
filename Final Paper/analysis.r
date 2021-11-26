################################################################################
################################ Analysis #####################################
################################################################################
source("preprocessing.R")
# Libraries
library(keras)








######### TRYING TO TRAIN AN ENTIRE MODEL




# Train and test split

most_important_variables_list <- most_important_variables$features[1:5]


df_full <- merged %>% select(retx, permno,  most_important_variables_list) %>% 
  remove_cols_only_zero_and_NA(print_removed_cols = T) %>% 
  remove_NA(0.2, print_removed_cols = T) %>% 
  remove_nzv(print_removed_cols = T) %>% 
<<<<<<< HEAD
  remove_hcv(0.9, print_removed_cols = T) #%>% 
  #replace_NA_with_mean(print_replaced_cols = T)
=======
  remove_hcv(0.9, print_removed_cols = T)

df_full %<>% 
  remove_NA_rows() # Remove NA rows
>>>>>>> 8564034b3a9ad969b2f5fa7d2feaa963061c3a08

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
                                       data = train_df_reduced,
                                       method = "svmRadial",
                                       metric = "MAE", # Which metric makes the most sense to use RMSE or MAE. Leaning towards MAE
                                       trControl  = train_control, 
                                       tunegrid = tunegrid_svm,
                                       preProcess = c("center", "scale", "pca"),
                                       allowParallel=TRUE)



svm$results$MAE %>% min() # Validation accuracy




# Stop cluster
stopCluster(cl)






#### Functions for model analysis ##############################################



plot_confusion_matrix <- function(obs, preds) {
  #' @description: Function that plots confusion matrix 
  #' @obs: Observations in vector format
  #' @preds: predictions in vector format
  conf_data <- tibble(obs = obs, preds =preds )
  confusion_matrix <- conf_mat(conf_data, truth = "obs", estimate = "preds")
  autoplot(confusion_matrix, type = "heatmap") +
    theme(text = element_text(size = 25))    
}




make_table <- function(obs, preds, model_name) {
  #' @description: Function that produces a table of performance metrics: Accuracy, preciscion and recall
  #' @obs: Observations in vector format
  #' @preds: predictions in vector format
  #' @return: A dataframe 
  obs <- as.factor(obs)
  preds <- as.factor(preds)
  table_df <- tibble("Accuracy" = confusionMatrix(preds, obs)[[3]][1], 
                     "Recall" = caret::recall(table(obs, preds)),
                     "Preciscion" = caret::precision(table(obs, preds)))  %>% 
    t()  
  colnames(table_df) <- "Measure"
  table_df  %<>% 
    kbl(caption = model_name)  %>% 
    kable_classic(full_width = F, html_font = "Times New Roman")
  return (table_df)
}

#####################


### Select stocks


select_stocks <- function(test_df, validated_model) {
  
  
  
}