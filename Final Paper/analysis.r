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
  remove_hcv(0.9, print_removed_cols = T) %>% 
  replace_NA_with_mean(print_replaced_cols = T)

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
                                       trControl  = train_control, 
                                       tunegrid = tunegrid_svm,
                                       preProcess = c("center", "scale", "pca"),
                                       allowParallel=TRUE)



svm$results$MAE %>% min() # Validation accuracy




# Stop cluster
stopCluster(cl)




### Select stocks


select_stocks <- function(test_df, validated_model) {
  
  
  
}