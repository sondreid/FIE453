
################################################################################
########################### MODEL EVALUATION ####################################
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
  
  
  train_df_reduced_scaled <- train_df_reduced %>% dplyr::select(-costat) %>% 
    scale_py() %>% 
    as_tibble() %>% 
    mutate(costat = train_df_reduced$costat)
  
  
  test_df_reduced_scaled <- test_df_reduced %>% dplyr::select(-costat) %>% 
    scale_py() %>% 
    as_tibble() %>% 
    mutate(costat = test_df_reduced$costat)
  
  model_performance <- tibble()
  
  for (model in modelList) {
    test_predictions          <- predict(model, newdata = test_df_reduced_scaled)
    train_predictions         <- predict(model, newdata = train_df_reduced_scaled)
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

modelList <- list(knn_model,gbm_model, bayesian_ridge_model)   # List of all models
"Uncomment to perform model evaluation"
model_evaluation <- evaluate_models(modelList, train_df,  test_df)  %>%  arrange(`Test MAE`)
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






