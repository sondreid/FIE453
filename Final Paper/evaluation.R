
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

evaluate_models_nn <- function(modelList, train_df, test_df) {
  
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
model_evaluation <- evaluate_models(modelList, train_df_scaled,  test_df_scaled)  %>%  arrange(`Test MAE`)




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


get_company_name <- function(input_permno) {
  #'
  #'@description: Returns the name of a company based on its company identification number
  company_name <- company_names_df %>% 
    filter(permno == input_permno) 
  # If several names are registered. Pick the most recent
  company_name %<>% arrange(desc(date))
  return( company_name$comnam[1])
}



stock_level_predictions_caret <- function(selected_test_df, selected_model, verbose = F) {
  
  #' @description:         Function that selects stocks based on predictability
  #'                       with their performance metrics
  #' 
  #' @param selected_test_df        Passing a test data frame 
  #' @param selected_model Passing a selected model
  #' @return               Companies with highest predictability
  
  companies <- selected_test_df$permno %>% unique()
  
  
  company_predictability <- tibble()
  num_companies <- length(companies)
  for (company in companies) {
    
    i <- i+1
    if (verbose) print(paste("> company", i, " of ", num_companies))
    
    company_data <- selected_test_df %>% 
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


stock_level_predictions_nn <- function(selected_test_df, selected_model, verbose = F) {
  
  #' @description:         Function that selects stocks based on predictability
  #'                       with their performance metrics
  #' 
  #' @param test_df        Passing a test data frame 
  #' @param selected_model Passing a selected model
  #' @return               Companies with highest predictability
  
  companies <- selected_test_df$permno %>% unique()
  
  
  company_predictability <- tibble()
  num_companies <- length(companies)
  i <- 0
  for (company in companies) {
    i <- i+1
    if (verbose) print(paste("> company", i, " of ", num_companies))
    
    company_data <- selected_test_df %>% 
      filter(permno == company)
    
    company_predictions <- selected_model %>% predict(company_data %>% dplyr::select(-retx))
   
    company_performance_metrics <- postResample(company_predictions[ , 1], 
                                                company_data$retx)
    company_predictability %<>% bind_rows(
      tibble("Company name"       = get_company_name(company_data$permno[1]),
             "Company identifier" = company_data$permno[1],
             "Test RMSE"          = company_performance_metrics[[1]],
             "Test MAE"           = company_performance_metrics[[3]])
    ) 
  }
  
  return (company_predictability)
  
}


stock_level_predictions_always_zero <- function(selected_test_df) {
  
  #' @description:         Function that selects stocks based on predictability
  #'                       with their performance metrics
  #' 
  #' @param test_df        Passing a test data frame 
  #' @return               Companies with highest predictability
  
  companies <- selected_test_df$permno %>% unique()
  
  
  company_predictability <- tibble()
  
  for (company in companies) {
    company_data <- selected_test_df %>% 
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


#always_0_stocks <- select_stocks_always_0(test_df_scaled)

#selected_stocks_nn_1_layer <- stock_level_predictions_nn(test_df_scaled, best_model_nn_1_layer_all[[1]]) %>%  arrange(`Test MAE`)






mean_metric_stock_level <- function(num_top, selected_test_df,  nn_model_metrics, caret_model_metrics) {
  
  #'
  #'
  #'
  model_performance <- tibble()
  
  for (model in caret_models) {
    stock_level_predictions <- model[[1]] %>%  arrange(`Test MAE`) %>% head(num_top)
    model_mean_mae <- stock_level_predictions$`Test MAE` %>% mean()
    model_mean_rmse <- stock_level_predictions$`Test RMSE` %>% mean()
    
    model_performance %<>% bind_rows(
      tibble(
        "Model name"               = model[[2]],
        "RMSE top stocks"          = model_mean_rmse,
        "MAE top stocks"           = model_mean_mae)
    )
    
    
  }
  for (model in nn_models) {
    stock_level_predictions <- model[[1]] %>%  arrange(`Test MAE`) %>% head(num_top)
    model_mean_mae <- stock_level_predictions$`Test MAE` %>% mean()
    model_mean_rmse <- stock_level_predictions$`Test RMSE` %>% mean()
    
    model_performance %<>% bind_rows(
      tibble(
        "Model name"               = model[[2]],
        "RMSE top stocks"          = model_mean_rmse,
        "MAE top stocks"           = model_mean_mae)
    )
    

  }
  
  
  naive_0_benchmark_predictions <- stock_level_predictions_always_zero(selected_test_df)
  naive_0_benchmark_mae <- naive_0_benchmark_predictions$`Test MAE` %>% mean()
  naive_0_benchmark_rmse <- naive_0_benchmark_predictions$`Test RMSE` %>% mean()
  model_performance %<>% bind_rows(
    tibble(
      "Model name"               = "Zero prediction naive model",
      "RMSE top stocks"          = naive_0_benchmark_rmse,
      "MAE top stocks"           = naive_0_benchmark_mae)
  )
  
  return (model_performance)
  
}

nn_models <- list(
  list(best_model_nn_1_layer_all[[1]], "Neural Network 1 hidden layer"), 
  list(best_model_nn_2_layers_all[[1]], "Neural Network 2 hidden layers"),
  list(best_model_nn_3_layers_all[[1]], "Neural Network 3 hidden layers"),
  list(best_model_nn_4_layers_all[[1]], "Neural Network 4 hidden layers")
)


caret_models <- list(
  list(gbm_model, "Gradient Boosting machine"),
  list(knn_model, "K-nearest neighbors"),
  list(bayesian_ridge_model, "Bayesian ridge regression")
)

model_metrics_stock_level <- mean_metric_stock_level(100, test_df_scaled, nn_models, caret_models) %>% arrange(`MAE top stocks`)

save(model_metrics_stock_level, file = "model_results/stock_level_performance.Rdata")


model_metrics_stock_level %>% 
  kable(caption = "Model performance on 20 most predictable stocks", 
        digits  = 2)  %>% 
  kable_classic(full_width = F, 
                html_font = "Times New Roman") %>% 
  save_kable("images/model_performance_stock_level.png", 
             zoom = 3, 
             density = 1900)


### Based on best performing model, which stocks are predictable ###


nn_model_3_layer_stock_preds <- stock_level_predictions_nn(test_df_scaled, best_model_nn_3_layers_all[[1]])







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



all_company_metrics <- function(selected_test_df) {
  mean_marketcap <- selected_test_df %>%
    summarise(mean_marketcap = mean(marketcap))
  
  mean_volume <- selected_test_df %>%
    summarise(mean_volume = mean(vol))
  
  mean_cash <- selected_test_df %>%
    summarise(mean_cash = mean(chq))
  
  
  mean_operating_income <- selected_test_df %>%
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

}


##### Monthly predictions



predict_monthly_returns_nn <- function(model, stocks, selection_data) {
  
  data_selected_stocks <- selection_data %>% 
    filter(permno %in% stocks)
  
  stock_predictions <- (model %>% predict(data_selected_stocks %>% select(-retx)))[,1]
  
  data_selected_companies %<>%
    mutate(predicted_returns = stock_predictions)
  
  return(data_selected_stocks)
  
  
}


stock_predictions <- predict_monthly_returns_nn(se)

plot_monthly_returns <- function(stock_predictions) {
  #'
  #'Plots series of observed and predicted returns.
  stock_predictions %>% 
    select(date, stock_predictions, retx) %>% 
    pivot_longer(names_to = "type",
                 values_to = "returns") %>% 
  ggplot() +
    geom_line(aes(x = date, y = returns, col = "type"), lwd = 1.06) +
    guides(colour = guide_legend("Series name")) +
    theme_bw() +
    theme(legend.position = "bottom") +
    labs(x = "Date", y = "Returns", title ="Predicted vs observed return in 2018-2019") +
    scale_colour_manual(values=c("orange"))
  
}

##### Calculate evaluation period MAE


evaluation_mae <- postResample(stock_predictions$predicted_returns, stock_predictions$retx)

tibble("Model" = "insert model",
       "Evaluation period mae" = evaluation_mae[[3]]) %>% 
  kable(caption = "Performance metric in evaluation period", 
        digits  = 4) %>% 
  kable_classic(full_width = F, 
                html_font  = "Times New Roman")  %>% 
  save_kable("images/evaluation_period_mae.png", 
             zoom = 3, 
             density = 1900)








