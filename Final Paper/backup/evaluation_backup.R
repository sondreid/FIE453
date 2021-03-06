
################################################################################
########################### MODEL EVALUATION ####################################
################################################################################
# Based on model test performance metrics




############### 
# Kan kanskje slettes
##########
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
  i <- 0
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








selected_stock_company_info <- function(selected_stocks, test_df) {
  #' @description:    Makes a summary of feature means of selcted stocks (companies)
  #' 
  #' @sele
  #' @test_df: orginal test set from which the selcted companies are drawn
  #' @n: number of included companies
  selected_stocks <- selected_stocks 
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

## Summaries for all companies in test sets



all_company_metrics <- function(selected_test_df) {
  #' Returns mean financial indicators of all stocks in input test dataset
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
    kable(caption = "Mean financial indicators of all test companies", 
          digits  = 2)  %>% 
    kable_classic(full_width = F, 
                  html_font = "Times New Roman") %>% 
    save_kable("images/all_company_summary.png", 
               zoom = 3, 
               density = 1900)
  
}



#always_0_stocks <- select_stocks_always_0(test_df_scaled)




mean_metric_stock_level <- function(num_top,  selected_test_df, nn_model_metrics, caret_model_metrics) {
  
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


selected_stocks_knn            <- stock_level_predictions_caret(test_df_scaled, knn_model, verbose = T)
selected_stocks_bayesian_ridge <- stock_level_predictions_caret(test_df_scaled, bayesian_ridge_model)
selected_stokcks_gbm           <- stock_level_predictions_caret(test_df_scaled, gbm_model)

selected_stocks_nn_1_layer  <- stock_level_predictions_nn(test_df, best_model_nn_1_layer_all[[1]]) 
selected_stocks_nn_2_layers <- stock_level_predictions_nn(test_df, best_model_nn_2_layers_all[[1]]) 
selected_stocks_nn_3_layers <- stock_level_predictions_nn(test_df, best_model_nn_3_layers_all[[1]]) 
selected_stocks_nn_4_layers <- stock_level_predictions_nn(test_df, best_model_nn_4_layers_all[[1]]) 

nn_stock_level_mae <- list(
  list(selected_stocks_nn_1_layer, "Neural Network 1 hidden layer"), 
  list(selected_stocks_nn_2_layers, "Neural Network 2 hidden layers"),
  list(selected_stocks_nn_3_layers, "Neural Network 3 hidden layers"),
  list(selected_stocks_nn_4_layers, "Neural Network 4 hidden layers")
)


caret_stock_level_mae <- list(
  list(selected_stokcks_gbm, "Gradient Boosting machine"),
  list(selected_stocks_knn, "K-nearest neighbors"),
  list(selected_stocks_bayesian_ridge, "Bayesian ridge regression")
)


#model_metrics_stock_level <- mean_metric_stock_level(30, test_df,  nn_stock_level_mae, caret_stock_level_mae) %>% arrange(`MAE top stocks`)

#save(nn_stock_level_mae, caret_stock_level_mae, model_metrics_stock_level,  file = "model_results/stock_level_performance.Rdata")

load(file = "model_results/stock_level_performance.Rdata")


## 2- layer neural network model predictable stocks
two_layer_nn_predictable_stocks <- nn_stock_level_mae[[2]][[1]] %>% 
  arrange(`Test MAE`) %>% 
  head(30) 


### Model performance on 30 most predictable stocks

model_metrics_stock_level %>% 
  kable(caption = "Model performance on 30 most predictable stocks", 
        digits  = 4)  %>% 
  kable_classic(full_width = F, 
                html_font = "Times New Roman") %>% 
  save_kable("images/model_performance_stock_level.png", 
             zoom = 3, 
             density = 1900)



#### Training process graph of optimal neural network model #####



best_model_nn_2_layers_all[[2]] %>% 
  plot() +
  theme_bw() +
  xlim(0, 100) +
  theme(legend.position = "bottom") +
  labs(x = "Epoch", title ="Training process of two hidden layer neural network") +
  scale_colour_manual(values=c("orange", "#fa3200", "#fa9664"))
ggsave(filename = "images/neural_net_training.png", scale = 1, dpi = 1000)







### Based on best performing model, which stocks are predictable ###

two_layer_nn_predictable_stocks %>% 
  mutate("Company name" = sapply(`Company name`, stringr::str_to_title)) %>% 
  kable(caption = "30 stocks of highest predictability", 
        digits  = 5)  %>% 
  kable_classic(full_width = F, 
                html_font = "Times New Roman") %>% 
  save_kable("images/predictable_stocks.png", 
             zoom = 3, 
             density = 1900)


#### Predictable companies financial indicators  ###


selected_stock_company_info(two_layer_nn_predictable_stocks, test_df ) %>% 
  kable(caption = "Financial indicators of 30 most predictable stocks", 
        digits  = 5)  %>% 
  kable_classic(full_width = F, 
                html_font = "Times New Roman") %>% 
  save_kable("images/predictable_stocks_characteristics.png", 
             zoom = 3, 
             density = 1900)


#### Predictable companies financial indicators, average indicator values ###

selected_stock_company_info(two_layer_nn_predictable_stocks, test_df ) %>% 
  summarise("Mean market cap" = mean(`Mean market cap`), 
            "Mean volume" = mean(`Mean volume`), 
            "Mean cash" = mean(`Mean cash`), 
            "Mean operating income" = mean(`Mean operating income`), ) %>% 
  kable(caption = "Mean financial indicators of 30 most predictable stocks", 
        digits  = 5)  %>% 
  kable_classic(full_width = F, 
                html_font = "Times New Roman") %>% 
  save_kable("images/predictable_stocks_characteristics_mean.png", 
             zoom = 3, 
             density = 1900)

## Average indicator values all test stocks ##### 


all_company_metrics(test_df)




################################################### Evaluation period ##################################


##### Monthly predictions ######################



predict_monthly_returns_nn <- function(selected_model, stocks, evaluation_data) {
  
  data_selected_stocks <- evaluation_data %>% 
    filter(permno %in% stocks)
  
  stock_predictions <- selected_model %>% predict(data_selected_stocks %>% dplyr::select(-retx, -date, -permno))
  
  data_selected_stocks %<>%
    mutate(predicted_returns = stock_predictions[,1])
  
  return(data_selected_stocks)
  
  
}


stock_predictions <- predict_monthly_returns_nn(best_model_nn_2_layers_all[[1]], 
                                                two_layer_nn_predictable_stocks$`Company identifier`  , # Company identifiers of the most predictable companies
                                                evaluation_data %>% filter(lubridate::year(date) < "2020"))

##### Calculate evaluation period MAE


postResample(stock_predictions$predicted_returns, stock_predictions$retx)

postResample(rep(0, nrow(stock_predictions)), stock_predictions$retx)



plot_monthly_returns_single_company <- function(stock_predictions, selected_stock) {
  #'
  #'Plots series of observed and predicted returns.
  predictions_and_company_info <- stock_predictions %>% 
    dplyr::filter(permno == selected_stock$`Company identifier` ) %>% 
    as_tibble() %>% 
    left_join(selected_stocks, by = c("permno" = "Company identifier")) %>% 
    mutate("Company name" = sapply(`Company name`, stringr::str_to_title)) %>% 
    pivot_longer(cols = c(predicted_returns, retx),
                 names_to = "type",
                 values_to = "returns")
  
  company_name <- predictions_and_company_info$`Company name`[1]
  predictions_and_company_info %>% ggplot() +
    geom_line(aes(x = date, y = returns, col = type), lwd = 0.99) +
    guides(colour = guide_legend("Series name")) +
    theme_bw() +
    theme(legend.position = "bottom") +
    labs(x = "Date", y = "Returns", title = paste("Predicted vs observed return in 2018-2019 of ", company_name )) +
    scale_colour_manual(values=c("orange", "black"))
  
}
plot_monthly_returns_single_company(stock_predictions, two_layer_nn_predictable_stocks %>% head(1))
ggsave(filename = "images/evaluation_plot_most_predictable_stock.png", scale = 1, dpi = 1200)



plot_monthly_returns_companies <- function(stock_predictions, selected_stocks, selected_title, selected_ncol = 2) {
  #'
  #'Plots series of observed and predicted returns.
  stock_predictions %>% 
    dplyr::filter(permno %in% selected_stocks$`Company identifier` ) %>% 
    left_join(selected_stocks, by = c("permno" = "Company identifier")) %>% 
    mutate("Company name" = sapply(`Company name`, stringr::str_to_title)) %>% 
    as_tibble() %>% 
    pivot_longer(cols = c(predicted_returns, retx),
                 names_to = "type",
                 values_to = "returns") %>% 
    ggplot() +
    geom_line(aes(x = date, y = returns, col = type), lwd = 0.99) +
    facet_wrap(~`Company name`, ncol = selected_ncol) +
    guides(colour = guide_legend("Series name")) +
    theme_bw() +
    theme(legend.position = "bottom") +
    labs(x = "Date", y = "Returns", title = selected_title) +
    scale_colour_manual(values=c("orange", "black"))
  
  
  
}

# Make a facet plot of the five most predictable stocks in the evaluation period


plot_monthly_returns_companies(stock_predictions,  two_layer_nn_predictable_stocks %>% head(15),
                              "Predicted vs Observed Returns of Most Predictable Stocks in 2018-2019", 3)
ggsave(filename = "images/evalution_plot_15_first_predictable_stocks.png", scale = 1.5, dpi = 700, width = 5, height = 8)



plot_monthly_returns_companies(stock_predictions,  two_layer_nn_predictable_stocks %>% tail(15),
                               "Predicted vs Observed Returns of Most Predictable Stocks in 2018-2019", 3)
ggsave(filename = "images/evalution_plot_15_last_predictable_stocks.png", scale = 1.5, dpi = 700, width = 5, height = 8)


plot_monthly_returns_companies(stock_predictions,  two_layer_nn_predictable_stocks %>% head(5), 
                                "Predicted vs Observed Returns of Five Most Predictable Stocks in 2018-2019" )
ggsave(filename = "images/evalution_plot_5_most_predictable_stocks.png", scale = 2, dpi = 500, width = 5, height = 5)


tibble("Model" = "insert model",
       "Evaluation period mae" = evaluation_mae[[3]]) %>% 
  kable(caption = "Performance metric in evaluation period", 
        digits  = 4) %>% 
  kable_classic(full_width = F, 
                html_font  = "Times New Roman")  %>% 
  save_kable("images/evaluation_period_mae.png", 
             zoom = 3, 
             density = 1900)








