
################################### HYPEPARAMETER TABLES #########################



## Neural network tested hyperparameters
tibble("Neural network models" = list("One hidden layer", "Two hidden layers", "Three hidden layers", "Four hiden layers"),
       "Learning rates" = list(list(0.005, 0.1, 0.4), c(0.005, 0.1, 0.4),list(0.005, 0.1, 0.4), list(0.005, 0.1, 0.4)),
       "Batch normalization" = list(list("No"), list("Yes", "No"), list("Yes", "No"), list("Yes", "No")),
       "Dropout rates" = list(list(0), list(0, 0.2, 0.4), list(0, 0.2, 0.4), list(0, 0.2, 0.4)),
       "Batch sizes"   = list(list(300, 500, 1000), list(500, 1000, 2000), list(500, 1000, 2000), list(500, 1000, 2000)),
       "Maximum epochs" = list(list(200), list(200), list(200), list(200)),
       "Patience"       = list(list(1, 2,10, 5, 15), list(1, 5, 20), list(1, 5, 20), list(1, 5, 20))
       ) %>% 
  kable(caption = "Tested Hyperparameters of Neural Network models", 
        digits  = 4) %>% 
  kable_classic(full_width = F, 
                html_font  = "Times New Roman")  %>% 
  save_kable("images/neural_network_tested_parameters.png", 
             zoom = 3, 
             density = 1900)





## Best performing neural network architecture


tibble("Number of layers" = list(2),
       "Learning rates" = 0.005,
       "Batch normalization" = "No",
       "Dropout rates" = 0,
       "Batch sizes"   = 500,
       "Patience" =  20) %>% 
  t() %>% 
  kable(caption = "Best Performing Neural Network Hyperparameters", 
        digits  = 4) %>% 
  kable_classic(full_width = F, 
                html_font  = "Times New Roman") %>% 
  save_kable("images/neural_network_optimal_parameters.png", 
             zoom = 3, 
             density = 1900)

