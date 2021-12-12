
################################### HYPEPARAMETER TABLES #########################



## Neural network tested hyperparameters
tibble("Neural network models" = list("One hidden layer", "Two hidden layers", "Three hidden layers", "Four hiden layers"),
       "Learning rates" = list(c(0.005, 0.1, 0.4), c(0.005, 0.1, 0.4),list(0.005, 0.1, 0.4), list(0.005, 0.1, 0.4))
       ) %>% 
  kable(caption = "Tested hyperparameters in Neural Network models", 
        digits  = 4) %>% 
  kable_classic(full_width = F, 
                html_font  = "Times New Roman")  %>% 
  save_kable("images/neural_network_tested_parameters.png", 
             zoom = 3, 
             density = 1900)
