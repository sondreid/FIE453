
################################### HYPEPARAMETER TABLES #########################
library(tidyr)
library(kableExtra)

## Neural network tested hyperparameters
tibble("Neural network models" = list("One hidden layer", "Two hidden layers", "Three hidden layers", "Four hiden layers"),
       "Learning rates" = list(list(0.005, 0.1, 0.4), c(0.005, 0.1, 0.4),list(0.005, 0.1, 0.4), list(0.005, 0.1, 0.4)),
       "Batch normalization" = list(list("No"), list("Yes", "No"), list("Yes", "No"), list("Yes", "No")),
       "Dropout rates" = list(list(0), list(0, 0.2, 0.4), list(0, 0.2, 0.4), list(0, 0.2, 0.4)),
       "Batch sizes"   = list(list(300, 500, 1000), list(500, 1000, 2000), list(500, 1000, 2000), list(500, 1000, 2000)),
       "Maximum epochs" 
       ) %>% 
  kable(caption = "Tested hyperparameters in Neural Network models", 
        digits  = 4) %>% 
  kable_classic(full_width = F, 
                html_font  = "Times New Roman")  %>% 
  save_kable("images/neural_network_tested_parameters.png", 
             zoom = 3, 
             density = 1900)



tibble("vwretd" = "Market Excess Return", "vol" = "Volume", "shrout" = "Shared Outstanding", "cshfdq" = "Common Shared for Diluted EPS", 
       "aoq" = "Assets - Other - Total", "cheq" = "Cash and Short-Term Investments",
       "cogsq" = "Cost of Goods Sold", "apq" = "Account Payable/Creditors - Trade", "epsfxq" = "Earnings Per Share (Diluted) - Excluding Extraodinary Items", 
       "chq" = "Cash", "citotalq" = "Comprehensive Income - Parent", "ceqq" = "Common/Ordinary Equity - Total",
       "dpq" = "Depreciation and Amortization - Total", "lcoq" = "Current Liabilities - Other - Total",
       "capsq" = "Capital Surplus/Share Premium Reserve", "acoq" = "Current Asets - Other - Total",
       "invtq" = "Inventories - Total", "cstkq" = "Common/Ordinary Stock (Capital)",
       "intanoq" = "Other Intangibles", "dlttq" = "Long-Term Debt - Total") %>% 
  t() %>% 
  kable(caption = "Feature Description", 
        digits  = 4) %>% 
  kable_classic(full_width = F, 
                html_font  = "Times New Roman")  %>% 
  save_kable("C:/Users/joonl/Downloads/feature_description1.png", 
             zoom = 3, 
             density = 1900)
