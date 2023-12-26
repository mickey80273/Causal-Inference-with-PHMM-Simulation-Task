# data science packages
library(tidyverse)
library(dplyr)
library(magrittr) 

uniform_policy <- function(dir, outcome_vec, cost_vec, idx){
  
  # Final table
  results_table <- data.frame(
    Outcome   = character(),
    Setting   = character(),
    Cost      = numeric(),
    Welfare   = numeric(),
    Std       = numeric(),
    Treat     = numeric(),
    Non_treat = numeric(),
    stringsAsFactors = FALSE
  )
  
  treatment <- "treatment"

  index_dir <- paste0("/result_", idx)
  
  # all the combination of all outcome  cost vector
  combination <- expand.grid(outcome_vec, cost_vec)
  
  # convert those into single vector
  combined_vec <- as.vector(apply(combination, 1, paste, collapse = "-"))
  
  data <- read.csv(paste0(dir, index_dir, "/simulation_data_", idx,".csv"))
  
  for (value in combined_vec) {
    # split the string: outcome and cost
    value_split <-strsplit(value, "-")
    outcome <- value_split[[1]][1]
    cost_pr <- as.numeric(value_split[[1]][2])
    n <- nrow(data)
    
    # Extract Average Treatment effect: y = a + bT +u
    fmla <- formula(paste(outcome, ' ~ ', treatment))
    ols <-  lm(fmla, data= data)
    ATE <- ols$coefficients[treatment]
    # cost <- cost_pr * ATE
    cost <- cost_pr * 2
    
    # for calculate the welfare
    Y <- data[, outcome]
    W <- data[, treatment]
    
    # All Treatment Policy
    
    # Only valid for randomized setting.
    # Note the -cost here!
    AT.value.avg.estimate <- mean(Y[W==1]) - cost
    n_treatment <- sum(W == 1)
    AT.value.avg.stderr <- sqrt(sum((Y - AT.value.avg.estimate)^2 * W) / (n_treatment - 1)) / sqrt(n_treatment)
    
    results_table <- rbind(results_table, data.frame(
      Outcome = outcome,
      Setting   = "All Treatment",
      Cost      = cost_pr,
      Welfare   = AT.value.avg.estimate,
      Std       = AT.value.avg.stderr,
      Treat     = n_treatment,
      Non_treat = 0,
      stringsAsFactors = FALSE
    ))
    
    if (cost_pr == as.character(cost_vec[1])){
      # No Treatment Policy
      # Only valid for randomized setting.
      no.value.avg.estimate <- mean(Y[W==0])
      n_no_treatment <- sum(W == 0)
      no.value.avg.stderr <- sqrt(sum((Y - no.value.avg.estimate)^2 * (1 - W)) / (n_no_treatment - 1)) / sqrt(n_no_treatment)
      
      results_table <- rbind(results_table, data.frame(
        Outcome     = outcome,
        Setting     = "No Treatment",
        Cost        = 0,
        Welfare     = no.value.avg.estimate,
        Std         = no.value.avg.stderr,
        Treat       = n_no_treatment,
        Non_treat   = 0,
        stringsAsFactors = FALSE
      ))
    }
  }
  
  print(results_table)
  title <- paste0(dir, index_dir, "/uniform_policy_result_", idx,".csv")
  write.csv(results_table, title , row.names = TRUE)
  
}