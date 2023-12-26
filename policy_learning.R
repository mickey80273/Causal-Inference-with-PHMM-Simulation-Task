# data science packages
library(tidyverse)
library(dplyr)
library(magrittr) 

library(grf)          # causal forest
library(rpart)        # tree
library(glmnet)       # lasso
library(splines)      # splines
library(personalized) # propensity score plot
library(MASS)
library(lmtest)
library(sandwich)
library(corrplot)
library(ggplot2)

# policy learning
library(policytree)

policy_learning <- function(dir, outcome_vec, set_vec, cost_vec, idx, result_dir){
  
  # Final table
  results_table <- data.frame(
    Outcome   = character(),
    Algorithm = character(),
    Covariates = character(),
    Cost      = numeric(),
    Welfare   = numeric(),
    Std       = numeric(),
    Treat     = numeric(),
    Non_treat = numeric(),
    stringsAsFactors = FALSE
  )
  
  # binary treatmet
  treatment <- "treatment"

  index_dir <- paste0("/result_", idx)
  
  # all the combination of all outcome  cost vector
  combination <- expand.grid(outcome_vec, set_vec)
  
  # convert those into single vector
  combined_vec <- as.vector(apply(combination, 1, paste, collapse = "-"))
  
  data <- read.csv(paste0(dir, index_dir, "/simulation_data_", idx,".csv"))
  n <- nrow(data)
  
  # write here into loop
  for (value in combined_vec){
    # split the string: outcome and cost
    value_split <-strsplit(value, "-")
    outcome <- value_split[[1]][1]
    setid <- as.numeric(value_split[[1]][2])
    
    if(setid == 1){
      # setting 1
      covariates <- c("X")
    } else if(setid == 2){
      # setting 2
      covariates <- c("state2p")
    } else if(setid == 3){
      # setting 3
      covariates <- c("value_period10")
    } else if(setid == 4){
      # setting 4
      covariates <- c("X", "state2p")
    } else if(setid == 5){
      # setting 5
      covariates <- c("X", "value_period10")
    }
    
    # Extract Average Treatment effect: y = a + bT +u
    fmla <- formula(paste(outcome, ' ~ ', treatment))
    ols <-  lm(fmla, data= data)
    ATE <- ols$coefficients[treatment]
    # print(ATE)
    
    # Causal Forest
    print("Start for Causal Forest...")
    print(value)
    if(setid <= 3){
      fmla <- formula(paste0("~ 0 +" , covariates))
    } else{
      fmla <- formula(paste0("~ 0 +", paste0(covariates, collapse="+")))
    }
    
    XX <- model.matrix(fmla, data)
    W <- data[,treatment]
    Y <- data[,outcome]
    
    # Comment or uncomment as appropriate.
    # Randomized setting with known and fixed probabilities (here: 0.5).
    system.time(
      forest.tau <- causal_forest(XX, Y, W, W.hat = .5) 
    )
    
    # Get predictions from forest fitted above.
    tau.hat <- predict(forest.tau)$predictions  # tau(X) CATE estimates
    
    # Policy learning
    X <- data[,covariates]
    Y <- data[,outcome]
    W <- data[,treatment]
    
    # Randomized setting: pass the known treatment assignment as an argument.
    # forest <- causal_forest(X, Y, W, W.hat=.5)
    forest <- forest.tau
    
    # Fit a policy tree on forest-based AIPW scores
    gamma.matrix <- double_robust_scores(forest)  
    
    for (cost_pr in cost_vec){
      print(paste0("Start for cost: ", cost_pr))
      # calculate the cost of the treatment
      # cost <- cost_pr * ATE
      cost <- cost_pr * 2
      
      # doubly robust score
      gamma.matrix[,2] <- gamma.matrix[,2] - cost  # Subtracting cost of treatment
      
      train <- 1:(.8*n)
      test <- (.8*n):n
      
      # Fit policy on training subset
      # Predicting treatment on test set
      if(setid <= 3){
        system.time(
          policy <- policy_tree(matrix(X[train]), gamma.matrix[train,], depth = 2, min.node.size=1, split.step=5)
        )
        pi.hat <- predict(policy, matrix(X[test])) - 1
      } else{
        system.time(
          policy <- policy_tree(X[train,], gamma.matrix[train,], depth = 2, min.node.size=1, split.step=5)
        )
        pi.hat <- predict(policy, X[test,]) - 1
      }
      
      # Estimating the value of the learned policy. 
      A <- pi.hat == 1
      Y.test <- data[test, outcome]
      W.test <- data[test, treatment]
      
      # policy evaluation
      # Note the -cost here!
      # Note: this is to handle !A & W.test==0 is NA
      # estimate value
      est_A = ifelse(sum(A & (W.test==1)) > 0, mean(Y.test[A & (W.test==1)]) - cost, 0) * mean(A)
      est_notA = ifelse(sum(!A & W.test==0) > 0, mean(Y.test[!A & (W.test==0)]), 0) * mean(!A)
      value.avg.estimate = est_A + est_notA
      # estimate standard error
      stderr_A = ifelse(sum(A & (W.test==1)) > 0, var(Y.test[A & (W.test==1)]) / sum(A & (W.test==1)) * mean(A)^2, 0)
      stderr_notA = ifelse(sum(!A & W.test==0) > 0, var(Y.test[!A & (W.test==0)]) / sum(!A & W.test==0) * mean(!A)^2, 0)
      value.avg.stderr = sqrt(stderr_A + stderr_notA)
      
      # the number of agent being treated
      treat <- sum(pi.hat)
      no_treat <- length(W.test) - treat
      
      results_table <- rbind(results_table, data.frame(
        Outcome     = outcome,
        Algorithm   = "Causal Forest - Policy Learning",
        Covariates   = ifelse(length(covariates) > 1, paste0(covariates, collapse = "-"), covariates),
        Cost        = cost_pr,
        Welfare     = value.avg.estimate,
        Std         = value.avg.stderr,
        Treat       = treat / length(W.test),
        Non_treat   = no_treat / length(W.test),
        stringsAsFactors = FALSE
      ))
    }
    
  }
  
  print(results_table)
  title <- paste0(dir, index_dir, "/policy_learning_result_", idx, ".csv")
  write.csv(results_table, title , row.names = TRUE)
  
  print("Done!!!")
}

