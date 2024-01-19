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

# X-Learner
library(randomForest) # random forest
library(glmnet)       # lasso
library(grf)
library(hdm)

first_best_policy <- function(dir, outcome_vec, set_vec, cost_vec, idx){
  
  # Final table
  results_table <- data.frame(
    Outcome   = character(),
    Covariates = character(),
    Algorithm = character(),
    Cost      = numeric(),
    Welfare   = numeric(),
    Std       = numeric(),
    Treat     = numeric(),
    Non_treat = numeric(),
    stringsAsFactors = FALSE
  )
  
  # binary treatmet
  treatment <- "treatment"
  
  # all the combination of all outcome  cost vector
  combination <- expand.grid(outcome_vec, set_vec)

  index_dir <- paste0("/result_", idx)
  
  # convert those into single vector
  combined_vec <- as.vector(apply(combination, 1, paste, collapse = "-"))
  
  data <- read.csv(paste0(dir, index_dir, "/simulation_data_", idx,".csv"))
  
  # average treatment effect
  ## Extract the average treatment effect
  print("Average Treatment Effect")
  fmla <- formula(paste(outcome, ' ~ ', treatment))
  print(fmla)
  ols <-  lm(fmla, data= data)
  print(coeftest(ols, vcov=vcovHC(ols, type='HC2')))
  ATE <- ols$coefficients[treatment]
  
  # # real specification
  # print("Real Specification")
  # fmla <- as.formula("outcome ~ treatment + X + theta + treatment:X + treatment:theta  + treatment:X:theta")
  # print(fmla)
  # ols <-  lm(fmla, data= data)
  # robust_se <- vcovHC(ols, type = 'HC2')
  # coeftest_result <- coeftest(ols, vcov = robust_se)
  # print(coeftest_result)
  
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
      covariates <- c("predicted_transaction")
    } else if(setid == 5){
      # setting 5
      covariates <- c("X", "state2p")
    } else if(setid == 6){
      # setting 5
      covariates <- c("X", "value_period10")
    } else if(setid == 7){
      # setting 5
      covariates <- c("X", "predicted_transaction")
    }
    
    # Generate CATE with causal forest
    print("Start for Causal Forest algorithm...")
    print(value)
    
    if(setid < 5){
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
      forest.tau <- causal_forest(XX, Y, W, W.hat=.5) 
    )
  
    # Get predictions from forest fitted above.
    tau.hat <- predict(forest.tau)$predictions  # tau(X) CATE estimates
      
    for (cost_pr in cost_vec){
      
      # calculate the cost of the treatment
      # cost <- cost_pr * ATE
      cost <- cost_pr * 2
      # assign treatment for those whose predicted CATE larger than 0
      pi.hat <- ifelse(tau.hat > cost, 1, 0)
      
      # Estimating the value of the learned policy. 
      A <- pi.hat == 1
      Y.test <- data[, outcome]
      W.test <- data[, treatment]
      
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
        Covariates   = ifelse(length(covariates) > 1, paste0(covariates, collapse = "-"), covariates),
        Algorithm   = "First Best - Causal Forest",
        Cost        = cost_pr,
        Welfare     = value.avg.estimate,
        Std         = value.avg.stderr,
        Treat       = treat / length(W.test),
        Non_treat   = no_treat / length(W.test),
        stringsAsFactors = FALSE
      ))
    }
      
    # X-Learner: # of covariate should be larger than 1
    if(setid >= 5){
      print("Start for X-Learner...")
      cov <- ifelse(length(covariates) > 1, paste0(covariates, collapse = "-"), covariates)
      print(paste0(outcome, " ", cov))
      
      # X-learner with Lasso
      Y <- data[, outcome]
      W <- data[, treatment]
      X <- as.matrix(data[, covariates])
      
      # Step 1. we construct prediction functions for mu1.hat and mu0.hat, respectively
      TL.mu0 <- cv.glmnet(X[W==0,], Y[W==0], nfolds = 10, alpha =1,  standardize = TRUE)
      TL.mu0.coef <- as.vector(t(coef(TL.mu0, s = "lambda.min")))
      
      TL.mu1 <- cv.glmnet(X[W==1,], Y[W==1], nfolds = 10, alpha =1,  standardize = TRUE)
      TL.mu1.coef <- as.vector(t(coef(TL.mu1, s = "lambda.min")))
      
      # Step 2. For treated and control groups, we respectively calculated the noisy proxies for individual treatment effect
      yhat1 <- cbind(1,X[W==0, ]) %*% TL.mu1.coef
      yhat0 <- cbind(1,X[W==1, ]) %*% TL.mu0.coef
      
      tau.tilde.1 <- Y[W==1] - yhat0
      tau.tilde.0 <- yhat1 - Y[W==0]
      
      #Step 3. For treated and control groups, we respectively construct the prediction function (tau.hat) by predicting tau.tilde from X.
      XL.tau.hat.1 <- cv.glmnet(X[W==1, ], tau.tilde.1, nfolds = 10, alpha = 1, standardize = TRUE)
      XL.tau.hat.1.coef <- as.vector(t(coef(XL.tau.hat.1, s = "lambda.min")))
      XL.tau.hat.1.preds <- cbind(1,X) %*% XL.tau.hat.1.coef
      
      XL.tau.hat.0 <- cv.glmnet(X[W==0, ], tau.tilde.0, nfolds = 10, alpha = 1, standardize = TRUE)
      XL.tau.hat.0.coef <- as.vector(t(coef(XL.tau.hat.0, s = "lambda.min")))
      XL.tau.hat.0.preds <- cbind(1,X) %*% XL.tau.hat.0.coef
      
      #Step 4. we estimated e.hat and report the CATE estimator.
      e_fit <- cv.glmnet(X, W, nfolds = 10, keep = TRUE, alpha = 1, standardize = TRUE)
      e_hat <- e_fit$fit.preval[,!is.na(colSums(e_fit$fit.preval))][, e_fit$lambda[!is.na(colSums(e_fit$fit.preval))] ==  e_fit$lambda.min]
      
      # final predicted CATE
      XL.cate <- ( 1 - e_hat ) * XL.tau.hat.1.preds + e_hat * XL.tau.hat.0.preds
      
      for (cost_pr in cost_vec){
        print(paste0("Start for cost: ", cost_pr))
        
        # calculate the cost of the treatment
        # cost <- cost_pr * ATE
        cost <- cost_pr * 1
        
        # assign treatment for those whose predicted CATE larger than 0
        pi.hat <- ifelse(XL.cate > cost, 1, 0)
        
        # Estimating the value of the learned policy. 
        # Note in the code below that we must subtract the cost of treatment.
        A <- pi.hat == 1
        Y.test <- data[, outcome]
        W.test <- data[, treatment]
        
        # the number of agent being treated
        x_treat <- sum(pi.hat)
        x_no_treat <- length(W.test) - x_treat
        
        # policy evaluation
        # Note the -cost here!
        # estimate value
        est_A = ifelse(sum(A & (W.test==1)) > 0, mean(Y.test[A & (W.test==1)]) - cost, 0) * mean(A)
        est_notA = ifelse(sum(!A & W.test==0) > 0, mean(Y.test[!A & (W.test==0)]), 0) * mean(!A)
        x_value.avg.estimate = est_A + est_notA
        
        stderr_A = ifelse(sum(A & (W.test==1)) > 0, var(Y.test[A & (W.test==1)]) / sum(A & (W.test==1)) * mean(A)^2, 0)
        stderr_notA = ifelse(sum(!A & W.test==0) > 0, var(Y.test[!A & (W.test==0)]) / sum(!A & (W.test==0)) * mean(!A)^2, 0)
        x_value.avg.stderr = sqrt(stderr_A + stderr_notA)
        
        results_table <- rbind(results_table, data.frame(
          Outcome     = outcome,
          Covariates   = ifelse(length(covariates) > 1, paste0(covariates, collapse = "-"), covariates),
          Algorithm   = "First Best - X-Learner",
          Cost        = cost_pr,
          Welfare     = x_value.avg.estimate,
          Std         = x_value.avg.stderr,
          Treat       = x_treat / length(W.test),
          Non_treat   = x_no_treat / length(W.test),
          stringsAsFactors = FALSE
        ))
      }
    }
  }
  print(results_table)
  title <- paste0(dir, index_dir, "/first_best_policy_", idx, ".csv")
  write.csv(results_table, title , row.names = TRUE)
  
  print("Done!!!")
}