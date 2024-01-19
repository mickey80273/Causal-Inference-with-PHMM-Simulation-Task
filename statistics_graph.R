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
library(reshape2)
library(policytree)
# install.packages("DiagrammeRsvg")
library(DiagrammeRsvg)
# found from https://cran.r-project.org/web/packages/policytree/policytree.pdf

statistics_graph <- function(dir, outcome_vec, set_vec, cost_vec, idx){
  
  # set the path to the file
  index_dir <- paste0("/result_", idx)
  # all the combination of all outcome  cost vector
  combination <- expand.grid(outcome_vec, set_vec)
  combined_vec <- as.vector(apply(combination, 1, paste, collapse = "-"))
  data <- read.csv(paste0(dir, index_dir, "/simulation_data_", idx,".csv"))
  
  # treatment and outcome (for empirical experiment)
  outcome_vec <- "outcome"
  treatment <- "treatment"
  
  for (value in combined_vec){
  
    # store the pdf file
    pdf_path <- paste0(dir, '/result_', idx, '/graph_', value, '.pdf')
    pdf(pdf_path, onefile = TRUE)
    
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
    
    if(setid < 5){
      fmla <- formula(paste0("~ 0 +" , covariates))
    } else{
      fmla <- formula(paste0("~ 0 +", paste0(covariates, collapse="+")))
    }
    
    # correlation graph
    data_cor <- data[, c(outcome_vec, covariates)]
    cor_matrix <- cor(data_cor)
    cor_melted <- melt(cor_matrix)
    gg_cor_plot <- ggplot(cor_melted, aes(Var1, Var2, fill = value)) +
                      geom_tile() +
                      scale_fill_gradient2(limits = c(-1, 1), mid = "white", high = "blue", low = "red") +
                      theme_minimal() +
                      coord_fixed()
    # ggsave("correlation_plot_ggplot.png", plot = gg_cor_plot, width = 10, height = 8)
    print(gg_cor_plot)
    
    X <- model.matrix(fmla, data)
    W <- data[,treatment]
    Y <- data[,outcome]
    
    # Comment or uncomment as appropriate.
    # Randomized setting with known and fixed probabilities (here: 0.5).
    system.time(
      forest.tau <- causal_forest(X, Y, W, W.hat=.5) 
    )
    
    # Get predictions from forest fitted above.
    tau.hat <- predict(forest.tau)$predictions  # tau(X) CATE estimates
    
    gg_hist <- ggplot(data.frame(tau.hat), aes(x = tau.hat)) + 
                  geom_histogram(binwidth = 1, aes(y = after_stat(density)), fill = "blue", color = "black") +
                  ggtitle("CATE Estimates") +
                  theme_minimal() +
                  xlab("Estimated Treatment effect") +
                  ylab("Density")
    print(gg_hist)
    
    # 檢查變數重要性 -> write a document to record the number
    # var_imp <- c(variable_importance(forest.tau))
    # names(var_imp) <- covariates
    # sorted_var_imp <- sort(var_imp, decreasing = TRUE)
    # sorted_var_imp
    
    # treatment heterogeneity
    n <- nrow(data)
    num.rankings <- 5  
    num.folds <- 10
    folds <- sort(seq(n) %% num.folds) + 1 
    system.time(
      forest <- causal_forest(X, Y, W, W.hat=.5, clusters = folds) #樣本分成10組
    )
    
    tau.hat <- predict(forest)$predictions
    
    # Rank observations *within each fold* into quintiles according to their CATE predictions.
    ranking <- rep(NA, n)
    for (fold in seq(num.folds)) {
      tau.hat.quantiles <- quantile(tau.hat[folds == fold], probs = seq(0, 1, by=1/num.rankings))
      # print(tau.hat.quantiles)
      tau.hat.quantiles <- tau.hat.quantiles + runif(length(tau.hat.quantiles))/100000
      # print(tau.hat.quantiles)
      ranking[folds == fold] <- cut(tau.hat[folds == fold], tau.hat.quantiles, include.lowest=TRUE,labels=seq(num.rankings))
    }
    
    # ols
    
    # Formula y ~ 0 + ranking + ranking:w
    fmla <- paste0(outcome, " ~ 0 + ranking + ranking:", treatment)
    ols.ate <- lm(fmla, data=transform(data, ranking=factor(ranking)))
    ols.ate <- coeftest(ols.ate, vcov=vcovHC(ols.ate, type='HC2'))
    interact <- which(grepl(":", rownames(ols.ate)))
    # deal with the NA interaction (currently, only happen in simulation data)
    modified <- ols.ate[interact, 1:2]
    if(nrow(ols.ate[interact, 1:2]) != 5){
      modified <- data.frame(modified)
      new_row <- data.frame(
        Estimate = 0,
        "Std. Error" = 0,
        row.names = "ranking1:treatment"
      )
      modified <- as.matrix(rbind(new_row, modified))
    }
    ols.ate <- data.frame("ols", paste0("Q", seq(num.rankings)), modified)
    rownames(ols.ate) <- NULL # just for display
    colnames(ols.ate) <- c("method", "ranking", "estimate", "std.err")
    # ols.ate
    
    # AIPW
    e.hat <- forest$W.hat # P[W=1|X]
    m.hat <- forest$Y.hat # E[Y|X]
    mu.hat.0 <- m.hat - e.hat * tau.hat        # E[Y|X,W=0] = E[Y|X] - e(X)*tau(X)
    mu.hat.1 <- m.hat + (1 - e.hat) * tau.hat  # E[Y|X,W=1] = E[Y|X] + (1 - e(X))*tau(X)
    
    # AIPW scores
    aipw.scores <- tau.hat + W / e.hat * (Y -  mu.hat.1) - (1 - W) / (1 - e.hat) * (Y -  mu.hat.0)
    ols <- lm(aipw.scores ~ 0 + factor(ranking))
    # deal with the NA interaction (currently, only happen in simulation data)
    modified_AIPW <-coeftest(ols, vcov=vcovHC(ols, "HC2"))[,1:2]
    if(nrow(coeftest(ols, vcov=vcovHC(ols, "HC2"))[,1:2]) != 5){
      modified_AIPW <- data.frame(modified_AIPW)
      new_row_AIPW <- data.frame(
        Estimate = 0,
        "Std. Error" = 0,
        row.names = 'factor(ranking)1'
      )
      modified_AIPW <- as.matrix(rbind(new_row_AIPW, modified_AIPW))
    }
    forest.ate <- data.frame("aipw", paste0("Q", seq(num.rankings)), modified_AIPW) # changed
    colnames(forest.ate) <- c("method", "ranking", "estimate", "std.err")
    rownames(forest.ate) <- NULL # just for display
    # forest.ate
    
    res <- rbind(forest.ate, ols.ate)
    
    # Plotting the point estimate of average treatment effect 
    # and 95% confidence intervals around it.
    average_treatment_effect <- ggplot(res) +
                                    aes(x = ranking, y = estimate, group=method, color=method) + 
                                    geom_point(position=position_dodge(0.2)) +
                                    geom_errorbar(aes(ymin=estimate-2*std.err, ymax=estimate+2*std.err), width=.2, position=position_dodge(0.2)) +
                                    ylab("") + xlab("") +
                                    ggtitle("Average CATE within each ranking (as defined by predicted CATE)") +
                                    theme_minimal() +
                                    theme(legend.position="bottom", legend.title = element_blank())
    print(average_treatment_effect)
    
    df <- mapply(function(covariate) {
      # Looping over covariate names
      # Compute average covariate value per ranking (with correct standard errors)
      fmla <- formula(paste0(covariate, "~ 0 + ranking"))
      ols <- lm(fmla, data=transform(data, ranking=factor(ranking)))
      ols.res <- coeftest(ols, vcov=vcovHC(ols, "HC2"))
      # deal with NA problem
      modified_ols <- ols.res[,1:2]
      if(nrow(modified_ols) != 5){
        modified_ols <- data.frame(modified_ols)
        new_row_ols <- data.frame(
          Estimate = 0,
          "Std. Error" = 0,
          row.names = 'factor(ranking)1'
        )
        modified_ols <- as.matrix(rbind(new_row_ols, modified_ols))
      }
      
      # Retrieve results
      avg <- modified_ols[,1]
      stderr <- modified_ols[,2]
      
      # Tally up results
      data.frame(covariate, avg, stderr, ranking=paste0("Q", seq(num.rankings)), 
                 # Used for coloring
                 scaling=pnorm((avg - mean(avg))/sd(avg)), 
                 # We will order based on how much variation is 'explain' by the averages
                 # relative to the total variation of the covariate in the data
                 variation=sd(avg) / sd(data[,covariate]),
                 # String to print in each cell in heatmap below
                 labels=paste0(signif(avg, 3), "\n", "(", signif(stderr, 3), ")"))
    }, covariates, SIMPLIFY = FALSE)
    df <- do.call(rbind, df)
    
    # a small optional trick to ensure heatmap will be in decreasing order of 'variation'
    df$covariate <- reorder(df$covariate, order(df$variation))
    
    # plot heatmap
    heatmap <- ggplot(df) +
                aes(ranking, covariate) +
                geom_tile(aes(fill = scaling)) + 
                geom_text(aes(label = labels)) +
                scale_fill_gradient(low = "#E1BE6A", high = "#40B0A6") +
                ggtitle(paste0("Average covariate values within group (based on CATE estimate ranking)")) +
                theme_minimal() + 
                ylab("") + xlab("CATE estimate ranking") +
                theme(plot.title = element_text(size = 11, face = "bold"),
                      axis.text=element_text(size=11)) 
    print(heatmap)
    
    dev.off()
    
    cost <- 0
    # Fit a policy tree on forest-based AIPW scores
    gamma.matrix <- double_robust_scores(forest.tau)  
    gamma.matrix[,2] <- gamma.matrix[,2] - cost  # Subtracting cost of treatment
    
    train <- 1:(.8*n)
    test <- (.8*n):n
    
    # Fit policy on training subset
    if(setid < 5){
      system.time(
        policy <- policy_tree(matrix(X[train]), gamma.matrix[train,], depth = 2, min.node.size=1, split.step=5)
      )
      
    } else{
      system.time(
        policy <- policy_tree(X[train,], gamma.matrix[train,], depth = 2, min.node.size=1, split.step=5)
      )
    }
    
    policy_tree_graph <- paste0(dir, '/result_', idx, '/policy_tree_graph_', value, '.svg')
    tree.plot = plot(policy, leaf.labels = c("control", "treatment"))
    cat(DiagrammeRsvg::export_svg(tree.plot), file = policy_tree_graph)

    }
}