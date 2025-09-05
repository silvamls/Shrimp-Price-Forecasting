#--------------------------------------------------------------------------------------------------------#
#                               Machine learning algorithm to predict prices                             #
#             Feed Foward Neural Networks and Long-short term memory- LSTM to predict prices             #
#                Testing Multiple explanatory variables combinations (x1,x2,x3)                          #
#    Testing several architectures (Single layer, multi and deep layers neural networks)                 #
#                       testing multiple look_backs and activation functions                             #
#                                   by: Matheus Lourenco 04/2025                                         #
#--------------------------------------------------------------------------------------------------------#

#cleaning workspace..
rm(list=ls())

#defining workspace..
dir= "C:/Matheus/Universidade/Doutorado/Vinicius"
setwd(dir)
run_models=FALSE

#Python must be installed. version:3.10 is compatible with Keras: https://www.python.org/downloads/release/python-3100/
#Click in "Windows installer (64-bit)" and select "add Python 3.10 to PATH" to be used in any terminal
#Install Anaconda: https://www.anaconda.com/download/
#Install Rtools(34): https://cran.r-project.org/bin/windows/Rtools/
#installing Keras steps..
#install.packages("keras") 
#library(keras)
#install_keras(method = "virtualenv", python_version = "3.10") #now you can install KERAS API in R
#testing 
library(keras)
to_categorical(0:3) #testing keras 
#packages for parallel processing---
#install.packages("foreach")
#install.packages("doParallel")
#install.packages("tensorflow")
#install.packages("ggplot2")
library(gridExtra)
library(tensorflow)
library(keras)
library(foreach)
library(doParallel)
library(ggplot2)
library(scales)
library(GGally)
library(tidyr)
library(dplyr)
library(combinat)
library(corrplot)
library(randomForest)
library(DALEX)


#reading csv data file
prices<- read.csv("precos_camaroes_completo.csv",sep=",",dec=".")

#creating the year and month variables
split_year_month <- strsplit(prices$YearMonth, "-")
prices$year <- sapply(split_year_month, `[`, 1)
prices$month <- sapply(split_year_month, `[`, 2)
prices$index<- seq(1:length(prices$year))

# convert YearMonth to date
prices$YearMonth <- as.Date(paste0(prices$YearMonth, "-01"))
# Detect where Aquaculture_Export_Dol is NA
na_ranges <- prices %>%
  mutate(is_na = is.na(Aquaculture_Export_Dol),
         time_id = row_number()) %>%
  mutate(group = cumsum(is_na != lag(is_na, default = FALSE))) %>%
  group_by(group) %>%
  filter(is_na) %>%
  summarise(
    start = min(YearMonth),
    end = max(YearMonth),
    n_months = n()
  ) %>%
  filter(n_months >= 1)  # taking 1 month or more


#exploratory analysis -----
# First long format 
prices_long <- prices %>%
 tidyr::pivot_longer(cols = c(
    Aquaculture_Export_Dol,
    Aquaculture_whlsl_Dol,
    Capture_Export_Dol,
    Exchange_Rate
  ),
  names_to = "series", values_to = "value") %>%
  mutate(series = dplyr::recode(series,
                                "Aquaculture_Export_Dol" = "AED",  # Export farmed shrimp
                                "Aquaculture_whlsl_Dol" = "AWD",  # Wholesale farmed shrimp
                                "Capture_Export_Dol"    = "WED",  # Export wild shrimp
                                "Exchange_Rate"         = "ER"))  # USD/BRL exchange rate

#Time series plot...
p1 <- ggplot(prices_long, aes(x = YearMonth, y = value, color = series)) +
  geom_rect(data = na_ranges,
            aes(xmin = start, xmax = end, ymin = -Inf, ymax = Inf),
            inherit.aes = FALSE, fill = "gray90", alpha = 0.9) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 1.5) +
  scale_x_date(
    breaks = seq.Date(as.Date("2013-01-01"), as.Date("2025-09-01"), by = "6 months"),
    limits = as.Date(c("2013-01-01", "2025-09-01")),
    labels = date_format("%b\n%Y"),
    expand = c(0,0)) +
  labs(x = "Date", y = "Price (USD/kg)", color = "") +
  theme_classic(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.background = element_blank(),
    plot.margin = unit(rep(0.05, 4), "cm"),
    legend.position = "bottom",
    legend.box.spacing = unit(0, "pt"))

p1
#saving as png file
ggplot2::ggsave("Price_Series.png",plot=p1, device = "png", units = "cm",
                                      width = 29, height = 18)

#Correlation graphs using GGally
selected <- prices %>%
  select(Aquaculture_Export_Dol, Aquaculture_whlsl_Dol,
         Capture_Export_Dol, Exchange_Rate) %>%
  rename(
    AED = Aquaculture_Export_Dol,   # Export farmed shrimp
    AWD = Aquaculture_whlsl_Dol,    # Wholesale farmed shrimp
    WED = Capture_Export_Dol,       # Export wild shrimp
    ER  = Exchange_Rate             # USD/BRL exchange rate
  )


#Cross-Correlation graphics
custom_cor <- function(data, mapping, ...) {
  ggplot(data = data, mapping = mapping) +
    geom_point(alpha = 0.5, color = "gray40") +
    geom_smooth(method = "loess", se = FALSE, color = "blue", ...) +
    geom_smooth(method = "lm", se = FALSE, color = "red", linetype = "dashed", ...)
}
p2 <- GGally::ggpairs(selected,
                      upper = list(continuous = wrap("cor", size = 4)),
                      lower = list(continuous = custom_cor),
                      diag = list(continuous = wrap("barDiag", fill = "gray80"))) +
  theme_classic(base_size = 14)
p2

ggplot2::ggsave("Correlation_Series.png",plot=p2, device = "png", units = "cm",
                width = 30, height = 23)



#===========================================================================
# Trying Feed-Forward neural networks and LSTM models to forecast Price data
#===========================================================================
#Logical indicating if the Foreach loop to test all models will run
if (run_models==TRUE) {

#libraries..
library(dplyr)
library(keras)
library(tensorflow)
library(doParallel)
library(foreach)


#Filtering non-NA values and creating the explanatory variables
prices_sub <- filter(prices, !is.na(Aquaculture_Export_Dol))
y  <- prices_sub$Aquaculture_Export_Dol 
x1 <- prices_sub$Aquaculture_whlsl_Dol
x2 <- prices_sub$Capture_Export_Dol
x3 <- prices_sub$Exchange_Rate

# Standardizing variables (Z-score)
df <- data.frame(y = y, x1 = x1, x2 = x2, x3 = x3)
means <- apply(df, 2, mean)
sds <- apply(df, 2, sd)
prices_scaled <- scale(df, center = means, scale = sds)

#Separating 70% of the data to train and 30% to test (keeping the temporal sequence)
n <- nrow(prices_scaled)
cut_point <- floor(0.7 * n)
train_data <- prices_scaled[1:cut_point, ]
test_data  <- prices_scaled[(cut_point + 1):n, ]

# --------------------------------------------------
# All explanatory variables combinations (features)
# --------------------------------------------------
feature <- c("x1", "x2", "x3")
#Using combn to avoid repeated combinations
feature_combinations <- do.call(c, lapply(1:length(feature), function(k) combn(feature, k, simplify = FALSE)))

# ------------------------------------------
# Defining hyperparameters and complete grid
# ------------------------------------------
nn_type <- c("FF", "LSTM") #Feed-Forward and Long-Short therm memory (LSTM)
look_back <- c(2,3,4)   #look back time steps
act_function <- c("relu", "tanh") #activation functions
batch_size <- c(8,16,32)   #size of the batch 
epochs <- c(100,150,200)  #epochs of training
units <- c(30,50,80) #units in each layer
dropout <- c(0.1,0.2,0.3)  #Percentage of lost neurons
optimizer <- c("adam","rmsprop") 
loss <- "mean_squared_error"
metrics <- "mean_absolute_error"
pattience <- 30 #callback to avoid over training (30 epochs without improvement)

#expanded grid of parameters
hyperparameter_combinations <- expand.grid(
  nn_type = nn_type,
  act_function = act_function,
  look_back = look_back,
  batch_size = batch_size,
  epochs = epochs,
  units = units,
  dropout = dropout,
  optimizer = optimizer,
  stringsAsFactors = FALSE
)
# For feed-forward networks look_Back is 0 
hyperparameter_combinations$look_back[hyperparameter_combinations$nn_type == "FF"] <- 0

# LSTMs have fixed activation functions "sigmoid & tanh" 
hyperparameter_combinations$act_function <- as.character(hyperparameter_combinations$act_function)
hyperparameter_combinations$act_function[hyperparameter_combinations$nn_type == "LSTM"] <- "sigmoid&tanh"

# ----------------------------------------------------------
# Function to create Sliding windows for the input LSTM data
# ----------------------------------------------------------
sliding_window <- function(data_x, data_y, look_back) {
  n <- nrow(data_x)
  n_features <- ncol(data_x)
  x <- array(NA, dim = c(n - look_back, look_back, n_features))
  y <- numeric(n - look_back)
  for (i in 1:(n - look_back)) {
    x[i,,] <- data_x[i:(i + look_back - 1), , drop = FALSE]
    y[i] <- data_y[i + look_back]
  }
  return(list(x = x, y = y))
}

# --------------------------------------------------------
# Function to evaluate the models (MAE, RMSE e R²)
# --------------------------------------------------------
evaluate_model <- function(model, x, y) {
  y_pred <- model %>% predict(x)
  mae <- mean(abs(y - y_pred))
  rmse <- sqrt(mean((y - y_pred)^2))
  ss_res <- sum((y - y_pred)^2)
  ss_tot <- sum((y - mean(y))^2)
  r2 <- 1 - ss_res / ss_tot
  return(c(mae = mae, rmse = rmse, r2 = r2))
}

set.seed(123)
n_trials  <- 50    # Number of samples by each batch 
n_batches <- 10    # number of batches

#Sampling the hyperparameters grid in equal samples
n_total       <- nrow(hyperparameter_combinations)
batch_indices <- cut(seq_len(n_total), breaks = n_batches, labels = FALSE)
batches       <- split(hyperparameter_combinations, batch_indices)

#Number of cores for parallel processing
numCores <- max(detectCores() - 1, 1)

# External loop for each features combination
for (features_used in feature_combinations) {
  features_used <- unlist(features_used)
  cat("Features:", paste(features_used, collapse = ", "), "\n")
  
  #Preparing training/testing data
  x_train <- train_data[, features_used, drop = FALSE]
  y_train <- train_data[, "y"]
  x_test  <- test_data[, features_used, drop = FALSE]
  y_test  <- test_data[, "y"]
  
  # Loop through the batches
  for (b in seq_along(batches)) {
    cat(sprintf("  Batch %d/%d\n", b, n_batches))
    hyper_batch <- batches[[b]]
    
    #Random Sampling inside the current batch
    trials <- hyper_batch %>% slice_sample(n = n_trials)
    
    #Clustering ...
    cl <- makeCluster(numCores)
    registerDoParallel(cl)
    
    # Parallel foreach in trials
    results_df <- foreach(idx = 1:n_trials, 
                          .combine = rbind,
                          .packages = c("keras","tensorflow")) %dopar% {
      # Extracting parameters from the current iteration
      param     <- trials[idx, ]
      nn_type   <- param$nn_type
      act_fun   <- param$act_function
      look_back <- param$look_back
      batch_sz  <- param$batch_size
      n_epoch   <- param$epochs
      n_units   <- param$units
      drop_r    <- param$dropout
      optim     <- param$optimizer
                    
      # Callback
      cb <- callback_early_stopping(
           monitor = "val_mean_absolute_error",
           patience = pattience,
           restore_best_weights = TRUE)
                            
      #Preparing 2D data for FF and 3D data for LSTM
      if (nn_type == "LSTM") {
        sw_tr <- sliding_window(x_train, y_train, look_back)
        sw_ts <- sliding_window(x_test,  y_test,  look_back)
        x_tr <- sw_tr$x;   y_tr <- sw_tr$y
        x_ts <- sw_ts$x;   y_ts <- sw_ts$y
        input_shape <- c(look_back, ncol(x_train))
        } else {
        x_tr <- x_train;   y_tr <- y_train
        x_ts <- x_test;    y_ts <- y_test
        input_shape <- ncol(x_train)
        }
                          
        #Generic function for creating, training and evaluating each architecture
        run_arch <- function(arch) {
          m <- keras_model_sequential()
          
          # If nn_type== FF               
          if (nn_type == "FF") {
          # First layer +dropout
          m <- m %>% 
          layer_dense(units = n_units, activation = act_fun, input_shape = input_shape) %>%
          layer_dropout(rate = drop_r)
          # Second layer if multi/deep
          if (arch %in% c("multi","deep")) {
          m <- m %>% 
          layer_dense(units = n_units, activation = act_fun) %>%
          layer_dropout(rate = drop_r)
          }
          #Third layer if deep
          if (arch == "deep") {
          m <- m %>% 
          layer_dense(units = n_units, activation = act_fun) %>%
          layer_dropout(rate = drop_r)
          }
            } else {
          # If nn_type== LSTM
          if (arch == "single") {
          m <- m %>% layer_lstm(units = n_units, input_shape = input_shape)
          } else if (arch == "multi") {
          m <- m %>% 
          layer_lstm(units = n_units, input_shape = input_shape, return_sequences = TRUE) %>%
          layer_dropout(rate = drop_r) %>%
          layer_lstm(units = n_units, return_sequences = FALSE)
          } else {
          m <- m %>% 
          layer_lstm(units = n_units, input_shape = input_shape, return_sequences = TRUE) %>%
          layer_dropout(rate = drop_r) %>%
          layer_lstm(units = n_units, return_sequences = TRUE) %>%
          layer_dropout(rate = drop_r) %>%
          layer_lstm(units = n_units, return_sequences = FALSE)
          }
          m <- m %>% layer_dropout(rate = drop_r)
          }
            
         #Output + compiling 
         m <- m %>% 
         layer_dense(units = 1, activation = "linear") %>%
         compile(loss = loss, optimizer = optim, metrics = metrics)
                            
         #Fitting and evaluating
         history <- m %>% fit(
         x = x_tr, y = y_tr,
         batch_size = batch_sz,
         epochs = n_epoch,
         validation_split = 0.2,
         callbacks = list(cb),
         verbose = 0
         )
        #evaluating
        ev <- evaluate_model(m, x_ts, y_ts)
                              
        #Removing the history and the fitted model (saving memmory)
        rm(m)
        rm(history)
                              
      #Cleaning session 
      k_clear_session(); tf$keras$backend$clear_session(); gc()
                            
      data.frame(
      features        = paste(features_used, collapse = ","),
      nn_type         = nn_type,
      arch            = arch,
      look_back       = look_back,
      act_fun         = act_fun,
      batch_size      = batch_sz,
      epochs          = n_epoch,
      units           = n_units,
      dropout         = drop_r,
      optimizer       = optim,
      mae             = ev["mae"],
      rmse            = ev["rmse"],
      r2              = ev["r2"],
      stringsAsFactors = FALSE
      )
      }
    
    #Closing foreach and returning the result
    do.call(rbind, lapply(c("single", "multi", "deep"), run_arch))
   }  #closing %dopar% block
    
  # Encerrar cluster e liberar recursos
  stopCluster(cl)
  registerDoSEQ()
  invisible(gc())
  
  # Saving in CSV each batch
  csv_name <- sprintf("random_search_%s_batch_%02d.csv",
                      paste(features_used, collapse = "_"), b)
  write.csv(results_df, csv_name, row.names = FALSE)
    
  #Print the best batch result
  best <- results_df %>% arrange(mae) %>% slice(1)
  print(best)
  }
}

#Combining all final csv files 
csv_files <- list.files(pattern = "^random_search_.*_batch_\\d+\\.csv$")
final_results <- do.call(rbind, lapply(csv_files, read.csv, stringsAsFactors = FALSE))
write.csv(final_results, "final_results_all_features.csv", row.names = FALSE)
}

#---------------------------------------------------------------------------------------------


#=====================================
#Exploratory analysis and best model
#====================================
library(tidyr)
library(dplyr)
library(ggplot2)
library(randomForest)
library(DALEX)


# ----------------------------------------------------------
# Function to create Sliding windows for the input LSTM data
# ----------------------------------------------------------
sliding_window <- function(data_x, data_y, look_back) {
  n <- nrow(data_x)
  n_features <- ncol(data_x)
  x <- array(NA, dim = c(n - look_back, look_back, n_features))
  y <- numeric(n - look_back)
  for (i in 1:(n - look_back)) {
    x[i,,] <- data_x[i:(i + look_back - 1), , drop = FALSE]
    y[i] <- data_y[i + look_back]
  }
  return(list(x = x, y = y))
}

# --------------------------------------------------------
# Function to evaluate the models (MAE, RMSE e R²)
# --------------------------------------------------------
evaluate_model <- function(model, x, y) {
  y_pred <- model %>% predict(x)
  mae <- mean(abs(y - y_pred))
  rmse <- sqrt(mean((y - y_pred)^2))
  ss_res <- sum((y - y_pred)^2)
  ss_tot <- sum((y - mean(y))^2)
  r2 <- 1 - ss_res / ss_tot
  return(c(mae = mae, rmse = rmse, r2 = r2))
}



#read "final_results_all_features2.csv" file if you don't want to run the foreach loop testing several models
if (!exists("final_results")) {
  final_results <- read.csv("final_results_all_features2.csv", sep = ",", dec = ".")
}

#best architectures (mean MAE,RMSE and R2)
final_results %>%
  group_by(arch) %>%
  summarise(
    mean_mae = mean(mae, na.rm = TRUE),
    sd_mae   = sd(mae, na.rm = TRUE),
    mean_rmse = mean(rmse, na.rm = TRUE),
    sd_rmse   = sd(rmse, na.rm = TRUE),
    mean_r2   = mean(r2, na.rm = TRUE),
    sd_r2     = sd(r2, na.rm = TRUE),
    n = n()
  )

#best MAE, best 5 models
top_models <- final_results %>%
  arrange(mae) %>%
  head(5)
print(top_models)

#write top models
write.csv(top_models, "top_models_final_results.csv", row.names = FALSE)


#long format to plot hyperparameters vs MAE
final_long <- final_results %>%
  mutate(across(c(nn_type,arch, units, look_back, act_fun, batch_size, epochs, features),
                as.character)) %>%
  pivot_longer(
    cols = c(arch,nn_type, units, look_back, act_fun, batch_size, epochs, features),
    names_to = "hyperparameter",
    values_to = "value"
  ) %>%
  mutate(value = as.factor(value)) %>%
  mutate(value = dplyr::recode(value,
                                "x1" = "AWD",  # Wholesale farmed shrimp
                                "x1,x2" = "AWD,WED",  # Wholesale farmed shrimp
                                "x1,x2,x3" = "AWD,WED,ER",  # Wholesale farmed shrimp
                                "x1,x3" = "AWD,ER",  # Wholesale farmed shrimp
                                "x2,x3" = "WED,ER",  # Wholesale farmed shrimp
                                "x2" = "WED",  # Export wild shrimp
                                "x3" = "ER"))  # USD/BRL exchange rate
 
head(final_long)

p3<-ggplot(final_long, aes(x = value, y = mae)) +
  geom_boxplot(fill = "grey80", color = "black") +
  facet_wrap(~hyperparameter, scales = "free", ncol = 2) +
  coord_cartesian(ylim = c(0.15, 0.6)) +  #changing the range for 0-0.6
  theme_classic(base_size = 14) +
  labs(
    title = "",
    x = "Hyperparameter value",
    y = "Mean Absolute Error (MAE)",
    fill = " "
  ) +
  theme(
    strip.background = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.margin = unit(c(0.05, 0.05, 0.05, 0.05), "mm"))
p3

ggplot2::ggsave("MAE_Hyperparameters.png",plot=p3, device = "png", units = "cm",
                width = 18, height = 25)

#R² by neural network type (LSTM ou FF)
library(ggplot2)
library(tidyr)
library(dplyr)

# Reestrutura os dados para formato longo
final_results_long <- final_results %>%
  select(nn_type, mae, rmse, r2) %>%
  pivot_longer(cols = c(mae, rmse, r2), names_to = "metric", values_to = "value")

# MAE,RMSE and R2 by neural network type 
p4<-ggplot(final_results_long, aes(x = nn_type, y = value)) +
  geom_boxplot(fill = "#009E73") +
  facet_wrap(~metric, scales = "free_y") +
  labs(
    title = "Metrics by Neural Network Type",
    x = "Neural Network type",
    y = "Metric Value"
  ) + 
  theme_classic(base_size = 14) +
  theme(strip.background = element_blank(),
        plot.margin = unit(c(0.05, 0.05, 0.05, 0.05), "mm"))
p4
ggplot2::ggsave("MAE_RMSE_R2_nn_type.png",plot=p4, device = "png", units = "cm",
                width = 18, height = 14)


#Correlation matrix between hyperparameters
numeric_vars <- final_results %>%
  select(mae, rmse, r2, look_back, units, dropout, epochs)

cor_matrix <- cor(numeric_vars, use = "complete.obs")
print(round(cor_matrix, 2))

library(corrplot)
#png file
png("correlation_matrix.png", width = 2000, height = 1600, res = 300)
# Plot
corrplot(cor_matrix, method = "color", type = "upper", tl.col = "black")
#close device
dev.off()
# Plot again in the current device
corrplot(cor_matrix, method = "color", type = "upper", tl.col = "black")

#-------------------------------------------
#evaluate what features are more important
#-------------------------------------------
#Train a Random Forest surrogate and permutation importance analysis
#libraries
library(randomForest)
library(DALEX)

# Removing  RMSE and R2 to avoid correlation with MAE
final_clean <- final_results[, !(names(final_results) %in% c("rmse", "r2"))]

#Categorical variables 
categorical_vars <- names(final_clean[!names(final_clean) %in% "mae"])

# Loop to calculate the metrics for each categorical vars
for (var in categorical_vars) {
  metrics <- final_clean %>%
    group_by(.data[[var]]) %>%
    summarise(
      median_mae = median(mae, na.rm = TRUE),
      sd_mae = sd(mae, na.rm = TRUE),
      count = n()
    )
  
  print(paste("Métricas para", var))
  print(metrics)
}


# Training a Random Forest model....
rf <- randomForest(mae ~ ., data = final_clean, ntree = 500)

# Explain model
expl <- explain(
  model = rf,
  data = final_clean[, !(names(final_clean) %in% c("mae"))],
  y = final_clean$mae
)

# Calculating permutation features importance Via DALEX
vi <- model_parts(expl, type = "variable_importance", B = 50)

#Graphic output
p5 <- plot(vi) +
  labs(x = "Root Mean Square Error (RMSE) loss after permutations") +
  theme_classic() +
  theme(
    legend.position = "none",
    plot.title = element_blank(),
    plot.subtitle = element_blank(),
    axis.title.y = element_blank(),
    strip.text = element_blank(),
    strip.background = element_blank()
  )
p5

#saving as png file..
ggplot2::ggsave("Feature importance.png",plot=p5, device = "png", units = "cm",
                width = 18, height = 14)



                            #=================================================#
                            #      Defining the best model (Lowest MAE)       #
                            #=================================================#
#reading csv data file
setwd(dir)
#reading csv data file
prices<- read.csv("precos_camaroes_completo.csv",sep=",",dec=".")

#creating the year and month variables
split_year_month <- strsplit(prices$YearMonth, "-")
prices$year <- sapply(split_year_month, `[`, 1)
prices$month <- sapply(split_year_month, `[`, 2)
prices$index<- seq(1:length(prices$year))

# convert YearMonth to date
prices$YearMonth <- as.Date(paste0(prices$YearMonth, "-01"))
# Detect where Aquaculture_Export_Dol is NA
na_ranges <- prices %>%
  mutate(is_na = is.na(Aquaculture_Export_Dol),
         time_id = row_number()) %>%
  mutate(group = cumsum(is_na != lag(is_na, default = FALSE))) %>%
  group_by(group) %>%
  filter(is_na) %>%
  summarise(
    start = min(YearMonth),
    end = max(YearMonth),
    n_months = n()
  ) %>%
  filter(n_months >= 1)  # taking 1 month or more


#read "final_results_all_features2.csv" file if you don't want to run the foreach loop testing several models
if (!exists("final_results")) {
  final_results <- read.csv("final_results_all_features2.csv", sep = ",", dec = ".")
}

#Libraries 
library(gridExtra)
library(tensorflow)
library(keras)
library(foreach)
library(doParallel)
library(ggplot2)
library(GGally)
library(tidyr)
library(dplyr)
library(combinat)
library(corrplot)

#defining the best model (Lowest MAE)
best_model<-final_results[which.min(final_results$mae),]

#In case the best model is a LSTM model
if (best_model$nn_type=='LSTM') {
  
#best model hyperparameters
best_nn_type= best_model$nn_type
best_features=best_model$features
best_features <- unlist(strsplit(best_model$features, ",\\s*"))
best_n_features<- length(best_features)
best_architecture=best_model$arch
best_act_function=best_model$act_fun
best_look_back=best_model$look_back
best_batch_size=best_model$batch_size
best_epochs=best_model$epochs
best_units=best_model$units
best_dropout=best_model$dropout
best_optimizer=best_model$optimizer

#In LSTM case we need sliding windows 3D data (n_samples, timesteps, n_features)
#Filtering non-NA values and creating the explanatory variables
prices_sub <- filter(prices, !is.na(Aquaculture_Export_Dol))
y  <- prices_sub$Aquaculture_Export_Dol 
x1 <- prices_sub$Aquaculture_whlsl_Dol
x2 <- prices_sub$Capture_Export_Dol
x3 <- prices_sub$Exchange_Rate

# Standardizing variables (Z-score)
df <- data.frame(y = y, x1 = x1, x2 = x2, x3 = x3)
means <- apply(df, 2, mean)
sds <- apply(df, 2, sd)
prices_scaled <- scale(df, center = means, scale = sds)

#Train data
train_sw <- sliding_window(prices_scaled[, best_features, drop = FALSE], prices_scaled[,"y"], best_look_back)

x_train_array <- train_sw$x
y_train_sw    <- train_sw$y

#taking the entire data frame of prices (original data-frame)
x_pred<- prices 
y  <- prices$Aquaculture_Export_Dol 
x1 <- prices$Aquaculture_whlsl_Dol
x2 <- prices$Capture_Export_Dol
x3 <- prices$Exchange_Rate
# Standardizing variables (Z-score)
x_pred <- data.frame(y = y, x1 = x1, x2 = x2, x3 = x3)
means <- apply(x_pred, 2, function(x) mean(x, na.rm = TRUE))
sds   <- apply(x_pred, 2, function(x) sd(x, na.rm = TRUE))
x_pred_scaled <- scale(x_pred, center = means, scale = sds)

#predict (x_data)
pred_sw<-sliding_window(prices_scaled[, best_features, drop = FALSE], x_pred_scaled[,"y"], best_look_back)

x_pred_array <- pred_sw$x

# n_runs for a loop
n_runs <- 30  # n repetitions
pred_matrix <- matrix(NA, nrow = nrow(x_pred_array), ncol = n_runs)

for (r in 1:n_runs) {
  
  #taking the achitecture
  if (best_architecture=="single") {
    
    #singe LSTM layer + 1 layer dropout + 1 layer output without activation         
    model <- keras_model_sequential() %>%
      layer_lstm(units = best_units, input_shape = c(best_look_back, best_n_features)) %>%
      layer_dropout(rate = best_dropout) %>%
      layer_dense(units = 1, activation = "linear")
    
    
  } else if(best_architecture=='multi'){
    #Multilayer LSTM (2) + layer dropout (2) + 1 layer output without activation                
    model<- keras_model_sequential() %>%
      layer_lstm(units = best_units, input_shape = c(best_look_back, best_n_features), return_sequences = TRUE) %>%
      layer_dropout(rate = best_dropout) %>%
      layer_lstm(units = best_units, return_sequences = FALSE) %>%
      layer_dropout(rate = best_dropout) %>%
      layer_dense(units = 1, activation = "linear")
    
    
  } else if(best_architecture=='deep'){
    #Deeplayer LSTM (3) + layer dropout (3) + 1 layer output without activation                                                          
    model <- keras_model_sequential() %>%
      layer_lstm(units = best_units, input_shape = c(best_look_back, best_n_features), return_sequences = TRUE) %>%
      layer_dropout(rate = best_dropout) %>%
      layer_lstm(units = best_units, return_sequences = TRUE) %>%
      layer_dropout(rate = best_dropout) %>%
      layer_lstm(units = best_units, return_sequences = FALSE) %>%
      layer_dropout(rate = best_dropout) %>%
      layer_dense(units = 1, activation = "linear")
  }
  #compiling
  model %>% compile(
    loss = "mean_squared_error",  
    optimizer = best_optimizer,
    metrics = c("mae")
  )
  
  #Training
  model %>% fit(
    x = x_train_array,
    y = y_train_sw,
    batch_size = best_batch_size,
    epochs = best_epochs,
    validation_split = 0.2,
    verbose = 0
  )
  
  #Predicting
  pred <- predict(model, x_pred_array)
  pred_matrix[, r] <- pred[, 1]  
}

#Final statistics
pred_mean <- rowMeans(pred_matrix)
pred_lw <- apply(pred_matrix, 1, quantile, probs = 0.025)  # 2.5%
pred_up <- apply(pred_matrix, 1, quantile, probs = 0.975)  # 97.5%

# Back to the original scale
mean_y <- means["y"]
sd_y <- sds["y"]
pred_mean_orig <- pred_mean * sd_y + mean_y
pred_lw_orig   <- pred_lw * sd_y + mean_y
pred_up_orig   <- pred_up * sd_y + mean_y

#Aligne the real data with predictions
aligned_obs <- tail(prices$Aquaculture_Export_Dol, length(pred_mean))

# 1) Build pred_out with the exact dates that match aligned_obs:
n_preds <- length(pred_mean_orig)
aligned_dates <- tail(prices$YearMonth, n_preds)

#Final data frame
pred_out <- data.frame(
  date= aligned_dates,
  features = paste(best_features, collapse = ","),
  nn_type = best_nn_type,
  model_architecture = "best_model_retrained",
  act_function = best_act_function,
  look_back = best_look_back,
  batch_size = best_batch_size,
  epochs = best_epochs,
  units = best_units,
  dropout = best_dropout,
  optimizer = best_optimizer,
  obs = aligned_obs,
  pred = pred_mean_orig,
  lw = pred_lw_orig,
  up = pred_up_orig
)

library(ggplot2)

#Temporal column
pred_out$t <- 1:nrow(pred_out)

# Save pred_out to CSV
write.csv(pred_out, "pred_out_best_model.csv", row.names = FALSE)

# Plot...
#Plot...
library(ggplot2)
library(dplyr)

# Build the plot
p5 <- ggplot(pred_out, aes(x = date)) +
  # highlight missing‐data periods
  geom_rect(data = na_ranges,
            aes(xmin = start, xmax = end, ymin = -Inf, ymax = Inf),
            inherit.aes = FALSE,
            fill = "gray90", alpha = 0.5) +
  # shaded uncertainty ribbon
  geom_ribbon(aes(ymin = lw, ymax = up), fill = "steelblue2", alpha = 0.3) +
  # predicted line
  geom_line(aes(y = pred), color = "steelblue2", linewidth = 1.2) +
  # observed points and line
  geom_line(aes(y = obs), color = "gray40", linetype = "solid", linewidth= 1.2) +
  geom_point(aes(y = obs), color = "gray40", size = 1.5) +
  # custom x‐axis every 6 months
  scale_x_date(
    breaks = seq.Date(as.Date("2013-01-01"), as.Date("2025-09-01"), by = "6 months"),
    limits = as.Date(c("2013-01-01", "2025-09-01")),
    labels = date_format("%b\n%Y"),
    expand = c(0,0)) +
  labs(
    title = "",
    x     = "Date",
    y     = "Price (USD/kg)",
    color = NULL
  ) +
  theme_classic(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.background = element_blank(),
    plot.margin = unit(c(0.05, 0.05, 0.05, 0.05), "mm"))

# Display and save
print(p5)
#saving as png file
ggplot2::ggsave("Prices_Prediction_LSTM_Bestmodel.png",plot=p5, device = "png", units = "cm",
                width = 29, height = 17)

  #In case the best model is a Feed-Forward neural network
} else if(best_model$nn_type == 'FF') {
  
  # Taking the best hyperparameters
  best_nn_type     <- best_model$nn_type
  best_features    <- unlist(strsplit(best_model$features, ",\\s*"))
  best_n_features  <- length(best_features)
  best_architecture<- best_model$arch
  best_act_function<- best_model$act_fun
  best_batch_size  <- best_model$batch_size
  best_epochs      <- best_model$epochs+100
  best_units       <- best_model$units
  best_dropout     <- best_model$dropout
  best_optimizer   <- best_model$optimizer
  
  #Training data
  #Filtering non-NA values and creating the explanatory variables
  prices_sub <- filter(prices, !is.na(Aquaculture_Export_Dol))
  y  <- prices_sub$Aquaculture_Export_Dol 
  x1 <- prices_sub$Aquaculture_whlsl_Dol
  x2 <- prices_sub$Capture_Export_Dol
  x3 <- prices_sub$Exchange_Rate
  
  # Standardizing variables (Z-score)
  df <- data.frame(y = y, x1 = x1, x2 = x2, x3 = x3)
  means <- apply(df, 2, mean)
  sds <- apply(df, 2, sd)
  prices_scaled <- scale(df, center = means, scale = sds)
  
  #training data (2D data for a feed-Forward)
  x_train_array <- prices_scaled[,best_features]
  y_train_ff    <- prices_scaled[,"y"]
  
  #taking the entire data frame of prices (original data-frame)
  x_pred<- prices 
  y  <- prices$Aquaculture_Export_Dol 
  x1 <- prices$Aquaculture_whlsl_Dol
  x2 <- prices$Capture_Export_Dol
  x3 <- prices$Exchange_Rate
  # Standardizing variables (Z-score)
  x_pred <- data.frame(y = y, x1 = x1, x2 = x2, x3 = x3)
  means <- apply(x_pred, 2, function(x) mean(x, na.rm = TRUE))
  sds   <- apply(x_pred, 2, function(x) sd(x, na.rm = TRUE))
  x_pred_scaled <- scale(x_pred, center = means, scale = sds)
  
  # predict x_data
  x_pred_array <- x_pred_scaled[,best_features]
  
  # Loop through n_runs (Reproducibility)
  n_runs <- 30
  pred_matrix <- matrix(NA, nrow = nrow(x_pred_array), ncol = n_runs)
  
  for (r in 1:n_runs) {
    
    #taking the best model architecture
    if (best_architecture == "single") {
      model <- keras_model_sequential() %>%
        layer_dense(units = best_units, activation = best_act_function, input_shape = best_n_features) %>%
        layer_dropout(rate = best_dropout) %>%
        layer_dense(units = 1, activation = "linear")
      
    } else if (best_architecture == "multi") {
      model <- keras_model_sequential() %>%
        layer_dense(units = best_units, activation = best_act_function, input_shape = best_n_features) %>%
        layer_dropout(rate = best_dropout) %>%
        layer_dense(units = best_units, activation = best_act_function) %>%
        layer_dropout(rate = best_dropout) %>%
        layer_dense(units = 1, activation = "linear")
      
    } else if (best_architecture == "deep") {
      model <- keras_model_sequential() %>%
        layer_dense(units = best_units, activation = best_act_function, input_shape = best_n_features) %>%
        layer_dropout(rate = best_dropout) %>%
        layer_dense(units = best_units, activation = best_act_function) %>%
        layer_dropout(rate = best_dropout) %>%
        layer_dense(units = best_units, activation = best_act_function) %>%
        layer_dropout(rate = best_dropout) %>%
        layer_dense(units = 1, activation = "linear")
    }
    #compiling the model
    model %>% compile(
      loss = "mean_squared_error",
      optimizer = best_optimizer,
      metrics = c("mae")
    )
    #training..
    model %>% fit(
      x = x_train_array,
      y = y_train_ff,
      batch_size = best_batch_size,
      epochs = best_epochs,
      validation_split = 0,
      verbose = 0
    )
    #predict over the new x_data
    pred <- predict(model, x_pred_array)
    pred_matrix[, r] <- pred[, 1]
  }
  
  # Statistics 
  pred_mean <- rowMeans(pred_matrix)
  pred_lw <- apply(pred_matrix, 1, quantile, probs = 0.025)
  pred_up <- apply(pred_matrix, 1, quantile, probs = 0.975)
  
  
  #back to the original scale
  mean_y <- means["y"]
  sd_y   <- sds["y"]
  pred_mean_orig <- pred_mean * sd_y + mean_y
  pred_lw_orig   <- pred_lw   * sd_y + mean_y
  pred_up_orig   <- pred_up   * sd_y + mean_y
  
  aligned_obs <- prices$Aquaculture_Export_Dol
  #out data frame
  pred_out <- data.frame(
    date=prices$YearMonth,
    features = paste(best_features, collapse = ","),
    nn_type = best_nn_type,
    model_architecture = "best_model_retrained",
    act_function = best_act_function,
    look_back = NA,  # não se aplica
    batch_size = best_batch_size,
    epochs = best_epochs,
    units = best_units,
    dropout = best_dropout,
    optimizer = best_optimizer,
    obs = aligned_obs,
    pred = pred_mean_orig,
    lw = pred_lw_orig,
    up = pred_up_orig
  )
  #time couting variable
  pred_out$t <- 1:nrow(pred_out)
  
  # Save pred_out to CSV
  write.csv(pred_out, "pred_out_best_model.csv", row.names = FALSE)
  
  #Plot...
  library(ggplot2)
  library(dplyr)
  
  # Build the plot
  p6 <- ggplot(pred_out, aes(x = date)) +
    # highlight missing‐data periods
    geom_rect(data = na_ranges,
              aes(xmin = start, xmax = end, ymin = -Inf, ymax = Inf),
              inherit.aes = FALSE,
              fill = "gray90", alpha = 0.5) +
    # shaded uncertainty ribbon
    geom_ribbon(aes(ymin = lw, ymax = up), fill = "steelblue2", alpha = 0.3) +
    # predicted line
    geom_line(aes(y = pred), color = "steelblue2", linewidth = 1.2) +
    # observed points and line
    geom_line(aes(y = obs), color = "gray40", linetype = "solid", linewidth= 1.2) +
    geom_point(aes(y = obs), color = "gray40", size = 1.5) +
    # custom x‐axis every 6 months
    scale_x_date(
      breaks = seq.Date(as.Date("2013-01-01"), as.Date("2025-09-01"), by = "6 months"),
      limits = as.Date(c("2013-01-01", "2025-09-01")),
      labels = date_format("%b\n%Y"),
      expand = c(0,0)) +
    labs(
      title = "",
      x     = "Date",
      y     = "Price (USD/kg)",
      color = NULL
    ) +
    theme_classic(base_size = 14) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      strip.background = element_blank(),
      plot.margin = unit(c(0.05, 0.05, 0.05, 0.05), "mm"))
  
  # Display and save
  print(p6)
  #saving as png file
  ggplot2::ggsave("Prices_Prediction_FF_Bestmodel.png",plot=p6, device = "png", units = "cm",
                  width = 29, height = 17)
}

#====================================
#End of Price Predictions...
#====================================