setwd("~/Facebook")

library(readr)
library(dplyr)
library(Metrics)
library(plyr)

# Create MAP3 evaluation function
map3 <- function(preds, dtrain) {
  labels <- as.list(getinfo(dtrain,"label"))
  num.class = NROW(unique(labels))
  pred <- matrix(preds, nrow = num.class)
  top <- t(apply(pred, 2, function(y) order(y)[num.class:(num.class-2)]-1))
  top <- split(top, 1:NROW(top))
  map <- mapk(3, labels, top)
  return(list(metric = "map3", value = map))
}

# Load train and test data
train <- read_csv("train.csv")
test <- read_csv("test.csv")

# Feature engineering for training set
minute = train$time %% 60
train['hour'] = train['time']%/%60
# train$time <- NULL
train['weekday'] = train['hour']%/%24
train['month'] = train['weekday']%/%30
train['year'] = (train['weekday']%/%365+1)*10.0
train['hour'] = ((train['hour']%%24+1)+minute/60.0)*4.0
add_data = train[train$hour<10,]# add data for periodic time that hit the boundary
add_data$hour = add_data$hour+96
train = rbind(train, add_data)
add_data = train[train$hour>90,]
add_data.hour = add_data$hour-96
train = rbind(train, add_data)
rm(add_data)
train['weekday'] = (train['weekday']%%7+1)*3.0
train['month'] = (train['month']%%12+1)*2.0
train['accuracy'] = log10(train['accuracy'])*10.0

# Feature engineering for test set
minute = test$time %% 60
test['hour'] = test['time']%/%60
# test$time <- NULL
test['weekday'] = test['hour']%/%24
test['month'] = test['weekday']%/%30
test['year'] = (test['weekday']%/%365+1)*10.0
test['hour'] = ((test['hour']%%24+1)+minute/60.0)*4.0
test['weekday'] = (test['weekday']%%7+1)*3.0
test['month'] = (test['month']%%12+1)*2.0
test['accuracy'] = log10(test['accuracy'])*10.0

# Extract
place_ids <- as.data.frame(table(train$place_id))
onetimers <- place_ids[place_ids$Freq<11,'Var1']
train <- filter(train, ! place_id %in% onetimers)

# Create grid
x_nsplit = 32
y_nsplit = 50
x_gridsize <- 10/x_nsplit
y_gridsize <- 10/y_nsplit
xmin = 0
ymin = 0
i=1
j=1
val_results <- NULL
num_rounds = 50

# Save validation results for every grid for fixed num_rounds
val_results <- read_csv(paste0('val_results_', num_rounds, '.csv'))

# Loop over the grid (we could do this in parallel with for each, but quick tests show that all cores could be used by XGBoost for a single grid which gave a slightly faster results)
for(i in 1:x_nsplit){
  for(j in 1:y_nsplit){
#     if(file.exists(paste0("test_output/test_", i, "_", j, ".csv"))) {
#       cat("skipping\n")
#       next;
#     }
    if(tail(duplicated(rbind(val_results[,c(1:2)],c(i,j))),1)) {
      cat("skipping\n")
      next;
    }
    xmin = (i-1)*x_gridsize
    ymin = (j-1)*y_gridsize
    xmax = i*x_gridsize
    ymax = j*y_gridsize
    cat(sprintf("xmin, ymin: %s, %s\n", xmin, ymin))
    cat(sprintf("i, j: %s, %s\n", i, j))

    train_1 <- filter(train, xmin <= x & x < xmax & ymin <= y & y < ymax)
    tlabels <- as.numeric(as.factor(train_1$place_id))-1
    train_1$label <- tlabels


    small_train = train_1[train_1$time < 7.5e5,]

    place_ids_1_train <- as.data.frame(table(small_train$place_id))
    onetimers_1_train <- place_ids_1_train[place_ids_1_train$Freq<10,'Var1']
    small_train <- filter(small_train, ! place_id %in% onetimers_1_train)

    # tlabels <- as.numeric(as.factor(small_train$place_id))-1
    # small_train$label <- tlabels

    small_val = train_1[train_1$time >= 7.5e5,]


    xgtrain = xgb.DMatrix(as.matrix(small_train[,-c(which(colnames(train_1) %in% c("row_id", "time", "label", "place_id")))]), label = small_train$label)
    xgtest = xgb.DMatrix(as.matrix(small_val[,-c(which(colnames(train_1) %in% c("row_id", "time", "label", "place_id")))]), label = small_val$label)

    # test_1 <- filter(test, x1_min <= x & x < x1_max & y1_min <= y & y < y1_max)
    # xgtest_all = xgb.DMatrix(as.matrix(test_1))

    # Set parameters for XGBoost
    param0 <- list(
      # some generic, non specific params
      "objective"  = "multi:softprob"
      , "eval_metric" = "mlogloss"
      , "eta" = 0.1
      , "subsample" = 0.80
      , "colsample_bytree" = 0.85
      , "min_child_weight" = 1
      , "max_depth" = 7
      , "num_class" = NROW(unique(train_1$place_id))
    )

    watchlist <- list('train' = xgtrain)
    set.seed(3017)
    model = xgb.train(
      nrounds = num_rounds
      , params = param0
      , data = xgtrain
      , watchlist = watchlist
      , print.every.n = 10
      , nthread = 31
    )
    p <- predict(model, xgtest)

    labels <- as.list(small_val$label)
    num.class = NROW(unique(train_1$place_id))
    pred <- matrix(p, nrow = num.class)
    top <- t(apply(pred, 2, function(y) order(y)[num.class:(num.class-2)]-1))
    top <- split(top, 1:NROW(top))
    map <- mapk(3, labels, top)

    print(map)
    val_results <- rbind(val_results, c(i,j,map))
    write.csv(val_results, paste0(paste0("val_results_", num_rounds, ".csv")), row.names = FALSE)

    gc()
  }
}

# Only extracts the tiles with a validation score higher than 0.57
val_results_sub = val_results[val_results[,3]>0.57,]

# For these tiles train on the complete training data set and predict on test
for(xyz in 1:NROW(val_results_sub)){
  i = val_results_sub[xyz,1][[1]]
  j = val_results_sub[xyz,2][[1]]

  if(file.exists(paste0("test_output/test_", i, "_", j, ".csv"))) {
    cat("skipping\n")
    next;
  }

  xmin = (i-1)*x_gridsize
  ymin = (j-1)*y_gridsize
  xmax = i*x_gridsize
  ymax = j*y_gridsize
  cat(sprintf("xmin, ymin: %s, %s\n", xmin, ymin))
  cat(sprintf("i, j: %s, %s\n", i, j))

  train_1 <- filter(train, xmin <= x & x < xmax & ymin <= y & y < ymax)
  tlabels <- as.numeric(as.factor(train_1$place_id))-1
  train_1$label <- tlabels

  place_ids_1 <- as.data.frame(table(train_1$place_id))
  onetimers_1 <- place_ids_1[place_ids_1$Freq<10,'Var1']
  train_1 <- filter(train_1, ! place_id %in% onetimers_1)

  tlabels <- as.numeric(as.factor(train_1$place_id))-1
  train_1$label <- tlabels


  xgtrain = xgb.DMatrix(as.matrix(train_1[,-c(which(colnames(train_1) %in% c("row_id", "time", "label", "place_id")))]), label = train_1$label)

  param0 <- list(
    # some generic, non specific params
    "objective"  = "multi:softprob"
    , "eval_metric" = "mlogloss"
    , "eta" = 0.1
    , "subsample" = 0.80
    , "colsample_bytree" = 0.85
    , "min_child_weight" = 1
    , "max_depth" = 7
    , "num_class" = NROW(unique(train_1$place_id))
  )

  watchlist <- list('train' = xgtrain)
  set.seed(3017)
  model = xgb.train(
    nrounds = (num_rounds*1.07)
    , params = param0
    , data = xgtrain
    , watchlist = watchlist
    , print.every.n = 10
    , nthread = 31
  )

  test_1 <- filter(test, xmin <= x & x < xmax & ymin <= y & y < ymax)
  xgtest_all = xgb.DMatrix(as.matrix(test_1[,-c(which(colnames(test_1) %in% c("row_id", "time", "label", "place_id")))]))

  p_test <- predict(model, xgtest_all)
  num.class = NROW(unique(train_1$place_id))
  pred <- matrix(p_test, nrow = num.class)
  top <- t(apply(pred, 2, function(y) order(y)[num.class:(num.class-4)]-1))
  top <- split(top, 1:NROW(top))
  test_preds <- cbind(test_1$row_id, data.frame(matrix(unlist(top), nrow=NROW(top), byrow=T),stringsAsFactors=FALSE))

  test_preds$X1 <- mapvalues(test_preds$X1, from = tlabels, to = train_1$place_id, warn_missing = FALSE)
  test_preds$X2 <- mapvalues(test_preds$X2, from = tlabels, to = train_1$place_id, warn_missing = FALSE)
  test_preds$X3 <- mapvalues(test_preds$X3, from = tlabels, to = train_1$place_id, warn_missing = FALSE)
  test_preds$X4 <- mapvalues(test_preds$X4, from = tlabels, to = train_1$place_id, warn_missing = FALSE)
  test_preds$X5 <- mapvalues(test_preds$X5, from = tlabels, to = train_1$place_id, warn_missing = FALSE)

  write.csv(test_preds, paste0("test_output/test_", i, "_", j,".csv"), row.names = FALSE)

  top_prob <- t(apply(pred, 2, function(y) sort(y)[num.class:(num.class-4)]))
  write.csv(top_prob, paste0("test_output_prob/test_", i, "_", j,".csv"), row.names = FALSE)

  gc()

}





# Combine predictions of tiles
dataset <-NULL
file_list <- list.files("test_output/")
for (file in file_list){

#   if(NROW(val_results_sub[val_results_sub$V1 == substr(file, 6,7) & val_results_sub$V2 == substr(file, 9,10),]) == 0) {
#     print("skipping")
#     next;
#   }

  # if the merged dataset doesn't exist, create it
  if (!exists("dataset")){
    dataset <- read.table(paste0('test_output/', file), header=TRUE, sep=",")
  }

  # if the merged dataset does exist, append to it
  if (exists("dataset")){
    temp_dataset <-read.table(paste0('test_output/', file), header=TRUE, sep=",")
    dataset<-rbind(dataset, temp_dataset)
    rm(temp_dataset)
  }

}
write.csv(dataset, paste0("test_output_", num_rounds, ".csv"), row.names = FALSE)

dataset_prob <- NULL
file_list_prob <- list.files("test_output_prob/")
for (file in file_list){

#   if(NROW(val_results_sub[val_results_sub$V1 == substr(file, 6,7) & val_results_sub$V2 == substr(file, 9,10),]) == 0) {
#     print("skipping")
#     next;
#   }

  # if the merged dataset doesn't exist, create it
  if (!exists("dataset")){
    dataset_prob <- read.table(paste0('test_output_prob/', file), header=TRUE, sep=",")
  }

  # if the merged dataset does exist, append to it
  if (exists("dataset")){
    temp_dataset <-read.table(paste0('test_output_prob/', file), header=TRUE, sep=",")
    dataset_prob<-rbind(dataset_prob, temp_dataset)
    rm(temp_dataset)
  }

}

dataset_sub = dataset[dataset_prob$V1>0.46,]


test_preds <- dataset_sub

# Combine predictions with best KNN submission
submission_1 <- NULL
submission_1$row_id <- test_preds$test_1.row_id
submission_1$place_id <- paste(test_preds$X1, test_preds$X2, test_preds$X3, sep = ' ')
submission_1 <- as.data.frame(submission_1)
NROW(submission_1)

submission_best <- read_csv("submission_sample_bestLB_20160622.csv")

submission_best_f <- filter(submission_best, ! row_id %in% submission_1$row_id)

submission <- rbind(submission_1, submission_best_f)

write.csv(submission_1, "submission_xgb_26_sub.csv", row.names = FALSE)