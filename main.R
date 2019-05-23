install.packages('neuralnet')
install.packages('quantmod')
install.packages('Hmisc')
install.packages('Metrics')
library(neuralnet)
library(quantmod)
library(Hmisc)
library(Metrics)

# Make sure working directory is same as file location

# Keep runs consistent
set.seed(2305)

# Load in the dataset
aus <- read.csv("aus_homicide_suicide.csv")  # read csv file 
aus <- aus[1:90,]

# convert to workable format (why is csv being a wee bam)
datanames <- c('fa_h', 'fa_s', 'nfa_h', 'nfa_s')
titlenames <- c('Firearm Homicide', 'Firearm Suicide','Non-Firearm Homicide', 'Non-Firearm Suicide')

# Scaling needs to be changed to Min Max so we can rescale it after the NN 
# Smoothing also applied here
years <- as.numeric(levels(aus[,1]))[1:90]

fa_h <- scale(as.numeric(smooth(aus[,2])))
fa_s <- scale(as.numeric(smooth(aus[,3])))
nfa_h <- scale(as.numeric(smooth(aus[,4])))
nfa_s <- scale(as.numeric(smooth(aus[,5])))

init_data <- data.frame(as.numeric(aus[,2]), as.numeric(aus[,3]), as.numeric(aus[,4]), as.numeric(aus[,5]))
names(init_data) <- paste(datanames)
init_data <- ts(init_data, start=c(1915), end=c(2004), frequency =1)
plot(init_data, main = "Death rates by category in Australia (1915-2004)")

# Create a lagged data frame for all the variables
LAG_1 <- 1
LAG_2 <- 2

lagged_data <- data.frame(years,fa_h,fa_s,nfa_h,nfa_s,
                     fa_h_l1=Lag(fa_h, LAG_1), fa_s_l1 = Lag(fa_s, LAG_1), nfa_h_l1 = Lag(nfa_h, LAG_1), nfa_s_l1 = Lag(nfa_s, LAG_1),
                     fa_h_l2=Lag(fa_h, LAG_2), fa_s_l2 = Lag(fa_s, LAG_2), nfa_h_l2 = Lag(nfa_h, LAG_2), nfa_s_l2 = Lag(nfa_s, LAG_2))

lagged_data <- lagged_data[complete.cases(lagged_data),]
lagged_data <- ts(lagged_data, start = c(1917), end=c(2004), frequency=1)

# create a test/train split
n_instances <- dim(lagged_data)[1]
trainIndex <- (n_instances / 10) * 8

train <- as.data.frame(lagged_data[1:trainIndex-1,])
test <- as.data.frame(lagged_data[trainIndex:n_instances,])


train_errors = list()
test_errors = list()

for (i in (1:4)){
  # what column are we currently working on?
  print(titlenames[i])
  
  # Train split
  train_y <- subset(train, select=c(i +1, i+5, i+9))
  names(train_y) <- c(datanames[i], 'a', 'b')
  train_years <- train[,1]

  # create column formula for NN 
  f = as.formula(paste(datanames[i], " ~ a + b"))
  
  #..then create the NN
  current_nn <- neuralnet(f, data = as.data.frame(train_y), hidden=c(6, 4), threshold = 0.01, stepmax=9000000)
  
  # Calculate NN predictions
  nn_train_predictions <- compute(current_nn, train_y)
  nn_train_results = nn_train_predictions$net.result
  
  # Get Training Error
  nn_train_rmse <- rmse(train[,i+1], nn_train_results)
  train_errors <- c(train_errors, nn_train_rmse)
  
  # Moving Average
  train_ma <- rowMeans(train_y[c(datanames[i], 'a', 'b')], na.rm=TRUE)
  
  # Plot what we got
  plot(train_years, train[,i+1], type='l', col=2, lwd=5.0, main=paste(titlenames[i],"Training Set"), xlab="Time", ylab="Value", ylim = c(-0.5,2))
  lines(train_years, nn_train_results, col=3, lwd = 3.0)
  lines(train_years, train_ma, col=4, lwd=2.0)
  legend("bottomleft", c("Recorded Values", "NN Predicted Values", "Moving Average"), cex=1.0, fill=2:4)
  
  print("Training Complete")
  
  
  # Test split
  test_y <- subset(test, select=c(i +1, i+5, i+9))
  names(test_y) <- c(datanames[i], 'a', 'b')
  test_years <- test[,1]
  
  nn_test_predictions <- compute(current_nn, test_y)
  nn_test_results = nn_test_predictions$net.result
  
  # Test RMSE
  nn_test_rmse <- rmse(test[,i+1], nn_test_results)
  
  # Moving Average
  test_ma <- rowMeans(test_y[c(datanames[i], 'a', 'b')], na.rm=TRUE)
  
  plot(test_years, test[,i+1], type='l', col=2, lwd=5.0, main=paste(titlenames[i],"Test Set"), xlab="Time", ylab="Value", ylim = c(-0.5,2))
  lines(test_years, nn_test_results, col=3, lwd = 3.0)
  lines(test_years, test_ma, col=4, lwd=2.0)
  legend("top", c("Recorded Values", "NN Predicted Values", "Moving Average"), cex=1.0, fill=2:4)

  # Holt Winters
  current_hw <- HoltWinters(train_y[1], beta=FALSE, gamma=FALSE)
  hw_test_results <- predict(current_hw, n.ahead = (length(test_years) + 1), prediction.interval = T, level = 0.95)
  plot(current_hw, hw_test_results, main = paste(titlenames[i], "Holt Winter Forecast"), xlab="Time", ylab="Value")
  lines(test[,1], test_ma, col=4, lwd=2.0)
  
  print("Testing Complete")
  test_errors <- c(test_errors, nn_test_rmse)
}

# Error plot
error_frame <- do.call(rbind, Map(data.frame, A=train_errors, B=test_errors))
colnames(error_frame) <- paste(c("Train", "Test"))
rownames(error_frame) <- paste(datanames)
barplot(t(as.matrix(error_frame)), main="Error of Neural Network By Data Column", 
        xlab = "Death Category", ylab = "NN RMSE", beside=TRUE, legend=colnames(error_frame))

