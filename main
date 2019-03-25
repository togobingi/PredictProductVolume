install.packages("corrplot")
install.packages("caret")
install.packages("dplyr")

library("corrplot")
library("caret")
library(dplyr)


#Import Data sets for analysis
Exisiting_Products <- read_csv("existingproductattributes2017.csv")
New_Products <- read_csv("newproductattributes2017.csv")


#Dummify Data (Change categorical data to numeric data)
ExProd_DataFrame <- dummyVars("~.", data=Exisiting_Products)
NewProd_dataFrame <- dummyVars("~.", data=New_Products)

readyData <- data.frame(predict(ExProd_DataFrame, newdata=Exisiting_Products))
newReadyData <- data.frame(predict(NewProd_dataFrame, newdata=New_Products))


#Replace NAs with average of BestSellers
readyData$BestSellersRank = ifelse(is.na(readyData$BestSellersRank),
                                ave(readyData$BestSellersRank,
                                FUN = function(x) mean(x, na.rm = TRUE)),
                                readyData$BestSellersRank)

#Correlation Matrix view
CorrMatrix <- cor(readyData)
CorrMatrix #Print Correlation matrix
corrplot(CorrMatrix)


#Filter columns in ExistingProducts dataset for feature selection
myReadyData <- select(readyData, ProductTypePC, 
                      ProductTypeSmartphone, 
                      ProductTypeLaptop,
                      ProductTypeNetbook,
                      x4StarReviews,
                      x3StarReviews,
                      x1StarReviews,
                      x2StarReviews,
                      PositiveServiceReview,
                      NegativeServiceReview,
                      Volume)

myReadyData <- subset(myReadyData, Volume<7036)


#Filter columns in NewProducts dataset for feature selection
myNewReadyData <- select(newReadyData, ProductTypePC, 
                         ProductTypeSmartphone, 
                         ProductTypeLaptop,
                         ProductTypeNetbook,
                         x4StarReviews,
                         x3StarReviews,
                         x1StarReviews,
                         x2StarReviews,
                         PositiveServiceReview,
                         NegativeServiceReview,
                         Volume)

#### Build Linear Regression Model ####
TrainData <-  createDataPartition(y=myReadyData$Volume, p =.80, list= FALSE) #Split set

#Create Training Set
trainingSet <- myReadyData[TrainData,]

#Create Test Set
TestingSet <- myReadyData[-TrainData,]

#Linear_Model code
set.seed(123)

#Linear_Model training
tree_mod_lm <- train(Volume~ 
                     ProductTypePC 
                     +ProductTypeSmartphone
                     +ProductTypeLaptop
                     +ProductTypeNetbook
                     +x4StarReviews
                     +x3StarReviews
                     +x1StarReviews
                     +x2StarReviews
                     +PositiveServiceReview
                     +NegativeServiceReview,
                     data = trainingSet, 
                     method="lm", 
                     tuneLength = 1,
                     trControl=fitControl)

#RF_Model training
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)
rfGrid = expand.grid(mtry = c(1:10))

tree_mod <- train(Volume~., data = trainingSet, 
                  method="rf", 
                  trControl=fitControl, 
                  tuneLength = 2,
                  tuneGrid=rfGrid)

#SVM_Model training
tree_mod_svm <- train(Volume~., data = trainingSet, 
                  method="svmLinear", 
                  trControl=fitControl, 
                  tuneLength = 2)

#Second SVM Model
x <- subset(myReadyData, select = -Volume)
y <- myReadyData$Volume
svm_model <- svm(x,y, cost=.1, kernel="linear", scale = FALSE)

pred_svm <- predict(svm_model, x)
pred_svm

#KNN Model training
tree_mod_knn <- train(Volume~., data = trainingSet, 
                      method="knn", 
                      preProcess = c("center","scale"),
                      trControl=fitControl, 
                      tuneLength = 2)

print(svm_model)
summary(svm_model)

#Predict with Training set to guage performance
TestingSet$predicted.volume = predict(tree_mod, TestingSet)

postResample(TestingSet$predicted.volume, TestingSet$Volume )

#Predict with new unseen dataset (Actual prediction)
myNewReadyData$predicted.volume = predict(tree_mod, myNewReadyData)
finalPred = predict(tree_mod, myNewReadyData)

#Add predictions to the new products data set 
output <- New_Products
output$prediction <- finalPred

#Create a csv file and write it to your hard drive
write.csv(output, file="Task3_output.csv", row.names = TRUE)

#Save Models in variable to plot comparison graph
results <- resamples(list("SVM"=tree_mod_svm,
                          "Random Forest"=tree_mod,
                          "KNN"=tree_mod_knn,
                          "Linear Model"= tree_mod_lm))

dotplot(results, metric="Rsquared")


