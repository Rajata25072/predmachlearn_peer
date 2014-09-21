##Executive Summary
The project is aiming to predict whether the participants did the exercise (barbell lifts) correctly using data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants rom [Groupware@LES](http://groupware.les.inf.puc-rio.br/har). This project result could be expand later on with the implication of recent "technology wearable" such as Jawbone Up, Nike FuelBand, and Fitbit which make it possible to collect a large amount of data about personal activity in an inexpensive manner.  The research methodology and findings are as followed:  
**1. The data is readed and cleaned** by screening out variables with NA in *Submission* data set.  
**2. The data used is splitted for cross-validation with random subsampling and expected out of sample error of 0.2** to training and testing set with the proportion of 60:40  
**3. The variables are further screened out with these criteria:**  
3.1 The variables with low sd/mean, less than 2, are screened out.  
3.2 The variables with with high correlation with other predictor variables, more than 0.8, are also filtered out.  
**4. Finding the model with highest accuracy** among selected 5 models, namely, *Linear Discriminant Analysis (LDA)*, *Quadratic Discriminant Analysis (QDA)*, *Decision Tree*, *Random Forest*, and *Bagging*. The models chosen are **Random Forest** and **Bagging** due to their high accuracy (0.97 and 0.95 respectively).  
**5. Predicting the 20 different test cases using the model built.** After testing with the answer, the results are 100% correct for both model.

______________________________________________________________________________________________________________________________________________________

##Main Report


###1. Loading Required Packages
In order to make the research reproducible, the required packages are loaded. 

```r
library(caret); library(MASS); library(klaR); library(rpart) ; library(randomForest); library(e1071); library(ipred); library(plyr)
```

###2. Reading Data  
Reading required both training data for model building (*Activity*) and submission data for prediction submission (*Submission*). (Assuming that the data is already stored in directory) 

```r
Activity <- read.csv("pml-training.csv"); Submission <- read.csv("pml-testing.csv")
```

###3. Screening Variables#1  
The data is screened by screening out the variables with NA value in Submission data set in both Activity and Submission dataset.

```r
submission <- Submission[,colSums(is.na(Submission)) < nrow(Submission)]; selectName <- names(submission)
activity <- Activity[,colnames(Activity) %in% c(selectName,"classe")]
```
*The variable of interests are 52 predictors which are 13 data (roll, pitch, yaw, total_accel, gyros(x-z), accel(x-z) and magnet(x-z)) collected from 4 sensors (belt, arm, dumbbell and forearm).

###4. Data Cross Validation
The activity data is split for cross validation using **random method** with 60:40 proportion into *training dataset* and *testing dataset*.

```r
library(caret); library(lattice); library(ggplot2); set.seed(30)
inTrain <- createDataPartition(y=activity$classe, p=0.6, list=FALSE)
training <- activity[inTrain,]; testing <- activity[-inTrain,]
```

###5. Screening Variables#2  
The variables in training dataset is screen out with 2 criteria, low sd/mean and high correlation with other predictors.  
  
**5.1 Screen Out Variables with Low sd/mean**  
The numeric predictors with sd/mean value lower than 2 will be screened out.

```r
sd_mean <- abs(sapply(training[,8:59], sd) / colMeans(training[,8:59])) #variable #60 is classe, outcome variable
selectName2 <- names(training[,8:59])[sd_mean < 2]
training <- training[,colnames(training) %in% c(selectName2,"classe")]
```

**5.2 Screen Out Variables with High Correlation**  
The numeric predictors with correlation with other predictors higher than 0.7 will be screened out. The results shown that there're high correlation among this variable group.  
**1) roll_belt** with *total_accel_belt(2)*, *accel_belt_y(5)* and *accel_belt_z(6)*  
**2) magnet_belt_y** with *magnet_belt_z(9)*  
**3) accel_arm_z** with *magnet_arm_y(13)* and *magnet_arm_z(14)*  
**4) total_accel_dumbbell** with *accel_dumbbell_y(16)*  
**5) magnet_dumbbell_x** with *magnet_dumbbell_y(18)*  
**6) accel_forearm_y** with *magnet_forearm_y(22)*  

```r
abs(as.matrix(cor(training[,-24]))) > 0.7 #variable #24 is classe, outcome variable
training <- training[,-c(2, 5, 6, 9, 13, 14, 16, 18, 22)] #Filter out predictor with with high correlation (> 0.8)
```

###6. Choosing The Right Model
The selected 5 models will be compared by its accuracy. The results are as followed:

**6.1 Linear Discriminant Analysis (LDA)** - Accuracy = 0.403  

```r
modlda = train(classe~., data=training, method="lda")
plda = predict(modlda, testing); confusionMatrix(plda, testing$classe) #Accuracy = 0.403
```

**6.2 Quadratic Discriminant Analysis (QDA)** - Accuracy = 0.553  

```r
modqda = train(classe~., data=training, method="qda")
pqda = predict(modqda, testing); confusionMatrix(pqda, testing$classe) #Accuracy = 0.553  
```

**6.3 Decision Tree** - Accuracy = 0.42  

```r
modtree = train(classe~., data=training, method="rpart")
ptree = predict(modtree, testing); confusionMatrix(ptree, testing$classe) #Accuracy = 0.42  
```

**6.4 Random Forest**  - Accuracy = 0.9736

```r
modrf = train(classe~., data=training, method="rf")
prf = predict(modrf, testing); confusionMatrix(prf, testing$classe) #Accuracy = 0.9736
```

**6.5 Bagging** - Accuracy = 0.9527

```r
modbag = train(classe~., data=training, method="treebag")
pbag = predict(modbag, testing); confusionMatrix(pbag, testing$classe) #Accuracy = 0.9527
```

The result has shown that *Random Forest* and *Bagging* have highest accuracy (higher than 0.8 expected out-of-sample accuracy) therefore **both model are chosen and compared**. The comparing table also shown that the prediction result are very close.

```r
table(prf, pbag)
```

```
##    pbag
## prf    A    B    C    D    E
##   A 2206   19   23   14    3
##   B   17 1410   36    7    7
##   C   21   23 1342   21    1
##   D   25   16   15 1199    3
##   E    6   17    2    4 1409
```


###7. Predicting The Test Cases
The comparing table of prediction by random forest and bagging show identical results with 100% accuracy after comparing with the answer therefore **both model can be used interchangeably**.

```r
library(randomForest); library(ipred); library(plyr)
rfSubmit = predict(modrf, Submission); bagSubmit = predict(modbag, Submission)
table(rfSubmit, bagSubmit)
```

```
##         bagSubmit
## rfSubmit A B C D E
##        A 7 0 0 0 0
##        B 0 8 0 0 0
##        C 0 0 1 0 0
##        D 0 0 0 1 0
##        E 0 0 0 0 3
```
