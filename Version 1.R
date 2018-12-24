## packages will use
library('dplyr') # data manipulation
library('ggplot2') # Data Visualization
library('ggthemes') # Data Visualization
library('scales') #visualization
library('mice') #visualization
library('randomForest') #classification algorithm
library('LogicReg') #classification algorithm
library('stringr')
library('e1071')
wq
## load files into RStudio
setwd('C:\\Users\\xhao1\\Desktop\\R')
train <- read.csv(file = "train.csv",stringsAsFactors = F)
test  <- read.csv(file = "test.csv", stringsAsFactors = F)
full  <- bind_rows(train, test)

## Description for the Data
print(head(full))
str(full)

## filling missing values
# Age Filling missing place and Density
age <- full$Age
n = length(age)
set.seed(110) # Replace the missing value with a random number
for(i in 1:n)
  {
    if(is.na(age[i]))
      {
        age[i] = sample(na.omit(full$Age),1)
      }
}
par(mfrow=c(1,2))
hist(age, freq=F, main='Age Density', col='darkgreen', ylim=c(0,0.04))

# Fare filling missing place
full$Fare[1044] <- median(full[full$Embarked == 'S' & full$Pclass == '3' , ]$Fare, na.rm = TRUE)

## Easy Diagram
# FamilySize Influence for Survival Rate
full$Fullsize <- full$SibSp + full$Parch + 1; # The Fullsize for Every team
full$Fullsized[full$Fullsize == 1] <- 'singleton'
full$Fullsized[full$Fullsize < 6 & full$Fullsize > 1] <- 'small'
full$Fullsized[full$Fullsize > 5] <- 'large'
png(filename="FamilySize&Survival.png")
graph1=mosaicplot(table(full$Fullsized, full$Survived), main='Family Size vs Survival', shade=TRUE)
print(graph1)
dev.off()

#Age Influence for Survival Rates
StoredData <- data.frame(Age = age[1:891], Survived = train$Survived)
png(filename="Age&Survival.png")
graph2=ggplot(StoredData, aes(Age,fill = factor(Survived))) +
       geom_histogram()
print(graph2)
dev.off()

#Embarked, Pclass & Fare relation
embark_new <- full %>%
  filter(PassengerId != 62 & PassengerId != 830)
png(filename="Embarked,Pclass&Fare.png")
graph3 = ggplot(embark_new, aes(x = Embarked, y = Fare, fill = factor(Pclass))) + geom_boxplot() + 
            geom_hline(aes(yintercept=60), colour='green', linetype='dashed', lwd=2) + scale_y_continuous(labels=dollar_format()) + theme_few()
print(graph3)
dev.off()

#Sex and Survival Relationship
png(filename="Sex&Survival.png")
graph4=ggplot(train, aes(Sex,fill = factor(Survived))) +
          geom_histogram(stat = "count")
print(graph4)
dev.off()

#Sex or, Age and Survival Relationship
png(filename = "Pclass,Age&Sruvival.png")
graph5 = ggplot(full[1:891,], aes(Age, fill = factor(Survived))) + 
            geom_histogram() + 
            facet_grid(.~Pclass) + 
            theme_few()
print(graph5)
dev.off()

#Age, Embarked and Survival Relationship
full$Embarked[c(62, 830)] <- 'C'
png(filename = "Embarked,Age&Sruvival.png")
graph6 = ggplot(full[1:891,], aes(Age, fill = factor(Survived))) + 
            geom_histogram() + 
            facet_grid(.~Embarked) + 
            theme_few()
print(graph6)
dev.off()


#Sex related with Survival Rate
print(tapply(train$Survived,train$Sex,mean))

#Name and Survival Relationship
Name<- full$Name
Data<-str_extract(string = Name, pattern = "(Mr|Mrs)\\.")
Data[is.na(Data)]<- "OtherName"
d <- data.frame(Data = Data[1:891],Survived = train$Survived)
png(filename = "Name&Survival.png")
graph8=ggplot(d, aes(Data,fill = factor(Survived))) + geom_histogram(stat = "count")
print(graph8)
print(tapply(d$Survived,d$Data,mean))
dev.off()

#Pclass and survival Relationship
png(filename = "Pclass&Survival.png")
graph9 = ggplot(train, aes(Pclass,fill = factor(Survived))) + geom_histogram(stat = "count")
print(graph9)
dev.off()

## Prediction
#Create Data for Everything We Had
forecast_train.survived = train$Survived
forecast_train.age = age[1:891]
forecast_test.age = age[892:1309]
forecast_train.fare = full$Fare[1:891]
forecast_test.fare = full$Fare[892:1309]
forecast_train.family = full$Fullsize[1:891]
forecast_test.family = full$Fullsize[892:1309]
forecast_train.name = Data[1:891]
forecast_test.name = Data[892:1309]
forecast_train.pclass = train$Pclass
forecast_test.pclass = test$Pclass
forecast_train.sex = train$Sex
forecast_test.sex = test$Sex
forecast_train.embarked = full$Embarked[1:891]
forecast_test.embarked = full$Embarked[892:1309]

#Enter Data into Logic Regression Model
n_train = data.frame(survived = forecast_train.survived , age = forecast_train.age, fare = forecast_train.fare, sex = forecast_train.sex, embarked = forecast_train.embarked,
                     family = forecast_train.family, name = forecast_train.name, pclass = forecast_train.pclass)
LG_model<-glm(factor(survived) ~ age + fare + sex + embarked + family + 
                name + pclass, data = n_train, family = binomial)
#Enter Data into Test and Prediction
n_test <- data.frame(age = forecast_test.age, fare = forecast_test.fare, sex = forecast_test.sex, embarked = forecast_test.embarked, 
                            family = forecast_test.family, name = forecast_test.name, pclass = forecast_test.pclass)
ans_LG_predict <- ifelse(predict(LG_model, n_test, type="response")>0.5,1,0)
solution <- data.frame(PassengerID = test$PassengerId, Survived = ans_LG_predict)
write.csv(solution, file = 'LG_solution.csv', row.names = F)



