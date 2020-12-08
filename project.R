library(caret)
library(e1071)
library(rpart)
library(randomForest)

# Datos de Entrenamiento
data.train <- read.csv("https://raw.githubusercontent.com/sergiosilvarubio/ml-vsi/master/data/CE_L_0.01_10_01.csv")
names(data.train) <- c(1:5000) # Cambio de nombres de variables

# Datos de Predicción
data.test <- read.csv("https://raw.githubusercontent.com/sergiosilvarubio/ml-vsi/master/data/CP_L_0.01_10_01.csv")
names(data.test) <- c(1:5000) # Cambio de nombres de variables

# Etiquetas de Entrenamiento
label.train <- data.train[,ncol(data.train)]
label.train <- as.factor(label.train)

# Etiquetas para evaluar Predicción
label.test <- data.test[,ncol(data.test)]
label.test <- as.factor(label.test)

# Preparación de las pruebas
set.seed(42) # Se establece el valor de la semilla para que sea reproducible

# KNN
model <- knn3(x = data.train, y = label.train, k = floor(sqrt(nrow(data.train))))
predicted <- predict(model, data.test, type = "class")
cmat <- confusionMatrix(data=predicted, reference=label.test)
knn_accuracy <- cmat$overall[1]
knn_accuracy

# SVM Radial
model <- svm(label.train ~ ., data=data.train, kernel="radial")
predicted <- predict(model, data.test)
cmat <- confusionMatrix(data=predicted, reference=label.test)
svmr_accuracy <- cmat$overall[1]
svmr_accuracy

# SVM Lineal
model <- svm(label.train ~ ., data=data.train, kernel="linear")
predicted <- predict(model, data.test)
cmat <- confusionMatrix(data=predicted, reference=label.test)
svml_accuracy <- cmat$overall[1]
svml_accuracy

#Naive Bayes
model <- naiveBayes(x = data.train, y = label.train)
predicted <- predict(model, data.test)
cmat <- confusionMatrix(data=predicted, reference=label.test)
nb_accuracy <- cmat$overall[1]
nb_accuracy

#CART
model <- rpart(label.train ~ ., data = data.train)
predicted <- predict(model, data.test, type="class")
cmat <- confusionMatrix(data=predicted, reference=label.test)
cart_accuracy <- cmat$overall[1]
cart_accuracy

#Random Forest
model <- randomForest(x = data.train, y = label.train)
predicted <- predict(model, data.test)
cmat <- confusionMatrix(data=predicted, reference=label.test)
rf_accuracy <- cmat$overall[1]
rf_accuracy
