import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Read training data from file
trainDF = pd.read_csv('TrainingData.txt', header=None)
y = trainDF[24].tolist()
trainDF = trainDF.drop(24, axis=1)
x = trainDF.values.tolist()

# Storing full training data before splitting
x = np.array(x)
y = np.array(y)
xTrainFull = x
yTrainFull = y

# Reading testing data to predict
test_DF = pd.read_csv('TestingData.txt', header=None)
xClassify = test_DF.values.tolist()

# Splitting training data for testing algorithm
xTrain, xTest, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Normalise by scaling between 0 and 1
scaler = MinMaxScaler()
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.transform(xTest)
xClassify = scaler.transform(xClassify)
xTrainFull = scaler.transform(xTrainFull)

# Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis()
lda.fit(xTrain, y_train)
yPredict = lda.predict(xClassify)
yPredict = [int(x) for x in yPredict]

# Testing and training scores
print("\nAccuracy for LDA classifier on full Training Dataset:", (lda.score(xTrainFull, yTrainFull)) * 100, "%")

# Printing results to output file
predict_DF = pd.DataFrame({'Prediction': yPredict})
test_DF = test_DF.join(predict_DF)
test_DF.to_csv("TestingResults.txt", header=None, index=None)
print("\nPredictions in output file TestingResults.txt")

# Find and print number of days with abnormal values
dayCount = 0
tracker = []
inputFile = open("TestingResults.txt", 'r')
while dayCount != 100:
    line = inputFile.readline()
    if int(line.split(",")[24]) == 1:
        tracker.append(dayCount + 1)
    dayCount += 1

print("Predicted count of abnormal scheduling values: " + str(len(tracker)), "\nOn days: " + str(tracker), sep="\n")
