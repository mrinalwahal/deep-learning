
#----------------------------------------------#
# Project   : Airline Passenger Predictions    #
# Model     : LSTM in Recurrent Neural Network #
# Architect : Mrinal Wahal                     #
#----------------------------------------------#

# Import all the required dependencies
import pandas
import sys
import math
import matplotlib.pyplot as plot
import numpy
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Design original data and look back data
def create_data(dataset, look_back):

    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i+look_back),0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

def main():

    for argument in sys.argv:
        if argument == "c": column_to_use = int(sys.argv[sys.argv.index(argument) + 1])
        elif argument == "w": weights = sys.argv[sys.argv.index(argument) + 1]
        elif argument == "e": epochs = int(sys.argv[sys.argv.index(argument) + 1])
        elif argument == "train": dataframe = sys.argv[sys.argv.index(argument) + 1]

    print "\n[+] Reading the Data Frame: ", dataframe
    print "[+] Epochs: ", epochs
    print "[+] Column to use: ", column_to_use
    print "[+] Weights to load: ", None

    dataframe = pandas.read_csv(dataframe, usecols = [column_to_use], engine = 'python', skipfooter=3)
    dataset = dataframe.values
    dataset = dataset.astype("float32")

    # Normalize the dataset
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)

    # Split the training and testing datasets
    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    train = dataset[0:train_size,:]
    test = dataset[train_size:len(dataset),:]
    print ("\n[+] Length of Training DataSet = %d" % len(train))
    print ("[+] Length of Testing DataSet = %d" % len(test))

    # Train the datasets with predefined function.
    # And reshape to be (x = t) and (y = t+1)
    look_back = 1
    trainX, trainY = create_data(train, look_back)
    testX, testY = create_data(test, look_back)

    # LSTM requires the input dataset to be in the form of [samples, time steps, features]
    # Currently the input dataset is in the form of [samples, features]
    # Now we will reshape the input data

    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # Design the model of the LSTM Netowrk
    model = Sequential()

    # Add the layers --> 1 input layer, 4 hidden LSTM layers and single Prediction output
    model.add(LSTM(4, input_shape=(1,look_back)))
    model.add(Dense(1))

    # Compile the model and fit the training data
    model.compile(loss = "mean_squared_error", optimizer = "adamax")
    model.fit(trainX,trainY,epochs=epochs, batch_size=1, verbose=2)

    # Now we will make predictions and test the data in order to get point of comparisons
    train_predict = model.predict(trainX)
    test_predict = model.predict(testX)

    # Now invert the predictions before calculating the errors in order to get the output
    # in the same form as the original data i.e. thousands of passengers per month

    train_predict = scaler.inverse_transform(train_predict)
    trainY = scaler.inverse_transform([trainY])
    test_predict = scaler.inverse_transform(test_predict)
    testY = scaler.inverse_transform([testY])

    # Calculate mean_squared_error

    train_score = math.sqrt(mean_squared_error(trainY[0], train_predict[:,0]))
    print ("\n[+] Train Score %.2f RMSE." % train_score)

    test_score = math.sqrt(mean_squared_error(testY[0], test_predict[:,0]))
    print ("[+] Test Score %.2f RMSE." % test_score)

    # Now we need to make sure the predictions align on the same x-axis as the original data

    # Shift the Training Predictions for Plotting

    train_predict_plot = numpy.empty_like(dataset)
    train_predict_plot[:,:] = numpy.nan
    train_predict_plot[look_back:len(train_predict)+look_back, :] = train_predict

    # Shift the Testing Predictions for Plotting

    test_predict_plot = numpy.empty_like(dataset)
    test_predict_plot[:,:] = numpy.nan
    test_predict_plot[len(train_predict)+(look_back*2)+1: len(dataset) - 1, :] = test_predict

    # Finally plot the baseline + predictions

    plot.plot(scaler.inverse_transform(dataset))
    plot.plot(train_predict_plot, "r-")
    plot.plot(test_predict_plot, "black")
    plot.show()

if __name__ == "__main__": main()
