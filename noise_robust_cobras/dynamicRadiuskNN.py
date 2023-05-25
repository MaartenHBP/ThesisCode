import pandas as pd
 
import numpy as np
 
from sklearn.model_selection import train_test_split
 
from scipy.stats import mode
 
from sklearn.neighbors import RadiusNeighborsClassifier
 
# Radius Nearest Neighbors Classification
 
class Radius_Nearest_Neighbors_Classifier() :
     
    def __init__( self, r, weights, data) :
         
        self.r = r 
        self.w = weights
        self.data = np.copy(data)
        X_mean = self.data.mean(axis=0) # centreren voor betere nauwkeurigheid
            # The copy was already done above
        self.data -= X_mean
         
    # Function to store training set
         
    def fit( self, X_train, Y_train ) : # gaan indices meegeven
         
        self.X_train = X_train
         
        self.Y_train = Y_train

        self.r = np.array([np.where(X_train == x)[0][0] for x in self.r])
         
     
    # Function for prediction
         
    def predict( self, X_test ) :
         
        self.X_test = X_test
        
         
        # initialize Y_predict
         
        Y_predict = np.zeros( len(X_test))
         
        for i in range( len(X_test) ) :
             
            x = self.X_test[i]
             
            # find the number of neighbors within a fixed
            # radius r of current training example
             
            neighbors = self.find_neighbors( x , self.r[i])
             
            # most frequent class in the circle drawn by current
            # training example of fixed radius r
             
            Y_predict[i] = mode( neighbors )[0][0]
             
        return Y_predict
     
    # Function to find the number of neighbors within a fixed radius
    # r of current training example
           
    def find_neighbors( self, x , r) :
         
        # list to store training examples which will fall in the circle

        distances = np.linalg.norm(self.data[self.X_train] - self.data[x], axis=1) # over de gehele dataset
        selection = np.less_equal(distances, distances[r])
                 
        return self.Y_train[selection]
     
    # Function to calculate euclidean distance
             
    def euclidean( self, x, x_train ) :
         
        return np.linalg.norm(  x - x_train )