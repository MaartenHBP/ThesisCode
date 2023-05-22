import pandas as pd
 
import numpy as np
 
from sklearn.model_selection import train_test_split
 
from scipy.stats import mode
 
from sklearn.neighbors import RadiusNeighborsClassifier
 
# Radius Nearest Neighbors Classification
 
class Radius_Nearest_Neighbors_Classifier() :
     
    def __init__( self, r, weights) :
         
        self.r = r # gaat een array zijn
        self.w = weights
         
    # Function to store training set
         
    def fit( self, X_train, Y_train ) :
         
        self.X_train = X_train
         
        self.Y_train = Y_train
         
        # no_of_training_examples, no_of_features
         
        self.m, self.n = X_train.shape
     
    # Function for prediction
         
    def predict( self, X_test ) :
         
        self.X_test = X_test
         
        # no_of_test_examples, no_of_features
         
        self.m_test, self.n = X_test.shape
         
        # initialize Y_predict
         
        Y_predict = np.zeros( self.m_test )
         
        for i in range( self.m_test ) :
             
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

        distances = np.linalg.norm(self.X_train - x, axis=1)

                 
        return self.Y_train[np.less_equal(distances, r + 1e-15)]
     
    # Function to calculate euclidean distance
             
    def euclidean( self, x, x_train ) :
         
        return np.linalg.norm(  x - x_train )