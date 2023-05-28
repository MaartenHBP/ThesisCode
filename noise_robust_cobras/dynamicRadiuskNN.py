import pandas as pd
 
import numpy as np
 
from sklearn.model_selection import train_test_split
 
from scipy.stats import mode
 
from sklearn.neighbors import RadiusNeighborsClassifier
from collections import Counter
 
# Radius Nearest Neighbors Classification
 
class Radius_Nearest_Neighbors_Classifier() :
     
    def __init__( self, r, weights, data) :
         
        self.r = r 
        self.w = True if weights == 'distance' else False
        self.data = np.copy(data)
        X_mean = self.data.mean(axis=0) # centreren voor betere nauwkeurigheid
            # The copy was already done above
        self.data -= X_mean
        self.closestPerPoint = None
         
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
        self.closestPerPoint = np.zeros( len(X_test))
         
        for i in range( len(X_test) ) :
             
            x = self.X_test[i]
             
            # find the number of neighbors within a fixed
            # radius r of current training example
             
            neighbors, distances, selection = self.find_neighbors( x , self.r[i])

            if self.w:
                label_weights = Counter()  # Counter to keep track of label weights

                for label, distance in zip(neighbors[selection], distances[selection]):
                    label_weights[label] += 1 / distance  # Increment label weight based on distance
                Y_predict[i] = max(label_weights, key=label_weights.get)
            else:
                Y_predict[i] = mode( neighbors[selection] )[0][0]

            # Get the label with the maximum weight
            
            distances[np.invert(neighbors == Y_predict[i])] = np.inf
            self.closestPerPoint[i] = np.argmin(distances)
             
        return Y_predict
     
    # Function to find the number of neighbors within a fixed radius
    # r of current training example
           
    def find_neighbors( self, x , r) :
         
        # list to store training examples which will fall in the circle

        distances = np.linalg.norm(self.data[self.X_train] - self.data[x], axis=1) # over de gehele dataset
        selection = np.less_equal(distances, distances[r])
                 
        return self.Y_train, distances, selection
     
    # Function to calculate euclidean distance
             
    def euclidean( self, x, x_train ) :
         
        return np.linalg.norm(  x - x_train )