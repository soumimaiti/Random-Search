import csv
import numpy as np


def get_dorothea():
    #Dorothea
    x_train = np.load('dorothea_data/train_array.npy')
    y_train = np.load('dorothea_data/train_labels.npy')
    
    x_test = np.load('dorothea_data/test_array.npy')
    y_test = np.load('dorothea_data/test_labels.npy')
    
    print('Training data:', x_train.shape, y_train.shape)
    print('Testing data:', x_test.shape, y_test.shape)

    return x_train, y_train, x_test, y_test


def get_dexter():
    #Dexter
    x_train = np.load('dorothea_data/dexter_train.npy')
    y_train = np.loadtxt('dorothea_data/dexter_train.labels')
    y_train[y_train == -1] = 0
    x_test = np.load('dorothea_data/dexter_valid.npy')
    y_test = np.loadtxt('dorothea_data/dexter_valid.labels')
    y_test[y_test == -1] = 0
    print('Training data:', x_train.shape, y_train.shape)
    print('Testing data:', x_test.shape, y_test.shape)

    return x_train, y_train, x_test, y_test

def get_arcene():
    x_train = np.loadtxt('dorothea_data/arcene_train.data')
    y_train = np.loadtxt('dorothea_data/arcene_train.labels') 
    y_train[y_train == -1] = 0

    x_test = np.loadtxt('dorothea_data/arcene_valid.data')
    y_test = np.loadtxt('dorothea_data/arcene_valid.labels')
    y_test[y_test == -1] = 0
    #x_test = np.load('dorothea_data/test_array.npy')
    #y_test = np.load('dorothea_data/test_labels.npy')
    print('Training data:', x_train.shape, y_train.shape)
    print('Testing data:', x_test.shape, y_test.shape)
    
    return x_train, y_train, x_test, y_test

