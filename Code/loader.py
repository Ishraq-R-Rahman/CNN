from mlxtend.data import loadlocal_mnist
import platform
import numpy as np
from utils import initialize_parameters


def read_input():
        convolution_layers = []
        max_pooling_layers = []
        output_dimension = 0
        relu_activation = []

        with open('input.txt' , 'r' ) as f:
                for line in f:
                        row = line.strip().split(' ')
                        if row[0] == 'Conv':
                                convolution_layers.append({
                                        'outputChannels' : int(row[1]),
                                        'filterDimension' : int(row[2]),
                                        'stride' : int(row[3]),
                                        'padding' : int(row[4])
                                })

                                max_pooling_layers.append([])
                                relu_activation.append(False)
                                
                        elif row[0] == 'Pool':
                                max_pooling_layers[-1] =  [ int(i) for i in row[1:] ] 
                        elif row[0] == 'FC':
                                output_dimension =  int(row[1])
                        elif row[0] == 'Softmax':
                                pass
                        elif row[0] == 'ReLU':
                                relu_activation[-1] = True
                                

        return convolution_layers , max_pooling_layers , output_dimension , relu_activation

def load_MNIST_dataset():
        

        if not platform.system() == 'Windows':
                X, y = loadlocal_mnist(
                        images_path='./Data/MNIST/train-images-idx3-ubyte', 
                        labels_path='./Data/MNIST/train-labels-idx1-ubyte')
                X_t , y_t = loadlocal_mnist(
                        images_path='./Data/MNIST/t10k-images-idx3-ubyte', 
                        labels_path='./Data/MNIST/t10k-labels-idx1-ubyte')

        else:
                X, y = loadlocal_mnist(
                        images_path='./Data/MNIST/train-images-idx3-ubyte', 
                        labels_path='./Data/MNIST/train-labels.idx1-ubyte')
                X_t , y_t = loadlocal_mnist(
                        images_path='./Data/MNIST/t10k-images-idx3-ubyte', 
                        labels_path='./Data/MNIST/t10k-labels.idx1-ubyte')

        # print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
        # print('\n1st row', X[0])

        X = X[:50000] # temp
        y = y[:50000] # temp
        # print(y.shape)

        X_train = np.frombuffer(X, dtype=np.uint8).astype(np.float32).reshape(X.shape[0] , X.shape[1])
        y_train = np.frombuffer(y, dtype=np.uint8).astype(np.int64).reshape( X.shape[0] , 1 )

        X_train -= int(np.mean(X_train))
        X_train /= int(np.std(X_train))

        train_data = np.hstack((X_train , y_train))

        np.random.shuffle(train_data)
        

        X_test = np.frombuffer( X_t, dtype=np.uint8).astype(np.float32).reshape( X_t.shape[0] , X.shape[1])
        y_test = np.frombuffer(y_t, dtype=np.uint8).astype(np.int64).reshape( X_t.shape[0] , 1 )

        X_test -= int(np.mean(X_test))
        X_test /= int(np.std(X_test))

        test_data = np.hstack((X_test , y_test))

        return train_data , test_data