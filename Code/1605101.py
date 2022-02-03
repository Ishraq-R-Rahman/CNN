import math
from mlxtend.data import loadlocal_mnist
import platform

def load_MNIST_dataset():
    if not platform.system() == 'Windows':
        X, y = loadlocal_mnist(
                images_path='../Data/MNIST/train-images-idx3-ubyte', 
                labels_path='../Data/MNIST/train-labels-idx1-ubyte')
        X_t , y_t = loadlocal_mnist(
                images_path='../Data/MNIST/t10k-images-idx3-ubyte', 
                labels_path='../Data/MNIST/t10k-labels-idx1-ubyte')

    else:
        X, y = loadlocal_mnist(
                images_path='../Data/MNIST/train-images.idx3-ubyte', 
                labels_path='../Data/MNIST/train-labels.idx1-ubyte')
        X_t , y_t = loadlocal_mnist(
                images_path='../Data/MNIST/t10k-images-idx3-ubyte', 
                labels_path='../Data/MNIST/t10k-labels.idx1-ubyte')
    
    # print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
    # print('\n1st row', X[0])

    X_train = X.reshape( (X.shape[0] , int(math.sqrt(X.shape[1])) , int(math.sqrt(X.shape[1])) , 1 ) )
    y_train = y

    X_test = X_t.reshape( (X_t.shape[0] , int(math.sqrt(X_t.shape[1])) , int(math.sqrt(X_t.shape[1])) , 1 ) )
    y_test = y

    # #checking the shape after reshaping
    # print(X_train.shape)
    # print(X_test.shape)

    return (X_train , y_train ) , (X_test , y_test)
    


if __name__ == '__main__':
    training_data , test_data = load_MNIST_dataset()
    print(test_data[0].shape)
    pass