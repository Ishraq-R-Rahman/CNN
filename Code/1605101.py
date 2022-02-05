from loader import load_MNIST_dataset , read_input
from utils import create_filters , gradient_descent, get_strides_list
import math
import numpy as np


def train_model( train_data , learning_rate , MU , filters , biases_list , theta , bias , relu_activation , max_pooling_layers ,stride,  epochs = 2 , batch_size = 2 ):

        cost = []
        acc = []

        for epoch in range(epochs):
                batches = [ train_data[ k:k + batch_size] for k in list(range( 0, train_data.shape[0] , batch_size ) ) ]

                x = 0

                for batch in batches:
                        gradient_descent( batch=batch , learning_rate=learning_rate , MU=MU , filters=filters , biases_list=biases_list , theta=theta , bias=bias , cost=cost , acc=acc ,stride=stride, img_width=int(math.sqrt(train_data.shape[1]-1)) , img_depth=1 , relu_activation=relu_activation , max_pooling_layers = max_pooling_layers)
                        pass
        pass

if __name__ == '__main__':
        training_data , test_data = load_MNIST_dataset()
        convolution_layers , max_pooling_layers , output_dimension , relu_activation = read_input()
        filters , theta , initial_bias , biases = create_filters(convolution_layers , math.sqrt(training_data.shape[1] - 1) , output_dimension )
        
        stride = get_strides_list( convolution_layers )

        learning_rate = 0.01
        batch_size = 2

        train_model( train_data= training_data , learning_rate = learning_rate , MU=0.95 , filters = filters , biases_list= biases , theta= theta , bias=initial_bias , relu_activation = relu_activation , max_pooling_layers = max_pooling_layers , stride = stride)
        pass