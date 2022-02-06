import math
import numpy as np
from mlxtend.data import loadlocal_mnist


learning_rate = 0.001

def padding( input , pad , remove = False):

    if pad == 0:
        output = input
    elif remove:
        output = input[:, pad : -pad, pad: -pad, : ]
    else:
        output = np.pad(
            input , 
            ( (0,0) , ( pad , pad ) , ( pad , pad ) , (0,0) )
        )

    return output


class CNN:

    def __init__(self , output_channels , filter_dimension , stride , padding , input_channels ) -> None:
        self.output_channels = output_channels
        self.filter_dimension = filter_dimension
        self.stride = stride
        self.padding = padding
        self.input_channels = input_channels

        self.bias = np.random.normal( loc=0.0, scale=1.0, size = output_channels )
        
        self.weights = np.random.rand(
            self.output_channels , self.filter_dimension , self.filter_dimension , self.input_channels
        ) / ( self.input_channels * self.filter_dimension * self.filter_dimension )

    
    def feed_forward(self , input ):
        self.input = padding(input , self.padding , False)
        
        self.output = np.zeros(
            (self.input.shape[0] , self.input.shape[1] - self.filter_dimension + 1 , self.input.shape[2] - self.filter_dimension + 1 , self.output_channels)
        )

        [ loop1 , loop2 , loop3 , loop4 ] = self.output.shape

        for i in range( 0 , loop1):

            for j in range( 0 , loop2 ):

                for x in range( 0 , loop3 ):

                    for y in range( 0 , loop4 ):
                        itr = j * self.stride
                        itr1 = x * self.stride

                        mul = np.multiply(
                            self.input[i , itr : itr + self.filter_dimension , itr1 : itr1 + self.filter_dimension , : ] ,
                            self.weights[ y , : , : , : ]
                        )
                        
                        self.output[ i , j , x , y ] = np.sum(mul) + self.bias[y]
        
        return self.output

    
    def backward_propagate( self , old_delta ):

        delta_weights = np.zeros( self.weights.shape )
        delta_bias = np.zeros( self.bias.shape )

        new_delta = np.zeros(self.input.shape)

        [ loop1 , loop2 , loop3 , loop4 ] = old_delta.shape

        for i in range( 0 , loop1):

            for j in range( 0 , loop2 ):

                for x in range( 0 , loop3 ):

                    for y in range( 0 , loop4 ):
                        itr = j * self.stride
                        itr1 = x * self.stride

                        mul = np.multiply(
                            old_delta[ i ][ j ] [ x ][ y ] , self.input[i , itr : itr + self.filter_dimension , itr1 : itr1+self.filter_dimension , : ]
                        )

                        delta_weights[ y , : , : , : ] += mul

                        delta_bias[ y ] += old_delta[ i ][ j ] [ x ][ y ]

                        mul = np.multiply(
                            old_delta[ i ][ j ] [ x ][ y ] , self.weights[y,:,:,:]
                        )

                        new_delta[ i , itr : itr + self.filter_dimension , itr1: itr1+self.filter_dimension ,:] += mul
        
        new_delta = padding( new_delta , self.padding , True )

        self.weights -= learning_rate * delta_weights
        self.bias -= learning_rate * delta_bias

        
        return new_delta


class ExternalLayers():
    def __init__(self) -> None:
        self.output = None
        self.input = None
    
    def feed_forward(self , input ):
        self.input = input
    
    def backward_propagate( self , gradient ):
        pass


class MaxPool(ExternalLayers):

    def __init__(self , filter_dimension , stride ) -> None:
        super().__init__()
        self.filter_dimension = filter_dimension 
        self.stride = stride 
    
    def feed_forward(self, input):
        super().feed_forward(input)

        [loop1 , loop2 , loop3 , loop4 ] = self.input.shape
        loop2 -= self.filter_dimension
        loop3 -= self.filter_dimension

        self.output = np.zeros(
            ( 
                loop1 , 
                int(loop2 / self.stride ) + 1 , 
                int(loop3 / self.stride ) + 1 , 
                loop4
            )
        )

        for i in range( 0 , loop1):

            for j in range( 0 , loop2 + 1):

                for x in range( 0 , loop3  + 1):

                    for y in range( 0 , loop4 ):

                        self.output[ i , int( j / self.stride ) , int( x / self.stride ) , y ] = np.max( self.input[ i , j: j + self.filter_dimension , x: x+self.filter_dimension , y])


        return self.output


    def backward_propagate(self, gradient):
        super().backward_propagate(gradient)

        result = np.zeros( self.input.shape )

        [loop1 , loop2 , loop3 , loop4 ] = self.output.shape

        for i in range( 0 , loop1):

            for j in range( 0 , loop2):

                for x in range( 0 , loop3):

                    for y in range( 0 , loop4):

                        itr = j * self.stride
                        itr1 = x * self.stride
                        mul = np.multiply(
                            gradient[i][j][x][y],
                            ( self.output[i,j,x,y] == self.input[i,itr:itr+self.filter_dimension,itr1:itr1+self.filter_dimension,y])
                        )
                        result[ i , itr : itr + self.filter_dimension , itr1:itr1+self.filter_dimension , y] = mul
        return result
        


class FlatteningLayer(ExternalLayers):
    def __init__(self) -> None:
        super().__init__()
    
    def feed_forward(self, input):
        super().feed_forward(input)
        [loop1 , loop2 , loop3 , loop4 ] = self.input.shape
        self.output = self.input.reshape( loop1 , loop2 * loop3 * loop4 )
        
        return self.output
    
    def backward_propagate(self, gradient):
        super().backward_propagate(gradient)

        return gradient.reshape( self.input.shape )



class FullyConnectedLayer(ExternalLayers):

    def __init__(self , input_channels , output_channels ) -> None:
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.bias = np.zeros(self.output_channels)

        self.weights = np.random.normal(
            size=(self.output_channels , self.input_channels),
            loc=0.0,
            scale=1.0
        )/ self.input_channels

    def feed_forward(self, input):
        super().feed_forward(input.T)

        [loop , *_ ] = input.shape

        self.output = np.zeros(
            (loop , self.output_channels)
        )

        for i in range(loop):
            for j in range( self.output_channels ):
                self.output[i][j] = np.dot(
                    self.weights[j][:],
                    input[i][:]
                ) + self.bias[j]

        return self.output
    
    def backward_propagate(self, old_delta):
        super().backward_propagate(None)

        [loop , *_ ] = old_delta.shape

        delta_weights = np.zeros(
            (self.output_channels , self.input_channels )
        )

        delta_bias = np.zeros( self.output_channels )
        
        for i in range(loop):
            for j in range(self.output_channels):

                delta_bias[j] += old_delta[i][j]
        
        delta_weights = (np.dot( self.input , old_delta )).T
        new_delta = np.dot( old_delta , self.weights )

        self.weights -= learning_rate * delta_weights
        self.bias -= learning_rate * delta_bias

        return new_delta
        

class CrossEntropy(ExternalLayers):

    def __init__(self , output , target ) -> None:
        super().__init__()
        self.output = output 
        self.target = target

    def calculate_cost(self):
        label = np.zeros( self.output.shape )
        label[ np.arange( self.target.size ) , self.target.ravel()] = 1
        [loop1 , loop2 , *_] = self.output.shape
        loss = 0

        for i in range( loop1 ):
            for j in range( loop2 ):
                loss += - label[i][j] * np.log(
                    self.output[i][j]
                )
        print('He: ', loss / loop1 )
        return loss/loop1

        



class ActivationLayers():

    def __init__(self):
        self.output = None
        self.input = None
        

    def feed_forward( self , input ):
        self.input = input
    
    def backward_propagate( self ):
        pass


class Relu( ActivationLayers ):

    def __init__(self):
        super().__init__()
    
    def feed_forward(self, input):
        super().feed_forward(input)
        result = np.maximum( 0 , self.input )
        return result
    
    def backward_propagate(self, previous_layer):
        
        return_delta = np.zeros(previous_layer.shape)
        return_delta[ self.input > 0 ] = previous_layer[ self.input > 0]
        return return_delta


class Softmax( ActivationLayers ):
    
    def __init__(self):
        super().__init__()

    def feed_forward(self, input):
        super().feed_forward(input)

        [loop1 , loop2 , *_] = self.input.shape

        self.output = np.zeros(
            (loop1 , loop2)
        )

        for i in range(loop1):
            for j in range(loop2):
                self.output[i][j] = np.exp(
                    self.input[i][j]
                ) / np.sum(
                    np.exp(
                        input[i][:]
                    )
                )

        return self.output
    
    def backward_propagate(self , gradient):
        super().backward_propagate()
        result = np.subtract( self.output , gradient)/self.output.shape[0]
        return result





def load_MNIST_dataset():
        
    X, y = loadlocal_mnist(
            images_path='./Data/MNIST/train-images-idx3-ubyte', 
            labels_path='./Data/MNIST/train-labels-idx1-ubyte')
    X_t , y_t = loadlocal_mnist(
            images_path='./Data/MNIST/t10k-images-idx3-ubyte', 
            labels_path='./Data/MNIST/t10k-labels-idx1-ubyte')


    X_train = np.frombuffer(X, dtype=np.uint8).astype(np.float32).reshape(X.shape[0] , int(math.sqrt(X.shape[1])) ,  int(math.sqrt(X.shape[1])),1 )
    y_train = np.frombuffer(y, dtype=np.uint8).astype(np.int64).reshape( X.shape[0] ,  )

    # X_train -= int(np.mean(X_train))
    # X_train /= int(np.std(X_train))

    # train_data = np.hstack((X_train , y_train))

    # np.random.shuffle(train_data)
    

    X_test = np.frombuffer( X_t, dtype=np.uint8).astype(np.float32).reshape( X_t.shape[0] , int(math.sqrt(X_t.shape[1])) , int(math.sqrt(X_t.shape[1])),1 )
    y_test = np.frombuffer(y_t, dtype=np.uint8).astype(np.int64).reshape( X_t.shape[0] ,  )

    # X_test -= int(np.mean(X_test))
    # X_test /= int(np.std(X_test))

    # test_data = np.hstack((X_test , y_test))

    return X_train , y_train , X_test , y_test 



if __name__ =='__main__':
    X_train , y_train , X_test , y_test = load_MNIST_dataset()

    
    mini_Xtrain_images = X_train[:5000]
    mini_ytrain_images = y_train[:5000]

    X_batch = X_train[:32]
    y_batch = y_train[:32]

    y_label = np.zeros(( y_batch.size , 10))
    y_label[ np.arange(y_batch.size) , y_batch.ravel()] = 1


    conv1 = CNN(6,5,1,2,1)
    input = conv1.feed_forward(X_batch)
    relu1 = Relu()
    input = relu1.feed_forward(input)
    maxpool1 = MaxPool(2 , 2)
    input = maxpool1.feed_forward(input)

    conv2 = CNN(12,5,1,0,6)
    input = conv2.feed_forward(input)
    relu2 = Relu()
    input = relu2.feed_forward(input)
    maxpool2 = MaxPool(2 , 2)
    input = maxpool2.feed_forward(input)

    conv3 = CNN(100,5,1,0,12)
    input = conv3.feed_forward(input)
    relu3 = Relu()
    input = relu3.feed_forward(input)
    
    flat = FlatteningLayer()
    input = flat.feed_forward(input)
    
    fullyConnected = FullyConnectedLayer(100,10)
    input = fullyConnected.feed_forward(input)

    softmax = Softmax()
    input = softmax.feed_forward(input)

    grad = softmax.backward_propagate(y_label)
    grad = fullyConnected.backward_propagate(grad)
    grad = flat.backward_propagate(grad)
    grad = flat.backward_propagate(grad)
    grad = relu3.backward_propagate(grad)
    grad = conv3.backward_propagate(grad)
    grad = maxpool2.backward_propagate(grad)
    grad = relu2.backward_propagate(grad)
    grad = conv2.backward_propagate(grad)
    grad = maxpool1.backward_propagate(grad)
    grad = relu1.backward_propagate(grad)
    grad = conv1.backward_propagate(grad)

    pass