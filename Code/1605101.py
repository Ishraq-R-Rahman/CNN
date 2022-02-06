import math
import numpy as np
from mlxtend.data import loadlocal_mnist
import platform
from six.moves import cPickle as pickle # type: ignore
import os



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


def read_input( file ):

    commands = []

    with open(file,'r') as f:
        for line in f:
            command = line.strip().split()

            commands.append({
                'type' : command[0],
                'value': [int(i) for i in command[1:]]
            })
    return commands


def create_command_structure( commands ,channel_in = 1):

    convolution_layers = []
    max_pool_layers = []
    relu_layers = []
    flat_layers = []
    fc_layers = []
    softmax_layers = []

    input_channel = channel_in
    
    for command in commands:

        if command['type'] == 'Conv':
            [ output_channels , filter_dimension , stride , padding ] = command['value']

            convolution_layer = CNN(output_channels=output_channels , filter_dimension=filter_dimension , stride=stride , padding=padding , input_channels=input_channel)

            convolution_layers.append(convolution_layer)

        elif command['type'] == 'ReLU':
            relu_layer = Relu()

            relu_layers.append(relu_layer)
        
        elif command['type'] == 'Pool':
            [ filter_dimension , stride ] = command['value']
            maxpool = MaxPool( filter_dimension= filter_dimension , stride=stride )

            max_pool_layers.append(maxpool)
        
        elif command['type'] == 'FC':
            output_channels = command['value'][0]

            flat = FlatteningLayer()

            fully_connected_layer = FullyConnectedLayer(input_channel,output_channels)

            flat_layers.append(flat)
            fc_layers.append(fully_connected_layer)
        
        elif command['type'] == 'Softmax':
            softmax = Softmax()

            softmax_layers.append(softmax)

    return {
        'conv': convolution_layers,
        'pool': max_pool_layers,
        'relu': relu_layers,
        'soft': softmax_layers,
        'fc': fc_layers,
        'flat': flat_layers
    }


def run_model( structure , commands , X_batch , y_label ,channel_in = 1 , final_return=False):

    convolution_layers = structure['conv']
    max_pool_layers = structure['pool']
    relu_layers = structure['relu']
    flat_layers = structure['flat']
    fc_layers = structure['fc']
    softmax_layers = structure['soft']

    input = X_batch

    input_channel = channel_in

    # Forward Propagation
    for command in commands:

        if command['type'] == 'Conv':
            [ output_channels , filter_dimension , stride , padding ] = command['value']

            convolution_layer = CNN(output_channels=output_channels , filter_dimension=filter_dimension , stride=stride , padding=padding , input_channels=input_channel)
            input = convolution_layer.feed_forward( input )
            input_channel = output_channels

            convolution_layers.append(convolution_layer)

        elif command['type'] == 'ReLU':
            relu_layer = Relu()
            input = relu_layer.feed_forward(input)

            relu_layers.append(relu_layer)
        
        elif command['type'] == 'Pool':
            [ filter_dimension , stride ] = command['value']
            maxpool = MaxPool( filter_dimension= filter_dimension , stride=stride )
            input = maxpool.feed_forward( input )

            max_pool_layers.append(maxpool)
        
        elif command['type'] == 'FC':
            output_channels = command['value'][0]

            flat = FlatteningLayer()
            input = flat.feed_forward(input)

            fully_connected_layer = FullyConnectedLayer(input_channel,output_channels)
            input = fully_connected_layer.feed_forward(input)

            flat_layers.append(flat)
            fc_layers.append(fully_connected_layer)
        
        elif command['type'] == 'Softmax':
            softmax = Softmax()
            input = softmax.feed_forward(input)

            softmax_layers.append(softmax)
    
    if final_return:
        return structure , input
    
    # Backward Propagation
    input = y_label

    # temp_convolution_layers = convolution_layers.copy()
    # temp_max_pool_layers = max_pool_layers.copy()
    # temp_relu_layers = relu_layers.copy()
    # temp_flat_layers = flat_layers.copy()
    # temp_fc_layers = fc_layers.copy()
    # temp_softmax_layers = softmax_layers.copy()

    convCounter = 1
    poolCounter = 1
    flatCounter = 1
    fcCounter = 1
    reluCounter = 1
    softmaxCounter = 1


    for command in commands[::-1]:
        if command['type'] == 'Softmax':
            # input = temp_softmax_layers[-1].backward_propagate( input )
            input = softmax_layers[-softmaxCounter].backward_propagate( input )
            softmaxCounter += 1

            # temp_softmax_layers.pop()
        
        elif command['type'] == 'FC':

            input = fc_layers[-fcCounter].backward_propagate( input )
            input = flat_layers[-flatCounter].backward_propagate( input )

            flatCounter += 1
            fcCounter += 1

            # temp_fc_layers.pop()
            # temp_flat_layers.pop()

        elif command['type'] == 'ReLU':

            input = relu_layers[-reluCounter].backward_propagate( input )
            reluCounter += 1

            # temp_relu_layers.pop()
        
        elif command['type'] == 'Pool':
            input = max_pool_layers[-poolCounter].backward_propagate( input )
            poolCounter += 1

            # temp_max_pool_layers.pop()
        
        elif command['type'] == 'Conv':
            input = convolution_layers[-convCounter].backward_propagate(input)
            convCounter += 1
            
            # temp_convolution_layers.pop()

    return structure
     


def train_models( X_train , y_train , X_valid , y_valid , X_test , y_test , batch_size , commands , channel_in = 1, epochs = 5 ):

    global learning_rate

    structure = create_command_structure( commands , channel_in )

    for epoch in range(epochs):
        
        loop = int( X_train.shape[0] / batch_size )
        for i in range( loop ):
            batch_X = X_train[ i * batch_size : i * batch_size + batch_size ]
            batch_y = y_train[ i * batch_size : i * batch_size + batch_size ]
    
            y_label = np.zeros( ( batch_y.size , 10 ))
            y_label[ np.arange( batch_y.size) , batch_y.ravel() ] = 1

            structure = run_model( structure , commands , batch_X , y_label , channel_in )
        
        structure , output_valid = run_model(structure , commands , X_valid , y_valid , channel_in , True )
        # structure , output_valid_accuracy = run_model(structure , commands , X_valid , y_valid , 1 , True )

        cost = CrossEntropy( output_valid , y_valid ).calculate_cost()

        if cost < 2.5:
            learning_rate = 0.0001
        else:
            learning_rate = 0.001

        print(f'Epoch - {epoch} \t Loss: {cost} ')
        print(f'Epoch - {epoch} \t Accuracy: {Accuracy( output_valid , y_valid ).calculate_cost()}  ')

    return structure



def test_model( structure , commands , X_test , y_test , channel_in ):

    # structure = run_model( structure , commands , X_test , y_test , channel_in )

    structure , output_test = run_model(structure , commands , X_test , y_test , channel_in , True )
    # structure , output_test_accuracy = run_model(structure , commands , X_valid , y_valid , 1 , True )

    print(f'\n\n Final Accuracy for the test : {Accuracy( output_test , y_test ).calculate_cost()}  ')


    


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
        

class Metrics():
    def __init__(self , output , target ) -> None:
        self.output = output 
        self.target = target

    def calculate_cost(self):
        pass


class CrossEntropy(Metrics):

    def __init__(self , output , target ) -> None:
        super().__init__( output , target )
        

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
        return loss/loop1


class Accuracy( Metrics ):

    def __init__(self, output, target) -> None:
        super().__init__(output, target)
    
    def calculate_cost(self):
        super().calculate_cost()

        count_correct = 0

        for i in range( self.output.shape[0] ):
            max = 0 
            argmax = 0

            for j in range( self.output.shape[1] ):

                if self.output[i][j] > max:
                    max = self.output[i][j]
                    argmax = j
            
            if argmax == self.target[i]:
                count_correct += 1
        return count_correct / self.output.shape[0]  


class F1( Metrics ):

    def __init__(self, output, target) -> None:
        super().__init__(output, target)
    
    def calculate_cost(self):
        super().calculate_cost()

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for i in range( self.output.shape[0] ):
            max = 0 
            argmax = 0

            for j in range( self.output.shape[1] ):

                if self.output[i][j] > max:
                    max = self.output[i][j]
                    argmax = j
            if argmax == self.target[i]:
                tp += 1

            elif argmax != self.target[i]:
                tn += 1     

        return 0



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



def load_CIFAR_10_dataset():
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    img_rows, img_cols = 32, 32
    input_shape = (img_rows, img_cols, 3)
    def load_pickle(f):
        version = platform.python_version_tuple()
        if version[0] == '2':
            return  pickle.load(f)
        elif version[0] == '3':
            return  pickle.load(f, encoding='latin1')
        raise ValueError("invalid python version: {}".format(version))

    def load_CIFAR_batch(filename):
        """ load single batch of cifar """
        with open(filename, 'rb') as f:
            datadict = load_pickle(f)
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000,3072)
            Y = np.array(Y)
            return X, Y

    def load_CIFAR10(ROOT):
        """ load all of cifar """
        xs = []
        ys = []
        for b in range(1,6):
            f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
            X, Y = load_CIFAR_batch(f)
            xs.append(X)
            ys.append(Y)
        Xtr = np.concatenate(xs)
        Ytr = np.concatenate(ys)
        del X, Y
        Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
        return Xtr, Ytr, Xte, Yte
    def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
        # Load the raw CIFAR-10 data
        cifar10_dir = './Data/CIFAR-10/'
        X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

        # Subsample the data
        mask = range(num_training, num_training + num_validation)
        X_val = X_train[mask]
        y_val = y_train[mask]
        mask = range(num_training)
        X_train = X_train[mask]
        y_train = y_train[mask]
        mask = range(num_test)
        X_test = X_test[mask]
        y_test = y_test[mask]

        x_train = X_train.astype('float32')
        x_test = X_test.astype('float32')

        x_train /= 255
        x_test /= 255

        return x_train, y_train, X_val, y_val, x_test, y_test


    # Invoke the above function to get our data.
    x_train, y_train, x_val, y_val, x_test, y_test = get_CIFAR10_data()

    x_train = x_train.reshape( x_train.shape[0], int(math.sqrt(x_train.shape[1]/3)) , int(math.sqrt(x_train.shape[1]/3)) , 3)
    x_val = x_val.reshape( x_val.shape[0], int(math.sqrt(x_val.shape[1]/3)) , int(math.sqrt(x_val.shape[1]/3)) , 3)
    x_test = x_test.reshape( x_test.shape[0], int(math.sqrt(x_test.shape[1]/3)) , int(math.sqrt(x_test.shape[1]/3)) , 3)

    # print('Train data shape: ', x_train.shape)
    # print('Train labels shape: ', y_train.shape)
    # print('Validation data shape: ', x_val.shape)
    # print('Validation labels shape: ', y_val.shape)
    # print('Test data shape: ', x_test.shape)
    # print('Test labels shape: ', y_test.shape)

    return x_train, y_train, x_val, y_val, x_test, y_test


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
    # print('Train data shape: ', X_train.shape)
    # print('Train labels shape: ', y_train.shape)
    # print('Validation data shape: ', x_val.shape)
    # print('Validation labels shape: ', y_val.shape)
    # print('Test data shape: ', X_test.shape)
    # print('Test labels shape: ', y_test.shape)

    return X_train , y_train , X_test , y_test 



if __name__ =='__main__':
    X_train , y_train , X_test , y_test = load_MNIST_dataset()
    # X_train , y_train , X_valid , y_valid , X_test , y_test = load_CIFAR_10_dataset()

    commands = read_input('input.txt')
    
    
    mini_Xtrain_images = X_train[:5000]
    mini_ytrain_images = y_train[:5000]

    X_valid = X_train[5000:6000]
    y_valid = y_train[5000:6000]
    # X_valid = X_valid[:1000]
    # y_valid = y_valid[:1000]

    mini_X_test = X_test[:5000]
    mini_y_test = y_test[:5000]


    structure = train_models(X_train=mini_Xtrain_images , y_train=mini_ytrain_images , X_valid=X_valid , y_valid= y_valid ,X_test= mini_X_test , y_test=mini_y_test , batch_size=32 , commands=commands , channel_in= 1 , epochs=5 )
    # structure = train_models(X_train=mini_Xtrain_images , y_train=mini_ytrain_images , X_valid=X_valid , y_valid= y_valid ,X_test= mini_X_test , y_test=mini_y_test , batch_size=32 , commands=commands , channel_in= 3 , epochs=5 )

    # test_model(structure , commands , X_test , y_test , channel_in = 1 )
    test_model(structure , commands , X_test , y_test , channel_in = 3 )


    
    pass