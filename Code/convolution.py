import numpy as np

def max_pooling_convolution_layer( convolution_layer , pool_dimension_w , pool_dimension_h , stride ):

    ( depth , width , _ ) = convolution_layer.shape

    pool = np.zeros( ( depth , int( ( width - pool_dimension_w ) / stride + 1 ) , int( ( width - pool_dimension_h ) / stride + 1 ) ) )

    print(pool.shape)

    for idx in range(depth):

        i = 0

        while i < width:

            j = 0

            while j < width:

                pool[ idx , int(i/2) , int(j/2) ] = np.max( convolution_layer[ idx , i : i + pool_dimension_w , j : j + pool_dimension_h ] )

                j += stride
            
            i += stride

    print('Pool: ', pool.shape)
    return pool

class Convolution:
    def __init__(self , image , label , filters , biases , theta , bias , relu_activation , max_pooling_layers , stride) -> None:
        self.image = image
        self.label = label
        self.filters = filters
        self.biases = biases
        self.theta = theta
        self.bias = bias
        self.relu_activation = relu_activation
        self.max_pooling_layers = max_pooling_layers
        self.stride = stride



    def feed_forward(self):
        
        '''
            Feed Forward to get all the layers
        '''



        ( depth , width , width ) = self.image.shape

        depth_list = []
        
        for f in self.filters:
            depth_list.append(len(f))

            # # Padding the image
            # pad = int((len(f)-1)/2)
            # np.pad( self.image , pad , mode='constant')
            # print(self.image.shape)
        
        w = []
        
        w.append( width - self.filters[0][0].shape[1] + 1 )

        for i in range( 1 , len(self.filters ) ):
            w.append( w[i-1] - self.filters[i][0].shape[1] + 1 )

        convolution_layers = []

        # Initialize the convolution layers
        for deep , wide in zip( depth_list , w ):
            convolution_layers.append(
                np.zeros( ( deep , wide , wide ) )
            )
        
        pooled_layer = []
        #Calculation for the convolution layers
        for layer_idx in range(len(convolution_layers)):

            for i in range( depth_list[layer_idx] ):

                for x in range( w[layer_idx] ):

                    for y in range( w[layer_idx] ):
                        
                        filter_dim = self.filters[layer_idx][i].shape[1]

                        # layer_considered = self.image if layer_idx == 0 else convolution_layers[layer_idx-1]
                        layer_considered = self.image if layer_idx == 0 else pooled_layer
                        
                        # print('In here : ' , layer_considered[ : , x : x + filter_dim , y : y + filter_dim ].shape )
                        # print('Out there : ' , self.filters[layer_idx][i].shape )

                        convolution_layers[layer_idx][ i , x , y ]  = np.sum( layer_considered[ : , x : x + filter_dim , y : y + filter_dim ] * self.filters[layer_idx][i] ) + self.biases[layer_idx][i]

            # Using Relu activation after every convolution layer           
            if self.relu_activation[layer_idx]:
                convolution_layers[layer_idx][ convolution_layers[layer_idx] <= 0] = 0
        
            # Pooling Layer and stride
            if len(self.max_pooling_layers[layer_idx]) != 0:
                pooled_layer = max_pooling_convolution_layer( convolution_layers[layer_idx] , self.max_pooling_layers[layer_idx][0] , self.max_pooling_layers[layer_idx][1] , self.stride[layer_idx]   )
                

            else:
                pooled_layer = convolution_layers[layer_idx]
        
        # Flatten Layer
        fully_connected_layer = pooled_layer.reshape( int(w[layer_idx]/2) * int(w[layer_idx]/2) * depth_list[layer_idx] , 1)
        output_layer = self.theta.dot(fully_connected_layer) + self.bias

        print(output_layer.shape)


        
            

    def model(self):

        self.feed_forward()

        pass