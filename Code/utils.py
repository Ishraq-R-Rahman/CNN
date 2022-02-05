import numpy as np
from convolution import Convolution

def initialize_theta(NUM_OUTPUT, l_in):
	
	return 0.01*np.random.rand(NUM_OUTPUT, int(l_in))

def initialize_parameters( filter_dimension , depth_image ):
    filter_shape = filter_dimension ** 2 * depth_image
    standard_deviation = np.sqrt(1./filter_shape)
    shape = (depth_image , filter_dimension , filter_dimension )

    return np.random.normal( loc = 0 , scale= standard_deviation , size= shape)

def create_filters( convolution_layers , image_width , output_dimension):

    filters = []
    biases = []
    temp_filters = {}
    temp_biases = {}

    for i in range(convolution_layers[0]['outputChannels']):
        temp_filters[i] = initialize_parameters( filter_dimension= convolution_layers[0]['filterDimension'] , depth_image=1 )
        temp_biases[i] = 0
    

    filters.append(temp_filters)
    biases.append(temp_biases)

    for idx in range( 1 , len(convolution_layers)):
        temp_filters = {}
        temp_biases = {}

        for i in range(convolution_layers[idx]['outputChannels']):
            temp_filters[i] = initialize_parameters( filter_dimension= convolution_layers[idx]['filterDimension'] , depth_image= convolution_layers[idx-1]['outputChannels'] )
            temp_biases[i] = 0

        filters.append(temp_filters)
        biases.append(temp_biases)


    # Initialize theta

    w = []
    
    w.append(image_width - convolution_layers[0]['filterDimension'] + 1)

    for i in range( 1 , len(filters ) ):
        w.append( w[i-1] - convolution_layers[i]['filterDimension'] + 1 )

    initial_theta = initialize_theta( output_dimension , ( w[-1]/2) ** 2 * convolution_layers[-1]['outputChannels'] )

    initial_bias = np.zeros(( output_dimension , 1 ))

    return filters , initial_theta , initial_bias , biases


def get_strides_list(convolution_layers):
    stride = []
    for item in convolution_layers:
        stride.append(item['stride'])
    return stride



def gradient_descent( batch , learning_rate , img_width , img_depth , filters , biases_list , theta , bias , cost , acc , relu_activation , max_pooling_layers ,stride, MU = 0.95 ):
    
    X = batch[: , 0 : -1]

    X = X.reshape(len(batch), img_depth , img_width , img_width)

    y = batch[:,-1]


    correct_count = 0
    cost_ = 0
    batch_size = len(batch)
    dfilters = []
    dbiases = []
    v_list = []
    bv_list = []

    for i in range(len(filters)):
        tempDfilter = {}
        tempDbias = {}
        tempV = {}
        tempBV = {}

        for k in range( len(filters[i]) ):
            tempDfilter[k] = np.zeros( filters[i][0].shape )
            tempDbias[k] = 0
            tempV[k] = np.zeros( filters[i][0].shape )
            tempBV[k] = 0
        
        dfilters.append(tempDfilter)
        dbiases.append(tempDbias)
        v_list.append(tempV)
        bv_list.append(tempBV)

    dtheta = np.zeros(theta.shape)
    dbias = np.zeros(bias.shape)
    v = np.zeros( theta.shape )
    bv = np.zeros( bias.shape )


    for i in range(batch_size):
        
        image = X[i]
        label = np.zeros( ( theta.shape[0] , 1 ) )
        label[int(y[i]),0] = 1

        # Fetch gradient for current parameters

        Convolution( image , label , filters=filters , biases=biases_list , theta=theta , bias=bias , relu_activation=relu_activation , max_pooling_layers = max_pooling_layers ,stride=stride).model()
        

    pass