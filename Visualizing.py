import numpy as np 
import tensorflow as tf 
from tensorflow import keras
from keras .applications.vgg16 import VGG16
from matplotlib import pyplot
from keras.models import Model
from keras.preprocessing.image import *
from keras.applications.vgg16 import preprocess_input
'''the result of applying filters called activation maps, or more generally, feature maps '''

# load model
model = VGG16()
#model.summary()

# Visual Filter
def visual_filter():
    # summarize filter shape and get filter weights
    
    '''
    We can access all of the layers of the model via the model.layers property.
    Each layer has a layer.name property, where the convolutional layers have a naming convolution like block#_conv#, where the ‘#‘ is an integer. 
    Therefore, we can check the name of each layer and skip any that don’t contain the string ‘conv‘.
    '''
    
    '''
    Each convolutional layer has two sets of weights.
    One is the block of filters and the other is the block of bias values. 
    These are accessible via the layer.get_weights() function. 
    We can retrieve these weights and then summarize their shape
    '''
    
    '''
    # get all filter
    for layer in model.layers:
        if 'conv' not in layer.name:
            continue
        filters, biases = layer.get_weights()
        print(layer.name, filters.shape)  #(h,w,chanel, số lượng filter)
    '''
        
    
    # retrieve weights from the second hidden layer
    filters, biaes = model.layers[1].get_weights() # layer conv thứ 1, layer hidden đánh chỉ số từ 1
    print(filters.shape)
    
    '''
    The weight values will likely be small positive and negative values centered around 0.0.
    We can normalize their values to the range 0-1 to make them easy to visualize.
    '''
    
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min)/(f_max - f_min)
    
    # plot first few filters
    n_filters = 6  # số filter lấy ra
    idx = 1
    for i in range(n_filters):
        # get the filter
        f = filters[:, :, :, i]
        # plot 3 channel separetely
        for j in range(3):
            ax = pyplot.subplot(n_filters, 3, idx) # nrow, ncol, index
            ax.set_xticks([]) # bỏ đánh chỉ số tọa độ
            ax.set_yticks([])
            pyplot.imshow(f[:, :, j], cmap='gray')
            idx += 1
    pyplot.show()


# Visualize Feature Map

def visual_feature_map():
    # định nghĩa model mới với đầu ra là đầu ra của tầng conv1
    model_new = Model(inputs=model.inputs, outputs=model.layers[1].output)
    model_new.summary()
    # Making a prediction with this model will give the feature map for the first convolutional layer
    img = load_img('bird.jpg', target_size=(224, 224))  # chiều giữ nguyên
    img = np.expand_dims(img, axis=0) #[sample, row, col. channels]
    # the pixel values then need to be scaled appropriately for the VGG model
    img = preprocess_input(img)
    feature_maps = model_new.predict(img)
    
    # plot all 64 maps in an 8X8 squares
    square = 8
    idx = 1
    
    for _ in range(square):
        for _ in range(square):
            ax = pyplot.subplot(square, square, idx)
            ax.set_xticks([])
            ax.set_yticks([])
            pyplot.imshow(feature_maps[0, :, :, idx-1], cmap='gray') # mẫu số 0, channel idx-1
            idx += 1

    pyplot.show()



visual_feature_map()








































