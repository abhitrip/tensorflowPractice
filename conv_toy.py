# import tensorflow as tf
# input = tf.placeholder(tf.float32,(None,32,32,3))
# filter_weights = tf.placeholder(tf.truncated_normal(8,8,3,20))
# filter_bias = tf.placeholder(tf.zeros(20))
# strides = [1,2,2,1] # (batch,height,width,depth)
# padding = 'SAME'
# conv = tf.nn.conv2d(input,filter_weights,strides,padding)+filter_bias

import tensorflow as tf
# output depth
k_output = 64

# Image properties
image_width = 10
image_height = 10
color_channels = 3

# Convolution filter
filter_size_width = 5
filter_size_height = 5

# Input/Image
input = tf.placeholder(tf.float32,shape=[None,image_height,image_width,
                                        image_width,color_channels])

# Weight and bias
Weight = tf.Variable(tf.truncated_normal([filter_size_height,
                                        filter_size_width,color_channels
                                        ,k_output]))
bias = tf.Variable(tf.zeros(k_output))

# Apply convolution
conv_layer = tf.nn.conv2d(input,Weight,strides=[1,2,2,1],padding='SAME')

# Add bias
conv_layer = tf.nn.bias_add(conv_layer,bias)

# Apply activation function
conv_layer = tf.nn.relu(conv_layer)

# Apply pooling
conv_layer = tf.nn.max_pool(
             conv_layer,
             ksize=[1,2,2,1],
             strides=[1,2,2,1],
             padding='SAME'
            )
