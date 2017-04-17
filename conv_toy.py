# import tensorflow as tf
# input = tf.placeholder(tf.float32,(None,32,32,3))
# filter_weights = tf.placeholder(tf.truncated_normal(8,8,3,20))
# filter_bias = tf.placeholder(tf.zeros(20))
# strides = [1,2,2,1] # (batch,height,width,depth)
# padding = 'SAME'
# conv = tf.nn.conv2d(input,filter_weights,strides,padding)+filter_bias

