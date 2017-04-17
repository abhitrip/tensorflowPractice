from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('mnist_data/', one_hot=True)

import tensorflow as tf

tf.reset_default_graph()
# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network parameters
n_hidden_1 = 256
n_input = 784
n_classes = 10

# Input graph
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# Create Model for multi layer perceptron


def multilayer_perceptron(x, weights, biases):

    # Hidden layer 1
    hidden_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    hidden_activated = tf.nn.relu(hidden_1)
    output = tf.add(tf.matmul(hidden_activated, weights['out']), biases['out'])
    return output

# Store weights and biases
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

logits = multilayer_perceptron(x, weights, biases)
# Loss

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= \
    logits, labels=y))

"""
softmax = tf.nn.softmax(logits)
cost = tf.reduce_mean(tf.reduce_sum(-y*tf.log(softmax),reduction_indices=[1]))
"""

save_file = './train_model.ckpt'

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

init = tf.initialize_all_variables()

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost=0
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            # Run optimizer
            _,c = sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})
            avg_cost += c/total_batch
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost)
            validation_accuracy = sess.run(
                    accuracy,
                    feed_dict={
                        x: mnist.validation.images,
                        y: mnist.validation.labels
                    }
                    )
            print('Epoch {:<3} - Validation Accuracy: {}'.format(
                epoch,
                validation_accuracy))

    print "Optimization Finished!"

    # Test model
    correct_test_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    # Calculate accuracy
    test_accuracy = tf.reduce_mean(tf.cast(correct_test_prediction, "float"))
    print "Accuracy:", test_accuracy.eval({x: mnist.test.images,
                                            y: mnist.test.labels})
    saver.save(sess,save_file)
    print('Trained Model saved')



