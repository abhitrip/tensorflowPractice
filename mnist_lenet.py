"""
Here I use Yann Le Cunn's Lenet to classify
handwritten digits of the Mnist

"""
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import flatten
import numpy as np
import tensorflow  as tf
import random
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

BATCH_SIZE = 50
EPOCHS = 10

class MnistDataHandler(object):
    """Class for handling mnist data"""
    def __init__(self, datapath,epochs,batch_size):
        self.mnist = input_data.read_data_sets(datapath,one_hot=True,reshape=False)
        self.X_train = self.mnist.train.images
        self.y_train = self.mnist.train.labels
        self.X_validation = self.mnist.validation.images
        self.y_validation = self.mnist.validation.labels
        self.X_test = self.mnist.test.images
        self.y_test = self.mnist.test.labels
        assert(len(self.X_train) == len(self.y_train))
        assert(len(self.X_validation) == len(self.y_validation))
        assert(len(self.X_test) == len(self.y_test))
        self.epochs = 10
        self.batch_size = BATCH_SIZE

        self.num_epochs = epochs
        self.batch_size = batch_size
        self.steps_per_epoch = self.mnist.train.num_examples//self.batch_size
        self.shuffle_train_data()


    def plot_random_data(self):
        index = random.randint(0, len(self.X_train))
        image = self.X_train[index].squeeze()

        plt.figure(figsize=(1,1))
        plt.imshow(image, cmap="gray")
        print(self.y_train[index])

    def shuffle_train_data(self):
        self.X_train, self.y_train = shuffle(self.X_train, self.y_train)

    def next_batch(self):
        mnist = self.mnist
        return mnist.train.next_batch(self.batch_size)

class LeNet(object):
    """
    Helper class to classify images using Lenet architecture
    """
    def __init__(self):
        pass

    def build(self,x):
        mu = 0
        sigma = 0.1
        x = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="CONSTANT")

        conv1_w = tf.Variable(tf.truncated_normal([5,5,1,6], mean=mu, stddev=sigma))
        conv1_b = tf.Variable(tf.zeros([6]))
        conv1 = tf.nn.conv2d(x,conv1_w,strides=[1,1,1,1],padding='VALID')
        conv1 = tf.nn.bias_add(conv1,conv1_b)

        # TODO: Activation.
        conv1 = tf.nn.relu(conv1)
        # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.

        conv1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

        # TODO: Layer 2: Convolutional. Output = 10x10x16.

        conv2_w = tf.Variable(tf.truncated_normal([5,5,6,16], mean=mu, stddev=sigma))
        conv2_b = tf.Variable(tf.zeros([16]))
        conv2 = tf.nn.conv2d(conv1,conv2_w,strides=[1,1,1,1],padding='VALID')
        conv2 = tf.nn.bias_add(conv2,conv2_b)




        # TODO: Activation.
        conv2 = tf.nn.relu(conv2)


        # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
        conv2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')


        # TODO: Flatten. Input = 5x5x16. Output = 400.
        dense0 = flatten(conv2)


        # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
        dense1_w = tf.Variable(tf.truncated_normal([400,120],mean=mu,stddev=sigma))
        dense1_b = tf.Variable(tf.zeros([120]))

        dense1 = tf.add(tf.matmul(dense0,dense1_w),dense1_b)



        # TODO: Activation.
        dense1 = tf.nn.relu(dense1)

        # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
        dense2_w = tf.Variable(tf.truncated_normal([120,84],mean=mu,stddev=sigma))
        dense2_b = tf.Variable(tf.zeros([84]))

        dense2 = tf.add(tf.matmul(dense1,dense2_w),dense2_b)




        # TODO: Activation.

        dense2 = tf.nn.relu(dense2)

        # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
        dense3_w = tf.Variable(tf.truncated_normal([84,10],mean=mu,stddev=sigma))
        dense3_b = tf.Variable(tf.zeros([10]))

        logits = tf.add(tf.matmul(dense2,dense3_w),dense3_b)

        return logits

# MNIST consists of 28x28x1, grayscale images
x = tf.placeholder(tf.float32, (None, 28, 28, 1))
# Classify over 10 digits 0-9
y = tf.placeholder(tf.float32, (None, 10))

lenet = LeNet()

fc2 = lenet.build(x)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc2, labels=y))
opt = tf.train.AdamOptimizer()
train_op = opt.minimize(loss_op)
correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def eval_data(dataset):
    """
    Given a dataset as input returns the loss and accuracy.
    """
    # If dataset.num_examples is not divisible by BATCH_SIZE
    # the remainder will be discarded.
    # Ex: If BATCH_SIZE is 64 and training set has 55000 examples
    # steps_per_epoch = 55000 // 64 = 859
    # num_examples = 859 * 64 = 54976
    #
    # So in that case we go over 54976 examples instead of 55000.
    steps_per_epoch = dataset.num_examples // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE
    total_acc, total_loss = 0, 0
    sess = tf.get_default_session()
    for step in range(steps_per_epoch):
        batch_x, batch_y = dataset.next_batch(BATCH_SIZE)
        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={x: batch_x, y: batch_y})
        total_acc += (acc * batch_x.shape[0])
        total_loss += (loss * batch_x.shape[0])
    return total_loss/num_examples, total_acc/num_examples


if __name__ == '__main__':
    # Load data
    #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    mnist_handler = MnistDataHandler("MNIST_data/",10,50)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        steps_per_epoch = mnist_handler.steps_per_epoch
        num_examples = steps_per_epoch * BATCH_SIZE

        # Train model
        for i in range(EPOCHS):
            for step in range(steps_per_epoch):
                batch_x, batch_y = mnist_handler.next_batch()
                loss = sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

            val_loss, val_acc = eval_data(mnist_handler.mnist.validation)
            print("EPOCH {} ...".format(i+1))
            print("Validation loss = {:.3f}".format(val_loss))
            print("Validation accuracy = {:.3f}".format(val_acc))
            print()

        # Evaluate on the test data
        test_loss, test_acc = eval_data(mnist_handler.mnist.test)
        print("Test loss = {:.3f}".format(test_loss))
        print("Test accuracy = {:.3f}".format(test_acc))









