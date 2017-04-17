import tensorflow as tf

# The file to save weights
save_file = './model.ckpt'

# Two tensor variables weight and bias
weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

# Class used to save / and or restore variable
saver = tf.train.Saver()


def save_session():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Show the weights
        print('weights:')
        print(sess.run(weights))
        print('bias:')
        print(sess.run(bias))

        saver.save(sess, save_file)
    """
    weights:
    [[-0.67600113  0.14297155 -0.32967988]
     [ 1.23757279  0.6295619   1.47220898]]
    bias:
    [ 0.96275079 -0.88853949  0.18894714]
    [Finished in 1.1s]
    """


def load_session():
    with tf.Session() as sess:
        saver.restore(sess, save_file)
        print('weights:')
        print(sess.run(weights))
        print('bias:')
        print(sess.run(bias))

        saver.save(sess, save_file)
    """
    should print the same as in save_session()
    """


if __name__ == "__main__":
    load_session()
