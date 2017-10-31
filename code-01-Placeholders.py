from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Placeholder that will be fed image data.
    x = tf.placeholder(tf.float32, [None, 784])
    # Placeholder that will be fed the correct labels.
    y_ = tf.placeholder(tf.float32, [None, 10])


if __name__ == '__main__':
    main()
