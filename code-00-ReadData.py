from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


if __name__ == '__main__':
    main()
