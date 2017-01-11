import tensorflow as tf
import numpy as np

class dA(object):
    def __init__(
        self,
        n_input,
        n_hidden,
        weight_input=None,
        bias_input=None,
        transfer_function=tf.nn.softplus,
        dropout_probability=0.05,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
    ):
        """
        :param n_input: number of input units
        :param n_hidden: number of hidden units
        :param transfer_function: activation function
        :param optimizer: optimization strategy
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.keep_dropout_probability = 1 - dropout_probability
        self.keep_prob = tf.placeholder(tf.float32)

        all_weights = dict()
        # forward weight
        if weight_input == None:
            all_weights['w1'] = tf.Variable(self.xavier_init(self.n_input, self.n_hidden))
        else:
            all_weights['w1'] = weight_input

        if bias_input == None:
            all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        else:
            all_weights['b1'] = bias_input
        # backward weight
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))  # pay attention: 0
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))

        self.weights = all_weights

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(tf.nn.dropout(self.x, self.keep_prob), self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])  # not activated

        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.sub(self.reconstruction, self.x), 2.0))  # pay attention: cost func.
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def xavier_init(self, fan_in, fan_out, constant=1):
        # define weights initialize domain
        low = -constant * np.sqrt(6.0/(fan_in + fan_out))
        high = constant * np.sqrt(6.0/(fan_in + fan_out))
        return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

    def partial_fit(self, X):
        # print 'dA running'
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X,
                                                                          self.keep_prob: self.keep_dropout_probability})
        return cost

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X, self.keep_prob: 1.0})

    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.keep_prob: 1.0})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])
