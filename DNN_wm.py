import tensorflow as tf
import numpy as np
import sklearn.preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data

class DNN(object):
    def __init__(self,
                 n_input=750,
                 n_class=2,
                 hidden_layers=[1000, 1000],
                 activate_function=tf.nn.relu,
                 learning_rate=0.001,
                 training_epochs=100,
                 batch_size=100,
                 constant=1,
                 ):
        self.n_layers = hidden_layers
        self.training_epochs = training_epochs
        self.batch_size = batch_size

        self.x = tf.placeholder(tf.float32, [None, n_input])
        self.y = tf.placeholder(tf.float32, [None, n_class])

        self.Weights = []
        self.Bias = []

        for i in range(len(self.n_layers)):
            if i == 0:
                # w, b
                low = -constant * np.sqrt(6.0 / (n_input + self.n_layers[i]))
                high = constant * np.sqrt(6.0 / (n_input + self.n_layers[i]))
                w = tf.Variable(tf.random_uniform((n_input, self.n_layers[i]), minval=low, maxval=high, dtype=tf.float32))
                b = tf.Variable(tf.zeros([self.n_layers[i]], dtype=tf.float32))
                self.Weights.append(w)
                self.Bias.append(b)
                # forward
                hidden_layer_i = activate_function(tf.add(tf.matmul(self.x, self.Weights[i]), self.Bias[i]))
            else:
                # w, b
                low = -constant * np.sqrt(6.0 / (self.n_layers[i] + self.n_layers[i-1]))
                high = constant * np.sqrt(6.0 / (self.n_layers[i] + self.n_layers[i-1]))
                w = tf.Variable(
                    tf.random_uniform((self.n_layers[i-1], self.n_layers[i]), minval=low, maxval=high, dtype=tf.float32))
                b = tf.Variable(tf.zeros([self.n_layers[i]], dtype=tf.float32))
                self.Weights.append(w)
                self.Bias.append(b)
                # forward
                hidden_layer_i = activate_function(tf.add(tf.matmul(hidden_layer_i, self.Weights[i]), self.Bias[i]))

        # w, b
        low = -constant * np.sqrt(6.0 / (self.n_layers[-1] + n_class))
        high = constant * np.sqrt(6.0 / (self.n_layers[-1] + n_class))
        w = tf.Variable(tf.random_uniform((self.n_layers[-1], n_class), minval=low, maxval=high, dtype=tf.float32))
        b = tf.Variable(tf.zeros([n_class], dtype=tf.float32))
        self.Weights.append(w)
        self.Bias.append(b)

        # forward
        self.pred = tf.nn.softmax(tf.add(tf.matmul(hidden_layer_i, self.Weights[-1]), self.Bias[-1]))

        # init = tf.initialize_all_variables()
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        # define cost and optimizer
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.y*tf.log(self.pred), reduction_indices=1))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.cost)

    def fit(self, X, Y):
        for epoch in range(self.training_epochs):
            avg_cost = 0.
            total_batch = int(X.shape[0]/self.batch_size)
            for i in range(total_batch):
                batch_x = X[i * self.batch_size: (i + 1) * self.batch_size]
                batch_y = Y[i * self.batch_size: (i + 1) * self.batch_size]
                cost, _ = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: batch_x, self.y: batch_y})
                avg_cost += cost/total_batch
            print 'epoch%s' % str(epoch+1), 'cost:', avg_cost
        # return cost

    def predict_proba(self, X):
        return self.sess.run(self.pred, feed_dict={self.x: X})

def DNN_test():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    def standard_scale(X_train, X_test):
        preprocessor = prep.StandardScaler().fit(X_train)
        X_train = preprocessor.transform(X_train)
        X_test = preprocessor.transform(X_test)
        return X_train, X_test

    X_train, X_test, y_train, y_test = mnist.train.images, mnist.test.images, mnist.train.labels, mnist.test.labels
    X_train, X_test = standard_scale(X_train, X_test)
    print y_train.shape
    dnn = DNN(n_input=784, n_class=10)
    dnn.fit(X_train, y_train)
    y_test_pred = dnn.predict_proba(X_test)
    print y_test_pred[0:100]


if __name__ == '__main__':
    DNN_test()