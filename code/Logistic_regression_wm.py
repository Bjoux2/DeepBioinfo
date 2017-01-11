import tensorflow as tf
import numpy as np
from load_data import load_data
import sklearn.preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import accuracy_score


class LR(object):
    def __init__(self,
                 n_input=750,
                 n_class=2,
                 learning_rate=0.001,

                 ):
        self.x = tf.placeholder(tf.float32, [None, n_input])
        self.y = tf.placeholder(tf.float32, [None, n_class])

        self.w = tf.Variable(tf.zeros([n_input, n_class], dtype=tf.float32))
        self.b = tf.Variable(tf.zeros([n_class], dtype=tf.float32))

        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)

        self.pred = tf.nn.softmax(tf.add(tf.matmul(self.x, self.w), self.b))
        # self.pred_ = np.argmax(self.pred, axis=1)
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.y*tf.log(self.pred), reduction_indices=1))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.cost)

    def fit(self, X, Y, train_epoch=25, batch_size=100):
        for epoch in range(train_epoch):
            total_batch = int(X.shape[0] / batch_size)
            avg_cost = 0.
            for i in range(total_batch):
                batch_x = X[i * batch_size: (i + 1) * batch_size]
                batch_y = Y[i * batch_size: (i + 1) * batch_size]
                _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.x: batch_x, self.y: batch_y})

                avg_cost += c/total_batch
            # print 'epoch%s,' % str(epoch + 1), 'cost:', avg_cost

    def predict_proba(self, X):
        return self.sess.run(self.pred, feed_dict={self.x: X})

    # def predict(self, X):
    #     return self.sess.run(self.pred_, feed_dict={self.x: X})

def test_LR():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    def standard_scale(X_train, X_test):
        preprocessor = prep.StandardScaler().fit(X_train)
        X_train = preprocessor.transform(X_train)
        X_test = preprocessor.transform(X_test)
        return X_train, X_test

    X_train, X_test, y_train, y_test = mnist.train.images, mnist.test.images, mnist.train.labels, mnist.test.labels
    X_train, X_test = standard_scale(X_train, X_test)
    print y_train.shape
    lr = LR(n_input=784, n_class=10)
    lr.fit(X_train, y_train)
    y_test_pred = lr.predict_proba(X_test)
    y_pred = np.argmax(y_test_pred, axis=1)
    print y_test
    print accuracy_score(y_pred, np.argmax(y_test, axis=1))


if __name__ == "__main__":
    test_LR()
