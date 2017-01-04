import tensorflow as tf
import numpy as np
from dA_wm import dA


class SdA(object):
    """Stacked denoising auto-encoder class (SdA)"""
    def __init__(
        self,
        n_input=784,
        hidden_layers_sizes=[500, 600, 700],
        n_classes=10,
        transfer_function_pretraining=tf.nn.softplus,
        transfer_function_finetuning=tf.nn.relu,
        transfer_function_output=tf.nn.softmax,
        lr_pretraining=0.1,
        lr_finetuning=0.1,
        dropout_probability=0.05,
        batch_size_pretraining=256,
        batch_size_finetuning=256,
        epochs_pretraining=10,
        epochs_finetuning=100
    ):

        # self.DNN_layers = []
        self.dA_layers = []
        self.Weights = []
        self.Biases = []
        self.n_layers = len(hidden_layers_sizes)

        # # batch size and epoch
        self.batch_size_pretraining = batch_size_pretraining
        self.batch_size_finetuning = batch_size_finetuning
        self.epochs_pretraining = epochs_pretraining
        self.epochs_finetuning = epochs_finetuning

        # # number of layers must larger than 0
        assert self.n_layers > 0

        # # define hidden layer for dA
        for i in range(self.n_layers):
            if i == 0:
                input_size = n_input
            else:
                input_size = hidden_layers_sizes[i-1]

            dA_layer = dA(
                n_input=input_size,
                n_hidden=hidden_layers_sizes[i],
                transfer_function=transfer_function_pretraining,
                dropout_probability=dropout_probability,
                optimizer=tf.train.AdamOptimizer(learning_rate=lr_pretraining),
                )
            self.dA_layers.append(dA_layer)
            self.Weights.append(dA_layer.getWeights())
            self.Biases.append(dA_layer.getBiases())

        # # placeholders of x and y
        self.x = tf.placeholder(tf.float32, [None, n_input])
        self.y = tf.placeholder(tf.float32, [None, n_classes])

        for i in range(self.n_layers):
            if i == 0:
                layer_input = self.x
            else:
                layer_input = transfer_function_finetuning(tf.add(tf.matmul(layer_input, self.Weights[i-1]), self.Biases[i-1]))

        layer_input_for_top_layer = transfer_function_finetuning(tf.add(tf.matmul(layer_input, self.Weights[-1]), self.Biases[-1]))

        # We now need to add a logistic layer on top of the MLP
        constant = 1
        low = -constant * np.sqrt(6.0 / (hidden_layers_sizes[-1] + n_classes))
        high = constant * np.sqrt(6.0 / (hidden_layers_sizes[-1] + n_classes))

        w_toplayer = tf.Variable(tf.random_uniform([hidden_layers_sizes[-1], n_classes],
                                                   minval=low, maxval=high,
                                                   dtype=tf.float32))
        b_toplayer = tf.Variable(tf.zeros([n_classes]), dtype=tf.float32)

        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)

        self.Weights.append(w_toplayer)
        self.Biases.append(b_toplayer)

        self.pred = transfer_function_output(tf.add(tf.matmul(layer_input_for_top_layer, self.Weights[-1]), self.Biases[-1]))
        self.finetune_cost = tf.reduce_mean(-tf.reduce_sum(self.y*tf.log(self.pred), reduction_indices=1))
        self.optimizer_finetuning = tf.train.GradientDescentOptimizer(learning_rate=lr_finetuning).minimize(self.finetune_cost)

    def fit_pretraining(self, train_set_x):
        n_samples = train_set_x.shape[0]
        total_batch = int(n_samples / self.batch_size_pretraining)
        # the tf.Session was defined in dA class.
        for li in range(len(self.dA_layers)):
            for epoch in range(self.epochs_pretraining):
                avg_cost = 0.
                for i in range(total_batch):
                    if li == 0:
                        batch_x = train_set_x[i * self.batch_size_pretraining: (i+1) * self.batch_size_pretraining]
                        cost = self.dA_layers[li].partial_fit(batch_x)
                        avg_cost += cost / n_samples * self.batch_size_pretraining
                    else:
                        batch_x = train_set_x[i * self.batch_size_pretraining: (i + 1) * self.batch_size_pretraining]
                        for j in range(li):
                            batch_x = self.dA_layers[j].transform(batch_x)
                        cost = self.dA_layers[li].partial_fit(batch_x)
                        avg_cost += cost / n_samples * self.batch_size_pretraining

                print 'pretraining layer%s' % str(li+1), 'epoch%s' % str(epoch+1), 'cost %s' % avg_cost
            self.Weights[li] = self.dA_layers[li].getWeights()
            self.Biases[li] = self.dA_layers[li].getBiases()

    def fit_finetuning(self, train_set_x, train_set_y):
        n_samples = train_set_x.shape[0]
        total_batch = int(n_samples / self.batch_size_finetuning)
        for epoch in range(self.epochs_finetuning):
            avg_cost = 0.
            for i in range(total_batch):
                batch_x = train_set_x[i * self.batch_size_finetuning: (i+1) * self.batch_size_finetuning]
                batch_y = train_set_y[i * self.batch_size_finetuning: (i+1) * self.batch_size_finetuning]
                cost, opt = self.sess.run([self.finetune_cost, self.optimizer_finetuning], feed_dict={self.x: batch_x,
                                                                                                      self.y: batch_y})
                avg_cost += cost / n_samples * self.batch_size_pretraining

            print 'finetuning', 'epoch%s' % str(epoch), 'cost %s' % avg_cost

    def predict_proba(self, X):
        return self.sess.run(self.pred, feed_dict={self.x: X})

if __name__ == '__main__':
    import sklearn.preprocessing as prep
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    def standard_scale(X_train, X_test):
        preprocessor = prep.StandardScaler().fit(X_train)
        X_train = preprocessor.transform(X_train)
        X_test = preprocessor.transform(X_test)
        return X_train, X_test


    def get_random_block_from_data(data, batch_size):
        start_index = np.random.randint(0, len(data) - batch_size)
        return data[start_index:(start_index + batch_size)]


    X_train, X_test, y_train, y_test = mnist.train.images, mnist.test.images, mnist.train.labels, mnist.test.labels
    X_train, X_test = standard_scale( X_train, X_test)
    sda = SdA()
    sda.fit_pretraining(X_train)
    sda.fit_finetuning(X_train, y_train)
