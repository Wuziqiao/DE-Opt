import tensorflow.compat.v1 as tf
import time
import scipy
import warnings
warnings.filterwarnings("ignore")

from RatingMetrics import *


class IAutoRec():
    def __init__(self, sess, num_user, num_item, learning_rate, reg_rate, epoch=2000, batch_size=64,
                 verbose=False, T=1, display_step=1000, min_RMSE_DE=1e10, min_round=0, min_rmse_Error=1e10, min_mae_Error=1e10,
                 delay_round=10, total_round=0):
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.reg_rate = reg_rate
        self.sess = sess
        self.num_user = num_user
        self.num_item = num_item
        self.verbose = verbose
        self.T = T
        self.display_step = display_step
        self.min_RMSE_DE = min_RMSE_DE
        self.min_round = min_round
        self.min_rmse_Error = min_rmse_Error
        self.min_mae_Error = min_mae_Error
        self.delay_round = delay_round
        self.total_round = total_round
        self.startTime = 0
        self.overTime = 0
        self.time_count = 0
        print("I-AutoRec.")

    def build_network(self, hidden_neuron=500):

        self.rating_matrix = tf.placeholder(dtype=tf.float32, shape=[self.num_user, None])
        self.rating_matrix_mask = tf.placeholder(dtype=tf.float32, shape=[self.num_user, None])
        self.keep_rate_net = tf.placeholder(tf.float32)
        self.keep_rate_input = tf.placeholder(tf.float32)

        V = tf.Variable(tf.random_normal([hidden_neuron, self.num_user], stddev=0.01))
        W = tf.Variable(tf.random_normal([self.num_user, hidden_neuron], stddev=0.01))

        mu = tf.Variable(tf.random_normal([hidden_neuron], stddev=0.01))
        b = tf.Variable(tf.random_normal([self.num_user], stddev=0.01))
        layer_1 = tf.nn.dropout(tf.sigmoid(tf.expand_dims(mu, 1) + tf.matmul(V, self.rating_matrix)),
                                self.keep_rate_net)
        self.layer_2 = tf.matmul(W, layer_1) + tf.expand_dims(b, 1)
        self.loss = tf.reduce_mean(tf.square(
            tf.norm(tf.multiply((self.rating_matrix - self.layer_2), self.rating_matrix_mask)))) + self.reg_rate * (
                            tf.square(tf.norm(W)) + tf.square(tf.norm(V)))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)



    def train(self, train_data):
        self.num_training = self.num_item
        total_batch = int(self.num_training / self.batch_size)
        idxs = np.random.permutation(self.num_training)  # shuffled ordering

        for i in range(total_batch):
            if i == total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size:]
            elif i < total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size: (i + 1) * self.batch_size]

            _, loss = self.sess.run([self.optimizer, self.loss],
                                    feed_dict={self.rating_matrix: self.train_data[:, batch_set_idx],
                                               self.rating_matrix_mask: self.train_data_mask[:, batch_set_idx],
                                               self.keep_rate_net: 0.95
                                               })
            # if i % self.display_step == 0:
            #     if self.verbose:
            #
            #         print("Index: %04d; cost= %.9f" % (i + 1, np.mean(loss)))
            #         print("one iteration: %s seconds." % (time.time() - start_time))

    def test(self, test_data):
        self.reconstruction = self.sess.run(self.layer_2, feed_dict={self.rating_matrix: self.train_data,
                                                                     self.rating_matrix_mask: self.train_data_mask,
                                                                     self.keep_rate_net: 1})
        error = 0
        error_mae = 0
        test_set = list(test_data.keys())
        for (u, i) in test_set:
            pred_rating_test = self.predict(u, i)
            error += (float(test_data.get((u, i))) - pred_rating_test) ** 2
            error_mae += (np.abs(float(test_data.get((u, i))) - pred_rating_test))
            rmse = RMSE(error, len(test_set))
            mae = MAE(error_mae, len(test_set))
        return rmse, mae

    def validation(self, validation_data):
        self.reconstruction = self.sess.run(self.layer_2, feed_dict={self.rating_matrix: self.train_data,
                                                                     self.rating_matrix_mask: self.train_data_mask,
                                                                     self.keep_rate_net: 1})
        error = 0
        error_mae = 0
        validation_set = list(validation_data.keys())
        for (u, i) in validation_set:
            pred_rating_validation = self.predict(u, i)
            error += (float(validation_data.get((u, i))) - pred_rating_validation) ** 2
            error_mae += (np.abs(float(validation_data.get((u, i))) - pred_rating_validation))
            rmse = RMSE(error, len(validation_set))
            mae = MAE(error_mae, len(validation_set))
        return rmse, mae

    def execute(self, train_data, validation_data, test_data):
        # print("参数设置： ", 'learning_rate: %.6f' %(self.learning_rate), ' ;reg_rate: %.6f' % (self.reg_rate))
        # self.train_data = self._data_process(train_data)
        # self.train_data_mask = scipy.sign(self.train_data)
        # init = tf.global_variables_initializer()
        # self.startTime = time.time()
        # self.sess.run(init)
        self.train_data = self._data_process(train_data)
        self.train_data_mask = scipy.sign(self.train_data)

        init = tf.global_variables_initializer()
        self.sess.run(init)


        self.startTime = time.time()
        print("参数设置： ", 'learning_rate: %.6f' % self.learning_rate, ' ;reg_rate: %.6f' % self.reg_rate)
        for epoch in range(self.epochs):
            time1 = time.time()
            self.train(train_data)
            rmse, mae = self.validation(validation_data)

            curErr = rmse
            if self.min_RMSE_DE > curErr:
                self.min_RMSE_DE = curErr
                self.min_round = epoch + 1

            else:
                rmse_test, mae_test = self.test(test_data)
                if self.min_rmse_Error > rmse_test:
                    self.min_rmse_Error = rmse_test
                    self.min_mae_Error = mae_test
                    self.overTime = time.time()
                    self.total_round = epoch
            if (epoch - self.min_round) >= self.delay_round:
                break
            if epoch == 1999:
                rmse_test, mae_test = self.test(test_data)
                self.min_rmse_Error = rmse_test
                self.min_mae_Error = mae_test
                self.overTime = time.time()
                self.total_round = epoch
                break
            time2 = time.time()
            self.time_count += (time2 - time1)
            print("Epoch: %04d; " % (epoch+1), "RMSE:%.6f" % rmse + "; MAE:%.6f" % mae, ' time per Epoch: %.4f' %
                  (time2 - time1), 's')

        if self.total_round == 0:
            self.total_round = 1
        print('Result on testset:\n', ' RMSE:%.6f' % self.min_rmse_Error, 'MAE:%.6f' % self.min_mae_Error, 'time_per_epoch:%.4f' % (self.time_count / (self.total_round + self.delay_round)),
              ' train total round:', self.total_round, ' time cost all: %.4f' % (self.overTime - self.startTime), 's')




    # def save(self, path):
    #     saver = tf.train.Saver()
    #     saver.save(self.sess, path)

    def predict(self, user_id, item_id):
        return self.reconstruction[user_id, item_id]

    def _data_process(self, data):
        output = np.zeros((self.num_user, self.num_item))
        for u in range(self.num_user):
            for i in range(self.num_item):
                output[u, i] = data.get((u, i))
        return output


class UAutoRec():
    def __init__(self, sess, num_user, num_item, learning_rate=0.0001, reg_rate=0.01, epoch=500, batch_size=64,
                 verbose=False, T=3, display_step=1000):
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.reg_rate = reg_rate
        self.sess = sess
        self.num_user = num_user
        self.num_item = num_item
        self.verbose = verbose
        self.T = T
        self.display_step = display_step
        print("U-AutoRec.")

    def build_network(self, hidden_neuron=500):

        self.rating_matrix = tf.placeholder(dtype=tf.float32, shape=[self.num_item, None])
        self.rating_matrix_mask = tf.placeholder(dtype=tf.float32, shape=[self.num_item, None])

        V = tf.Variable(tf.random_normal([hidden_neuron, self.num_item], stddev=0.01))
        W = tf.Variable(tf.random_normal([self.num_item, hidden_neuron], stddev=0.01))

        mu = tf.Variable(tf.random_normal([hidden_neuron], stddev=0.01))
        b = tf.Variable(tf.random_normal([self.num_item], stddev=0.01))
        layer_1 = tf.sigmoid(tf.expand_dims(mu, 1) + tf.matmul(V, self.rating_matrix))
        self.layer_2 = tf.matmul(W, layer_1) + tf.expand_dims(b, 1)

        self.loss = tf.reduce_mean(tf.square(
            tf.norm(tf.multiply((self.rating_matrix - self.layer_2), self.rating_matrix_mask)))) + self.reg_rate * (
        tf.square(tf.norm(W)) + tf.square(tf.norm(V)))

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self, train_data):

        self.num_training = self.num_user
        total_batch = int(self.num_training / self.batch_size)
        idxs = np.random.permutation(self.num_training)  # shuffled ordering

        for i in range(total_batch):
            start_time = time.time()
            if i == total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size:]
            elif i < total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size: (i + 1) * self.batch_size]

            _, loss = self.sess.run([self.optimizer, self.loss],
                                    feed_dict={self.rating_matrix: self.train_data[:, batch_set_idx],
                                               self.rating_matrix_mask: self.train_data_mask[:, batch_set_idx]
                                               })
            # if self.verbose and i % self.display_step == 0:
            #     print("Index: %04d; cost= %.9f" % (i + 1, np.mean(loss)))
            #     if self.verbose:
            #         print("one iteration: %s seconds." % (time.time() - start_time))

    def test(self, test_data):
        self.reconstruction = self.sess.run(self.layer_2, feed_dict={self.rating_matrix: self.train_data,
                                                                     self.rating_matrix_mask:
                                                                         self.train_data_mask})
        error = 0
        error_mae = 0
        test_set = list(test_data.keys())
        for (u, i) in test_set:
            pred_rating_test = self.predict(u, i)
            error += (float(test_data.get((u, i))) - pred_rating_test) ** 2
            error_mae += (np.abs(float(test_data.get((u, i))) - pred_rating_test))
        print("RMSE:" + str(RMSE(error, len(test_set))) + "; MAE:" + str(MAE(error_mae, len(test_set))), end='')

    def execute(self, train_data, test_data, validation_data):
        self.train_data = self._data_process(train_data.transpose())
        self.train_data_mask = scipy.sign(self.train_data)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        for epoch in range(self.epochs):
            time_start = time.time()
            print("Epoch: %04d; " % (epoch), end='')
            self.train(train_data)
            self.test(test_data)
            time_end = time.time()
            print('；timeCost_perEpoch:', time_end - time_start)

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def predict(self, user_id, item_id):
        return self.reconstruction[item_id, user_id]

    def _data_process(self, data):
        output = np.zeros((self.num_item, self.num_user))
        for u in range(self.num_user):
            for i in range(self.num_item):
                output[i, u] = data.get((i, u))
        return output
