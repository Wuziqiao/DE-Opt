import tensorflow.compat.v1 as tf
import time
import scipy
import warnings
warnings.filterwarnings("ignore")

from RatingMetrics import *


class DE_IAutoRec():
    def __init__(self, sess, num_user, num_item, lr_min, lr_max, reg_rate_min, reg_rate_max, learning_rate=0.00001, reg_rate=100, epoch=500, batch_size=64,
                 verbose=False, T=1, display_step=100, individual_num=5,
                 min_RMSE_DE=1e10, min_round=0, min_Error=1e10, delay_round=20, total_round=0):
        self.learning_rate = tf.Variable(learning_rate, )
        self.epochs = epoch
        self.batch_size = batch_size
        self.reg_rate = reg_rate
        self.sess = sess
        self.num_user = num_user
        self.num_item = num_item
        self.verbose = verbose
        self.T = T
        self.display_step = display_step

        self.individual_num = individual_num
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.reg_rate_min = reg_rate_min
        self.reg_rate_max = reg_rate_max
        self.min_RMSE_DE = min_RMSE_DE
        self.min_round = min_round
        self.min_Error = min_Error
        self.delay_round = delay_round
        self.total_round = total_round
        self.startTime = 0
        self.overTime = 0
        self.min_rmse_Error = 1e10
        self.min_mae_Error = 1e10
        self.time_count = 0
        print("DE_I-AutoRec.")

    def GenerateTrainVector(self, ID, maxID, lr_matrix, reg_rate_matrix):
        SFGSS = 8
        SFHC = 20
        Fl = 0.1
        Fu = 0.9
        tuo1 = 0.1
        tuo2 = 0.03
        tuo3 = 0.07

        Result = np.empty(shape=(2, 2))

        for i in range(2):
            u1 = ID
            u2 = ID
            u3 = ID
            while u1 == ID:
                u1 = np.random.randint(0, maxID)
            while (u2 == ID) or (u2 == u1):
                u2 = np.random.randint(0, maxID)
            while (u3 == ID) or (u3 == u2) or (u3 == u1):
                u3 = np.random.randint(0, maxID)

            rand1 = np.random.rand()
            rand2 = np.random.rand()
            rand3 = np.random.rand()
            F = np.random.rand()
            K = np.random.rand()

            if rand3 < tuo2:
                F = SFGSS
            elif tuo2 <= rand3 < tuo3:
                F = SFHC
            elif rand2 < tuo1 and rand3 > tuo3:
                F = Fl + Fu * rand1

            temp1 = lr_matrix[u2][i] - lr_matrix[u3][i]
            temp2 = temp1 * F
            temp_mutation = lr_matrix[u1][i] + temp2
            temp1 = temp_mutation - lr_matrix[ID][i]
            temp2 = temp1 * K
            Result[0][i] = lr_matrix[ID][i] + temp2

            temp1 = reg_rate_matrix[u2][i] - reg_rate_matrix[u3][i]
            temp2 = temp1 * F
            temp_mutation = reg_rate_matrix[u1][i] + temp2
            temp1 = temp_mutation - reg_rate_matrix[ID][i]
            temp2 = temp1 * K
            Result[1][i] = reg_rate_matrix[ID][i] + temp2

            if Result[0][i] <= self.lr_min:
                Result[0][i] = self.lr_min
            if Result[0][i] >= self.lr_max:
                Result[0][i] = self.lr_max
            if Result[1][i] <= self.reg_rate_min:
                Result[1][i] = self.reg_rate_min
            if Result[1][i] >= self.reg_rate_max:
                Result[1][i] = self.reg_rate_max

        return Result

    def build_network(self, hidden_neuron=500):

        self.rating_matrix = tf.placeholder(dtype=tf.float32, shape=[self.num_user, None])
        self.rating_matrix_mask = tf.placeholder(dtype=tf.float32, shape=[self.num_user, None])
        self.keep_rate_net = tf.placeholder(tf.float32)
        self.keep_rate_input = tf.placeholder(tf.float32)

        self.V = tf.Variable(tf.random_normal([hidden_neuron, self.num_user], stddev=0.01))
        self.W = tf.Variable(tf.random_normal([self.num_user, hidden_neuron], stddev=0.01))

        mu = tf.Variable(tf.random_normal([hidden_neuron], stddev=0.01))
        b = tf.Variable(tf.random_normal([self.num_user], stddev=0.01))
        layer_1 = tf.nn.dropout(tf.sigmoid(tf.expand_dims(mu, 1) + tf.matmul(self.V, self.rating_matrix)),
                                self.keep_rate_net)
        self.layer_2 = tf.matmul(self.W, layer_1) + tf.expand_dims(b, 1)


        # self.loss = tf.reduce_mean(tf.square(tf.norm(tf.multiply((self.rating_matrix - self.layer_2), self.rating_matrix_mask)))) + self.reg_rate * (tf.square(tf.norm(self.W)) + tf.square(tf.norm(self.V)))
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self, train_data):

        self.num_training = self.num_item
        total_batch = int(self.num_training / self.batch_size)
        idxs = np.random.permutation(self.num_training)  # shuffled ordering

        # print('lr:%.8f' % self.learning_rate, 'reg:%.8f' % self.reg_rate)

        self.loss = tf.reduce_mean(tf.square(
            tf.norm(tf.multiply((self.rating_matrix - self.layer_2), self.rating_matrix_mask)))) + self.reg_rate2 * (
                            tf.square(tf.norm(self.W)) + tf.square(tf.norm(self.V)))

        # variables_names = [v.name for v in tf.trainable_variables()]
        # values = self.sess.run(variables_names)
        # for k, v in zip(variables_names, values):
        #     print("Variables: ", k)
        #     print("Shape:", v.shape)
        #     print(v)
        # print(tf.trainable_variables())
        # layer_shape: 2, 2
        # Total_layer_nums: 2

        var1 = tf.trainable_variables()[0:2]
        var2 = tf.trainable_variables()[2:]
        # var3 = tf.trainable_variables()[214:327]
        # var4 = tf.trainable_variables()[327:435]
        # var5 = tf.trainable_variables()[435:543]


        opt1 = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate1).minimize(self.loss, var_list=var1)
        opt2 = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate2).minimize(self.loss, var_list=var2)
        # opt3 = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate3).minimize(self.loss, var_list=var3)
        # opt4 = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate4).minimize(self.loss, var_list=var4)
        # opt5 = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate5).minimize(self.loss, var_list=var5)
        self.optimizer = tf.group(opt1, opt2)
        # self.optimizer = tf.group(opt1, opt2, opt3, opt4, opt5)

        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

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
        self.train_data = self._data_process(train_data)
        self.train_data_mask = scipy.sign(self.train_data)

        self.lr_matrix = np.empty(shape=(self.individual_num, 2))
        self.reg_rate_matrix = np.empty(shape=(self.individual_num, 2))
        print('lr min:', self.lr_min, ' lr max:', self.lr_max, ' reg rate min:', self.reg_rate_min, ' reg rate max:', self.reg_rate_max)

        for i in range(self.individual_num):
            # layer_nums = 2
            for j in range(2):
                xx = self.lr_min + np.random.rand() * (self.lr_max - self.lr_min)
                yy = self.reg_rate_min + np.random.rand() * (self.reg_rate_max - self.reg_rate_min)
                self.lr_matrix[i][j] = xx
                self.reg_rate_matrix[i][j] = yy



        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.startTime = time.time()

        for epoch in range(self.epochs):
            time1 = time.time()
            rmse_no_de, tmp = self.validation(validation_data)
            rmse = -1
            mae = -1
            lr1 = 0
            lr2 = 0
            lr3 = 0
            lr4 = 0
            lr5 = 0
            reg1 = 0
            reg2 = 0
            reg3 = 0
            reg4 = 0
            reg5 = 0
            for ID in range(self.individual_num):
                evolution = self.GenerateTrainVector(ID, self.individual_num, self.lr_matrix, self.reg_rate_matrix)
                # Learning rate && Regularization coefficient in each layer
                self.learning_rate1 = evolution[0][0]
                self.reg_rate1 = evolution[1][0]
                self.learning_rate2 = evolution[0][1]
                self.reg_rate2 = evolution[1][1]
                # self.learning_rate3 = evolution[0][2]
                # self.reg_rate3 = evolution[1][2]
                # self.learning_rate4 = evolution[0][3]
                # self.reg_rate4 = evolution[1][3]
                # self.learning_rate5 = evolution[0][4]
                # self.reg_rate5 = evolution[1][4]

                if ID == 0:
                    lr1 = evolution[0][0]
                    reg1 = evolution[1][0]
                if ID == 1:
                    lr2 = evolution[0][0]
                    reg2 = evolution[1][0]
                if ID == 2:
                    lr3 = evolution[0][0]
                    reg3 = evolution[1][0]
                if ID == 3:
                    lr4 = evolution[0][0]
                    reg4 = evolution[1][0]
                if ID == 4:
                    lr5 = evolution[0][0]
                    reg5 = evolution[1][0]

                self.train(train_data)
                rmse_de, tmp = self.validation(validation_data)
                if rmse_de <= rmse_no_de:
                    self.lr_matrix[ID][0] = evolution[0][0]
                    self.reg_rate_matrix[ID][0] = evolution[1][0]
                    rmse_no_de = rmse_de
                if ID == self.individual_num-1:
                    rmse = rmse_de
                    mae = tmp

            # rmse, mae = self.validation(validation_data)
            time2 = time.time()
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

            self.time_count += (time2 - time1)
            print("Epoch: %04d; " % (epoch + 1), "RMSE:%.6f" % rmse + "; MAE:%.6f" % mae, ' time per Epoch: %.4f' %
                  (time2 - time1), 's')
            # print("Epoch: %04d " % (epoch + 1), lr1, reg1)
            # print("Epoch: %04d " % (epoch + 1), lr2, reg2)
            # print("Epoch: %04d " % (epoch + 1), lr3, reg3)
            # print("Epoch: %04d " % (epoch + 1), lr4, reg4)
            # print("Epoch: %04d " % (epoch + 1), lr5, reg5)

        print('Result on testset:\n', ' RMSE:%.6f' % self.min_rmse_Error, 'MAE:%.6f' % self.min_mae_Error,
              'time_per_epoch:%.4f' % (self.time_count / 2 * self.total_round),
              ' train total round:', self.total_round, ' time cost all: %.4f' % (self.overTime - self.startTime))

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def predict(self, user_id, item_id):
        return self.reconstruction[user_id, item_id]

    def _data_process(self, data):
        output = np.zeros((self.num_user, self.num_item))
        for u in range(self.num_user):
            for i in range(self.num_item):
                output[u, i] = data.get((u, i))
        return output


