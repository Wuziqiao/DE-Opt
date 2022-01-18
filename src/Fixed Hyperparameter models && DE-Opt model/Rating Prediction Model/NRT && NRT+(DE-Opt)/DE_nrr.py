#!/usr/bin/env python
"""Implementation of Neural Rating Regression.
Reference: Piji Li, Zihao Wang, Zhaochun Ren, Lidong Bing, Wai Lam. "Neural Rating Regression with Abstractive Tips Generation for Recommendation
Authors." https://arxiv.org/pdf/1708.00154.pdf
"""




import tensorflow.compat.v1 as tf
import time
import numpy as np

from RatingMetrics import *



class DE_NRR():
    def __init__(self, sess, num_user, num_item, lr_min, lr_max, reg_rate_min, reg_rate_max, learning_rate=0.001, reg_rate=0.01, epoch=50, batch_size=256,
                 show_time=False, T=1, display_step=1000, individual_num=5,
                 min_RMSE_DE=1e10, min_round=0, min_RMSE_Error=1e10, min_MAE_Error=1e10,delay_round=20, total_round=0):
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.reg_rate = reg_rate
        self.sess = sess
        self.num_user = num_user
        self.num_item = num_item
        self.show_time = show_time
        self.T = T
        self.display_step = display_step

        self.individual_num = individual_num
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.reg_rate_min = reg_rate_min
        self.reg_rate_max = reg_rate_max
        self.min_RMSE_DE = min_RMSE_DE
        self.min_round = min_round
        self.min_RMSE_Error = min_RMSE_Error
        self.min_MAE_Error = min_MAE_Error
        self.delay_round = delay_round
        self.total_round = total_round
        self.overTime = 0
        print("DE_NRR.")

    def GenerateTrainVector(self, ID, maxID, lr_matrix, reg_rate_matrix):
        SFGSS = 8
        SFHC = 20
        Fl = 0.1
        Fu = 0.9
        tuo1 = 0.1
        tuo2 = 0.03
        tuo3 = 0.07

        Result = np.empty(shape=(2, 5))

        for i in range(5):
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

    def build_network(self, num_factor_user=40, num_factor_item=40, d=50, hidden_dimension=40):

        self.reg_rate1 = self.reg_rate_min + np.random.rand() * (self.reg_rate_max - self.reg_rate_min)
        self.reg_rate2 = self.reg_rate_min + np.random.rand() * (self.reg_rate_max - self.reg_rate_min)
        self.reg_rate3 = self.reg_rate_min + np.random.rand() * (self.reg_rate_max - self.reg_rate_min)
        self.reg_rate4 = self.reg_rate_min + np.random.rand() * (self.reg_rate_max - self.reg_rate_min)
        self.reg_rate5 = self.reg_rate_min + np.random.rand() * (self.reg_rate_max - self.reg_rate_min)

        # model dependent arguments
        self.user_id = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
        self.item_id = tf.placeholder(dtype=tf.int32, shape=[None], name='item_id')
        self.y = tf.placeholder("float", [None], 'rating')

        self.U = tf.Variable(tf.random_normal([self.num_user, num_factor_user], stddev=0.01))
        self.V = tf.Variable(tf.random_normal([self.num_item, num_factor_item], stddev=0.01))
        self.b = tf.Variable(tf.random_normal([d]))

        user_latent_factor = tf.nn.embedding_lookup(self.U, self.user_id)
        item_latent_factor = tf.nn.embedding_lookup(self.V, self.item_id)

        self.W_User = tf.Variable(tf.random_normal([num_factor_user, d], stddev=0.01))
        self.W_Item = tf.Variable(tf.random_normal([num_factor_item, d], stddev=0.01))

        input = tf.matmul(user_latent_factor, self.W_User) + tf.matmul(item_latent_factor, self.W_Item) + self.b

        layer_1 = tf.layers.dense(inputs=input, units=d, bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer, activation=tf.sigmoid,
                                  kernel_regularizer=tf.keras.regularizers.l2(self.reg_rate1))
        layer_2 = tf.layers.dense(inputs=layer_1, units=hidden_dimension, activation=tf.sigmoid,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=tf.keras.regularizers.l2(self.reg_rate2))
        layer_3 = tf.layers.dense(inputs=layer_2, units=hidden_dimension, activation=tf.sigmoid,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=tf.keras.regularizers.l2(self.reg_rate3))
        layer_4 = tf.layers.dense(inputs=layer_3, units=hidden_dimension, activation=tf.sigmoid,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=tf.keras.regularizers.l2(self.reg_rate4))
        output = tf.layers.dense(inputs=layer_4, units=1, activation=None,
                                 bias_initializer=tf.random_normal_initializer,
                                 kernel_initializer=tf.random_normal_initializer,
                                 kernel_regularizer=tf.keras.regularizers.l2(self.reg_rate5))
        self.pred_rating = tf.reshape(output, [-1])

        # print(np.shape(output))
        reg_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        # self.loss = tf.reduce_sum(tf.square(self.y - self.pred_rating)) \
        #             + tf.losses.get_regularization_loss() + self.reg_rate * (
        # tf.norm(U) + tf.norm(V) + tf.norm(b) + tf.norm(W_Item) + tf.norm(W_User))
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        return self

    def train(self, train_data):
        # print('lr:', self.learning_rate, 'reg:', self.reg_rate)
        self.num_training = len(self.rating)
        total_batch = int(self.num_training / self.batch_size)
        idxs = np.random.permutation(self.num_training)  # shuffled ordering
        user_random = list(self.user[idxs])
        item_random = list(self.item[idxs])
        rating_random = list(self.rating[idxs])

        self.loss = tf.reduce_sum(tf.square(self.y - self.pred_rating)) \
                    + tf.losses.get_regularization_loss() + self.reg_rate * (
                            tf.norm(self.U) + tf.norm(self.V) + tf.norm(self.b) + tf.norm(self.W_Item) + tf.norm(
                        self.W_User))

        # print(tf.trainable_variables())

        # Total_layer_nums: 5

        var1 = tf.trainable_variables()[0:2]
        var2 = tf.trainable_variables()[2:4]
        var3 = tf.trainable_variables()[4:6]
        var4 = tf.trainable_variables()[6:8]
        var5 = tf.trainable_variables()[8:]

        opt1 = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate1).minimize(self.loss, var_list=var1)
        opt2 = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate2).minimize(self.loss, var_list=var2)
        opt3 = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate3).minimize(self.loss, var_list=var3)
        opt4 = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate4).minimize(self.loss, var_list=var4)
        opt5 = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate5).minimize(self.loss, var_list=var5)

        self.optimizer = tf.group(opt1, opt2, opt3, opt4, opt5)
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # train
        for i in range(total_batch):
            start_time = time.time()
            batch_user = user_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = item_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_rating = rating_random[i * self.batch_size:(i + 1) * self.batch_size]

            _, loss = self.sess.run([self.optimizer, self.loss], feed_dict={self.user_id: batch_user,
                                                                            self.item_id: batch_item,
                                                                            self.y: batch_rating
                                                                            })
            # if i % self.display_step == 0:
            #     print("Index: %04d; cost= %.9f" % (i + 1, np.mean(loss)))
            #     if self.show_time:
            #         print("one iteration: %s seconds." % (time.time() - start_time))

    def test(self, test_data):
        error = 0
        error_mae = 0
        test_set = list(test_data.keys())
        for (u, i) in test_set:
            pred_rating_test = self.predict([u], [i], False)
            error += (float(test_data.get((u, i))) - pred_rating_test) ** 2
            error_mae += (np.abs(float(test_data.get((u, i))) - pred_rating_test))
            rmse = RMSE(error, len(test_set))
            mae = MAE(error_mae, len(test_set))
        return rmse, mae

    def validation(self, validation_data):
        error = 0
        error_mae = 0
        validation_set = list(validation_data.keys())
        for (u, i) in validation_set:
            pred_rating_validation = self.predict([u], [i], False)
            error += (float(validation_data.get((u, i))) - pred_rating_validation) ** 2
            error_mae += (np.abs(float(validation_data.get((u, i))) - pred_rating_validation))
            rmse = RMSE(error, len(validation_set))
            mae = MAE(error_mae, len(validation_set))
        return rmse, mae

    def execute(self, train_data, validation_data, test_data):
        print('lr min:', self.lr_min, ' lr max:', self.lr_max, ' reg rate min:', self.reg_rate_min, ' reg rate max:',
              self.reg_rate_max)
        t = train_data.tocoo()
        self.user = t.row.reshape(-1)
        self.item = t.col.reshape(-1)
        self.rating = t.data
        self.lr_matrix = np.empty(shape=(self.individual_num, 5))
        self.reg_rate_matrix = np.empty(shape=(self.individual_num, 5))

        for i in range(self.individual_num):
            for j in range(5):
                xx = self.lr_min + np.random.rand() * (self.lr_max - self.lr_min)
                yy = self.reg_rate_min + np.random.rand() * (self.reg_rate_max - self.reg_rate_min)
                self.lr_matrix[i][j] = xx
                self.reg_rate_matrix[i][j] = yy

        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.startTime = time.time()

        for epoch in range(self.epochs):
            t1 = time.time()
            rmse_no_de, tmp = self.validation(validation_data)
            # print(rmse_no_de)
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
                self.learning_rate1 = evolution[0][0]
                self.learning_rate2 = evolution[0][1]
                self.learning_rate3 = evolution[0][2]
                self.learning_rate4 = evolution[0][3]
                self.learning_rate5 = evolution[0][4]
                self.reg_rate1 = evolution[1][0]
                self.reg_rate2 = evolution[1][1]
                self.reg_rate3 = evolution[1][2]
                self.reg_rate4 = evolution[1][3]
                self.reg_rate5 = evolution[1][4]

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
                # print(rmse_de)
                if rmse_de <= rmse_no_de:
                    self.lr_matrix[ID][0] = evolution[0][0]
                    self.lr_matrix[ID][1] = evolution[0][1]
                    self.lr_matrix[ID][2] = evolution[0][2]
                    self.lr_matrix[ID][3] = evolution[0][3]
                    self.lr_matrix[ID][4] = evolution[0][4]
                    self.reg_rate_matrix[ID][0] = evolution[1][0]
                    self.reg_rate_matrix[ID][1] = evolution[1][1]
                    self.reg_rate_matrix[ID][2] = evolution[1][2]
                    self.reg_rate_matrix[ID][3] = evolution[1][3]
                    self.reg_rate_matrix[ID][4] = evolution[1][4]
                    rmse_no_de = rmse_de
                if ID == self.individual_num-1:
                    rmse = rmse_de
                    mae = tmp

            t2 = time.time()
            curErr = rmse
            if self.min_RMSE_DE > curErr:
                self.min_RMSE_DE = curErr
                self.min_round = epoch + 1
            else:
                rmse_test, mae_test = self.test(test_data)
                if self.min_RMSE_Error > rmse_test:
                    self.min_RMSE_Error = rmse_test
                    self.min_MAE_Error = mae_test
                    self.overTime = time.time()
                    self.total_round = epoch
            if (epoch - self.min_round) >= self.delay_round:
                break

            print("Epoch: %04d; " % (epoch + 1), "RMSE:%.6f" % rmse + "; MAE:%.6f" % mae, ' time per Epoch: %.4f' %
                  (t2 - t1), 's')
            # print("Epoch: %04d " % (epoch + 1), lr1, reg1)
            # print("Epoch: %04d " % (epoch + 1), lr2, reg2)
            # print("Epoch: %04d " % (epoch + 1), lr3, reg3)
            # print("Epoch: %04d " % (epoch + 1), lr4, reg4)
            # print("Epoch: %04d " % (epoch + 1), lr5, reg5)
        print('Result on testset:\n', ' RMSE:%.6f' % self.min_RMSE_Error, 'MAE:%.6f' % self.min_MAE_Error,
                  ' train total round:', self.total_round, ' time cost all: %.4f' % (self.overTime - self.startTime))


    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def predict(self, user_id, item_id, prt=True):
        score = self.sess.run([self.pred_rating], feed_dict={self.user_id: user_id, self.item_id: item_id})[0]
        if prt:
            print(score)
        return score
