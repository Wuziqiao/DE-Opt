import tensorflow.compat.v1 as tf
import time
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

from RankingMetrics import *

class DE_NeuMF(object):
    def __init__(self, sess, num_user, num_item, lr_min, lr_max, reg_rate_min, reg_rate_max, learning_rate=0.5, reg_rate=0.01, epoch=500, batch_size=8192,
                 verbose=False, t=1, display_step=1000, individual_num=5,
                  min_round=0, delay_round=20, total_round=0):
        self.sess = sess
        self.num_user = num_user
        self.num_item = num_item
        self.learning_rate = learning_rate
        self.reg_rate = reg_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.verbose = verbose
        self.T = t
        self.display_step = display_step

        self.num_neg_sample = None
        self.user_id = None
        self.item_id = None
        self.y = None
        self.P = None
        self.Q = None
        self.mlp_P = None
        self.mlp_Q = None
        self.pred_y = None
        self.loss = None
        self.optimizer = None
        self.test_data = None
        self.validation_data = None
        self.user = None
        self.item = None
        self.label = None
        self.neg_items = None
        self.test_users = None
        self.validation_users = None
        self.num_training = None
        self.total_batch = None

        self.individual_num = individual_num
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.reg_rate_min = reg_rate_min
        self.reg_rate_max = reg_rate_max
        self.max_ndcg10_DE = -1
        self.ndcg_10 = 0
        self.ndcg_20 = 0
        self.recall_10 = 0
        self.recall_20 = 0


        self.min_round = min_round
        self.delay_round = delay_round
        self.total_round = total_round

        print("DE_NeuMF.")

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

    def build_network(self, num_factor=10, num_factor_mlp=64, hidden_dimension=10, num_neg_sample=30):

        self.reg_rate1 = self.reg_rate_min + np.random.rand() * (self.reg_rate_max - self.reg_rate_min)
        self.reg_rate2 = self.reg_rate_min + np.random.rand() * (self.reg_rate_max - self.reg_rate_min)
        self.reg_rate3 = self.reg_rate_min + np.random.rand() * (self.reg_rate_max - self.reg_rate_min)
        self.reg_rate4 = self.reg_rate_min + np.random.rand() * (self.reg_rate_max - self.reg_rate_min)
        self.reg_rate5 = self.reg_rate_min + np.random.rand() * (self.reg_rate_max - self.reg_rate_min)

        self.num_neg_sample = num_neg_sample
        self.user_id = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
        self.item_id = tf.placeholder(dtype=tf.int32, shape=[None], name='item_id')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None], name='y')

        self.P = tf.Variable(tf.random_normal([self.num_user, num_factor]), dtype=tf.float32)
        self.Q = tf.Variable(tf.random_normal([self.num_item, num_factor]), dtype=tf.float32)

        self.mlp_P = tf.Variable(tf.random_normal([self.num_user, num_factor_mlp]), dtype=tf.float32)
        self.mlp_Q = tf.Variable(tf.random_normal([self.num_item, num_factor_mlp]), dtype=tf.float32)

        user_latent_factor = tf.nn.embedding_lookup(self.P, self.user_id)
        item_latent_factor = tf.nn.embedding_lookup(self.Q, self.item_id)
        mlp_user_latent_factor = tf.nn.embedding_lookup(self.mlp_P, self.user_id)
        mlp_item_latent_factor = tf.nn.embedding_lookup(self.mlp_Q, self.item_id)

        _GMF = tf.multiply(user_latent_factor, item_latent_factor)

        layer_1 = tf.layers.dense(
            inputs=tf.concat([mlp_item_latent_factor, mlp_user_latent_factor], axis=1),
            units=num_factor_mlp * 2,
            kernel_initializer=tf.random_normal_initializer,
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(self.reg_rate1))

        layer_2 = tf.layers.dense(
            inputs=layer_1,
            units=hidden_dimension * 8,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(self.reg_rate2))

        layer_3 = tf.layers.dense(
            inputs=layer_2,
            units=hidden_dimension * 4,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(self.reg_rate3))

        layer_4 = tf.layers.dense(
            inputs=layer_3,
            units=hidden_dimension * 2,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(self.reg_rate4))

        _MLP = tf.layers.dense(
                inputs=layer_4,
                units=hidden_dimension,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer,
                kernel_regularizer=tf.keras.regularizers.l2(self.reg_rate5))

        self.pred_y = tf.nn.sigmoid(tf.reduce_sum(tf.concat([_GMF, _MLP], axis=1), 1))

        # self.pred_y = tf.layers.dense(
        #     inputs=tf.concat([_GMF, _MLP], axis=1), units=1, activation=tf.sigmoid,
        #     kernel_initializer=tf.random_normal_initializer,
        #     kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=self.reg_rate))

        # -{y.log(p{y=1}) + (1-y).log(1 - p{y=1})} + {regularization loss...}
        # self.loss = - tf.reduce_sum(
        #     self.y * tf.log(self.pred_y + 1e-10) + (1 - self.y) * tf.log(1 - self.pred_y + 1e-10)) + \
        #     tf.losses.get_regularization_loss() + \
        #     self.reg_rate * (tf.nn.l2_loss(self.P) + tf.nn.l2_loss(self.Q) +
        #                      tf.nn.l2_loss(self.mlp_P) + tf.nn.l2_loss(self.mlp_Q))
        #
        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        return self

    def prepare_data(self, train_data, validation_data, test_data):
        """
        You must prepare the data before train and test the model.

        :param train_data:
        :param test_data:
        :param validation_data:
        :return:
        """
        t = train_data.tocoo()
        self.user = list(t.row.reshape(-1))
        self.item = list(t.col.reshape(-1))
        self.label = list(t.data)
        self.validation_data = validation_data
        self.test_data = test_data

        self.neg_items = self._get_neg_items(train_data.tocsr())
        self.validation_users = set([u for u in self.validation_data.keys() if len(self.validation_data[u]) > 0])
        self.test_users = set([u for u in self.test_data.keys() if len(self.test_data[u]) > 0])

        print("data preparation finished.")
        return self

    def train(self):
        item_temp = self.item[:]
        user_temp = self.user[:]
        labels_temp = self.label[:]

        user_append = []
        item_append = []
        values_append = []
        for u in self.user:
            list_of_random_items = random.sample(self.neg_items[u], self.num_neg_sample)
            user_append += [u] * self.num_neg_sample
            item_append += list_of_random_items
            values_append += [0] * self.num_neg_sample

        item_temp += item_append
        user_temp += user_append
        labels_temp += values_append

        self.num_training = len(item_temp)
        self.total_batch = int(self.num_training / self.batch_size)
        # print(self.total_batch)
        idxs = np.random.permutation(self.num_training)  # shuffled ordering
        user_random = list(np.array(user_temp)[idxs])
        item_random = list(np.array(item_temp)[idxs])
        labels_random = list(np.array(labels_temp)[idxs])
        # train_test
        for i in range(self.total_batch):
            start_time = time.time()
            batch_user = user_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = item_random[i * self.batch_size:(i + 1) * self.batch_size]



        # print(tf.trainable_variables())

        self.loss = - tf.reduce_sum(
            self.y * tf.log(self.pred_y + 1e-10) + (1 - self.y) * tf.log(1 - self.pred_y + 1e-10)) + \
                    tf.losses.get_regularization_loss() + \
                    self.reg_rate * (tf.nn.l2_loss(self.P) + tf.nn.l2_loss(self.Q) +
                                     tf.nn.l2_loss(self.mlp_P) + tf.nn.l2_loss(self.mlp_Q))

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


        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        # train
        for i in range(self.total_batch):
            start_time = time.time()
            batch_user = user_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = item_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_label = labels_random[i * self.batch_size:(i + 1) * self.batch_size]

            _, loss = self.sess.run((self.optimizer, self.loss),
                                    feed_dict={self.user_id: batch_user, self.item_id: batch_item, self.y: batch_label})

            if i % self.display_step == 0:
                if self.verbose:
                    print("Index: %04d; cost= %.9f" % (i + 1, np.mean(loss)))
                    print("one iteration: %s seconds." % (time.time() - start_time))

    def test(self):
        Recall_5, Recall_10, ndcg, ndcg_5, ndcg_10 = evaluate_test(self)
        return Recall_5, Recall_10, ndcg, ndcg_5, ndcg_10

    def validation(self):
        Recall_5, Recall_10, ndcg, ndcg_5, ndcg_10 = evaluate_validation(self)
        return Recall_5, Recall_10, ndcg, ndcg_5, ndcg_10

    def execute(self, train_data, validation_data, test_data):
        self.prepare_data(train_data, validation_data, test_data)
        self.lr_matrix = np.empty(shape=(self.individual_num, 5))
        self.reg_rate_matrix = np.empty(shape=(self.individual_num, 5))
        print('lr min:', self.lr_min, ' lr max:', self.lr_max, ' reg rate min:', self.reg_rate_min, ' reg rate max:',
              self.reg_rate_max)

        for i in range(self.individual_num):
            for j in range(5):
                xx = self.lr_min + np.random.rand() * (self.lr_max - self.lr_min)
                yy = self.reg_rate_min + np.random.rand() * (self.reg_rate_max - self.reg_rate_min)
                self.lr_matrix[i][j] = xx
                self.reg_rate_matrix[i][j] = yy

        init = tf.global_variables_initializer()
        self.sess.run(init)
        startTime = time.time()

        for epoch in range(self.epochs):
            t1 = time.time()
            r10, r20, n, n10_no_de, n20 = self.validation()
            ndcg = -1
            recall = -1
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
                    lr01 = evolution[0][0]
                    lr02 = evolution[0][1]
                    lr03 = evolution[0][2]
                    lr04 = evolution[0][3]
                    lr05 = evolution[0][4]
                    reg01 = evolution[1][0]
                    reg02 = evolution[1][1]
                    reg03 = evolution[1][2]
                    reg04 = evolution[1][3]
                    reg05 = evolution[1][4]
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
                self.train()
                r10, r20, n, n10_de, n20 = self.validation()
                if n10_de >= n10_no_de:
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
                    n10_no_de = n10_de
                if ID == self.individual_num-1:
                    ndcg = n10_de
                    recall = r10

            t2 = time.time()
            curResult = ndcg
            if curResult > self.max_ndcg10_DE:
                self.max_ndcg10_DE = curResult
                self.min_round = epoch + 1
            else:
                r10_test, r20_test, n_test, n10_test, n20_test = self.test()
                if n10_test > self.ndcg_10:
                    self.ndcg_10 = n10_test
                    self.ndcg_20 = n20_test
                    self.recall_10 = r10_test
                    self.recall_20 = r20_test
                    self.overTime = time.time()
                    self.total_round = epoch
            if (epoch - self.min_round) >= self.delay_round:
                break

            # print("Epoch: %04d: " % (epoch), " Time Cost Per Epoch:"+ str(time2 - time1) + "s")
            #
            # print("Recall@5: " + str(r5) + " ; Recall@10: " + str(r10))
            #
            # print("ndcg: " + str(n) + " ; ndcg@5: " + str(n5) + " ; ndcg@10: " + str(n10_no_de))
            # print("Epoch: %04d " % (epoch + 1), lr1, reg1)
            # print("Epoch: %04d " % (epoch + 1), lr2, reg2)
            # print("Epoch: %04d " % (epoch + 1), lr3, reg3)
            # print("Epoch: %04d " % (epoch + 1), lr4, reg4)
            # print("Epoch: %04d " % (epoch + 1), lr5, reg5)
            print('epochs:', epoch + 1, ' ndcg@10:', ndcg, ' recall@10', recall, ' timeCost: %.2f' % (t2 - t1))
        print("Result on TestSet:", ' ndcg10:', self.ndcg_10, ' ndcg20:', self.ndcg_20, ' recall10:', self.recall_10,' recall20:', self.recall_20, "Time Cost All: %.2f" % (self.overTime - startTime))





    # def save(self, path):
    #     saver = tf.train.Saver()
    #     saver.save(self.sess, path)

    def predict(self, user_id, item_id):
        return self.sess.run([self.pred_y], feed_dict={self.user_id: user_id, self.item_id: item_id})[0]

    def _get_neg_items(self, data):
        all_items = set(np.arange(self.num_item))
        neg_items = {}
        for u in range(self.num_user):
            neg_items[u] = list(all_items - set(data.getrow(u).nonzero()[1]))

        return neg_items
