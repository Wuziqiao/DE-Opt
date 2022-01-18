'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import tensorflow.compat.v1 as tf
import os
import sys
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
warnings.filterwarnings("ignore")
from utility.helper import *
from utility.batch_test import *

tf.compat.v1.disable_eager_execution()


class NGCF(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings
        self.model_type = 'ngcf'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type

        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.n_fold = 100

        self.norm_adj = data_config['norm_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()

        self.lr = 100
        self.reg_rate = 100

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers)

        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.verbose = args.verbose

        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        # dropout: node dropout (adopted on the ego-networks);
        #          ... since the usage of node dropout have higher computational cost,
        #          ... please use the 'node_dropout_flag' to indicate whether use such technique.
        #          message dropout (adopted on the convolution operations).
        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = self._init_weights()

        """
        *********************************************************
        Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
        Different Convolutional Layers:
            1. ngcf: defined in 'Neural Graph Collaborative Filtering', SIGIR2019;
            2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
            3. gcmc: defined in 'Graph Convolutional Matrix Completion', KDD2018;
        """
        if self.alg_type in ['ngcf']:
            self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()

        elif self.alg_type in ['gcn']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcn_embed()

        elif self.alg_type in ['gcmc']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcmc_embed()

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)

        """
        *********************************************************
        Inference for the testing phase.
        """
        self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False,
                                       transpose_b=True)

        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        # self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.reg_rate, self.u_g_embeddings,
        #                                                                   self.pos_i_g_embeddings,
        #                                                                   self.neg_i_g_embeddings)
        # self.loss = self.mf_loss + self.emb_loss + self.reg_loss
        #
        # self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _init_weights(self):
        all_weights = dict()

        # initializer = tf.truncated_normal_initializer()
        initializer = tf.glorot_uniform_initializer()
        print('---------------------------------------')

        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                        name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                        name='item_embedding')
            print('using xavier initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            print('using pretrained initialization')

        self.weight_size_list = [self.emb_dim] + self.weight_size

        for k in range(self.n_layers):
            all_weights['W_gc_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_mlp_%d' % k)

        return all_weights

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _create_ngcf_embed(self):
        # Generate a set of adjacency sub-matrix.
        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            # transformed sum messages of neighbors.
            sum_embeddings = tf.nn.leaky_relu(
                tf.matmul(side_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])

            # bi messages of neighbors.
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])

            # non-linear activation.
            ego_embeddings = sum_embeddings + bi_embeddings

            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_gcn_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)
        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [embeddings]

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))

            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.nn.leaky_relu(
                tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_gcmc_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)

        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = []

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))
            embeddings = tf.concat(temp_embed, 0)
            # convolutional layer.
            embeddings = tf.nn.leaky_relu(
                tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            # dense layer.
            mlp_embeddings = tf.matmul(embeddings, self.weights['W_mlp_%d' % k]) + self.weights['b_mlp_%d' % k]
            mlp_embeddings = tf.nn.dropout(mlp_embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [mlp_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    # Loss Function为BPR，构建BPR损失函数
    def create_bpr_loss(self, reg_rate, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer / self.batch_size

        mf_loss = tf.reduce_sum(tf.nn.softplus(-(pos_scores - neg_scores)))

        emb_loss = reg_rate * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)


def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data


def GenerateTrainVector(ID, maxID, lr_matrix, reg_rate_matrix):
    SFGSS = 8
    SFHC = 20
    Fl = 0.1
    Fu = 0.9
    tuo1 = 0.1
    tuo2 = 0.03
    tuo3 = 0.07

    Result = np.empty(shape=(2, 1))

    u1 = ID
    u2 = ID
    u3 = ID
    while (u1 == ID):
        a = np.random.rand() * (maxID - 1)
        u1 = round(a)
    while ((u2 == ID) or (u2 == u1)):
        b = np.random.rand() * (maxID - 1)
        u2 = round(b)
    while ((u3 == ID) or (u3 == u2) or (u3 == u1)):
        c = np.random.rand() * (maxID - 1)
        u3 = round(c)

    rand1 = np.random.rand()
    rand2 = np.random.rand()
    rand3 = np.random.rand()
    F = np.random.rand()
    K = np.random.rand()

    if (rand3 < tuo2):
        F = SFGSS
    elif (tuo2 <= rand3 and rand3 < tuo3):
        F = SFHC
    elif (rand2 < tuo1 and rand3 > tuo3):
        F = Fl + Fu * rand1

    temp1 = lr_matrix[u2][0] - lr_matrix[u3][0]
    temp2 = temp1 * F
    temp_mutation = lr_matrix[u1][0] + temp2
    temp1 = temp_mutation - lr_matrix[ID][0]
    temp2 = temp1 * K
    Result[0][0] = lr_matrix[ID][0] + temp2

    temp1 = reg_rate_matrix[u2][0] - reg_rate_matrix[u3][0]
    temp2 = temp1 * F
    temp_mutation = reg_rate_matrix[u1][0] + temp2
    temp1 = temp_mutation - reg_rate_matrix[ID][0]
    temp2 = temp1 * K
    Result[1][0] = reg_rate_matrix[ID][0] + temp2

    if (Result[0][0] <= lr_min):
        Result[0][0] = lr_min
    if (Result[0][0] >= lr_max):
        Result[0][0] = lr_max
    if (Result[1][0] <= reg_rate_min):
        Result[1][0] = reg_rate_min
    if (Result[1][0] >= reg_rate_max):
        Result[1][0] = reg_rate_max
    return Result


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    users_to_validation = list(data_generator.validation_set.keys())
    users_to_test = list(data_generator.test_set.keys())
    print('data loading is ok')
    ndcg_Err = 0
    recall_Err = 0
    lr_min = 0
    lr_max = 0.01
    reg_rate_min = 0
    reg_rate_max = 1
    individual_num = 5
    max_ndcg = 0
    best_ndcg_10 = 0
    best_ndcg_20 = 0
    best_recall_10 = 0
    best_recall_20 = 0
    min_round = -1
    delay_round = 10
    print('lr min:', lr_min, '; lr max:', lr_max, '; reg min:', reg_rate_min, '; reg max:', reg_rate_max)

    lr_matrix = np.empty(shape=(individual_num, 1))
    reg_rate_matrix = np.empty(shape=(individual_num, 1))

    for i in range(individual_num):
        xx = lr_min + np.random.rand() * (lr_max - lr_min)
        yy = reg_rate_min + np.random.rand() * (reg_rate_max - reg_rate_min)
        lr_matrix[i][0] = xx
        reg_rate_matrix[i][0] = yy

    # print('lr:' + str(lr_min)+'-'+str(lr_max), ' reg:' + str(reg_rate_min)+'-'+str(reg_rate_max))

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    # the type of the adjacency (laplacian) matrix from {plain, norm, mean}
    if args.adj_type == 'plain':
        config['norm_adj'] = plain_adj
        print('use the plain adjacency matrix')

    elif args.adj_type == 'norm':
        config['norm_adj'] = norm_adj
        print('use the normalized adjacency matrix')

    elif args.adj_type == 'gcmc':
        config['norm_adj'] = mean_adj
        print('use the gcmc adjacency matrix')

    else:
        config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
        print('use the mean adjacency matrix')


    model = NGCF(data_config=config, pretrain_data=None)

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    cur_best_pre_0 = 0.

    """
    *********************************************************
    Train.
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False

    t0 = time()
    t3 = 0
    # 训练开始
    for epoch in range(args.epoch):
        t1 = time()
        # users_to_validation = list(data_generator.validation_set.keys())
        ret_validation = validation(sess, model, users_to_validation, drop_flag=True)
        ndcg_no_de = ret_validation['ndcg'][1]
        recall_no_de = ret_validation['recall'][1]
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
        for ID in range(5):
            evolution = GenerateTrainVector(ID, individual_num, lr_matrix, reg_rate_matrix)
            lr = evolution[0][0]
            reg_rate = evolution[1][0]
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
            # -----------------------------------------------------
            loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
            n_batch = data_generator.n_train // args.batch_size + 1
            model_mf_loss, model_emb_loss, model_reg_loss = model.create_bpr_loss(reg_rate, model.u_g_embeddings,
                                                                                  model.pos_i_g_embeddings,
                                                                                  model.neg_i_g_embeddings)
            model_loss = model_mf_loss + model_emb_loss + model_reg_loss
            model_opt = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(model_loss)
            for idx in range(n_batch):
                users, pos_items, neg_items = data_generator.sample()
                _, batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss = sess.run(
                    [model_opt, model_loss, model_mf_loss, model_emb_loss, model_reg_loss],
                    feed_dict={model.users: users, model.pos_items: pos_items,
                               model.node_dropout: eval(args.node_dropout),
                               model.mess_dropout: eval(args.mess_dropout),
                               model.neg_items: neg_items})
                loss += batch_loss
                mf_loss += batch_mf_loss
                emb_loss += batch_emb_loss
                reg_loss += batch_reg_loss
            if np.isnan(loss) == True:
                print('ERROR: loss is nan.')
                sys.exit()
            # -------------------------------------------------------

            # users_to_validation = list(data_generator.validation_set.keys())
            ret_validation = validation(sess, model, users_to_validation, drop_flag=True)
            ndcg_de = ret_validation['ndcg'][1]
            recall_de = ret_validation['recall'][1]
            if ndcg_de >= ndcg_no_de:
                lr_matrix[ID][0] = evolution[0][0]
                reg_rate_matrix[ID][0] = evolution[1][0]
                ndcg_no_de = ndcg_de
                recall_no_de = recall_de

        # users_to_validation = list(data_generator.validation_set.keys())
        # ret_validation = validation(sess, model, users_to_validation, drop_flag=True)
        ndcg_Err = ndcg_no_de
        recall_Err = recall_no_de
        t2 = time()
        if ndcg_Err > max_ndcg:
            max_ndcg = ndcg_Err
            min_round = epoch+1
        else:
            # users_to_test = list(data_generator.test_set.keys())
            ret_test = test(sess, model, users_to_test, drop_flag=True)
            ndcg_tmp_10 = ret_test['ndcg'][0]
            ndcg_tmp_20 = ret_test['ndcg'][1]
            recall_tmp_10 = ret_test['recall'][0]
            recall_tmp_20 = ret_test['recall'][1]
            if ndcg_tmp_10 > best_ndcg_10:
                best_ndcg_10 = ndcg_tmp_10
                best_ndcg_20 = ndcg_tmp_20
                best_recall_10 = recall_tmp_10
                best_recall_20 = recall_tmp_20
                t3 = time()
        if (epoch - min_round) >= delay_round:
            break

        # print('Epoch: %04d:'%epoch, 'NDCG@10:'+str(ndcg_Err) +' RECALL@10:'+str(recall_Err),
        #       'timecost:', t2-t1)
        print(" %04d " % (epoch + 1), lr1, reg1)
        print(" %04d " % (epoch + 1), lr2, reg2)
        print(" %04d " % (epoch + 1), lr3, reg3)
        print(" %04d " % (epoch + 1), lr4, reg4)
        print(" %04d " % (epoch + 1), lr5, reg5)
    print('Result on TestSet:\n', 'NDCG@10:%.6f'%best_ndcg_10, '; NDCG@20:%.6f'%best_ndcg_20, '; RECALL@10:%.6f'%best_recall_10, '; RECALL@20:%.6f'%best_recall_20,
          'TimeCost:', t3-t0)
