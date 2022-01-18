# %%
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from MetaMF import *

# # %%
# random.seed(1)
# np.random.seed(1)
# torch.manual_seed(1)  # set random seed for cpu
# torch.cuda.manual_seed(1)  # set random seed for current gpu
# torch.cuda.manual_seed_all(1)  # set random seed for all gpus
# %%
for k in range(10):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    # %%
    if torch.cuda.is_available():
        use_cuda = True
    else:
        use_cuda = False
    use_cuda

    df = pd.read_csv("./data/ml1m/ratings.dat", sep="::",
                     names=['user_id', 'item_id', 'rating', 'category'],
                     engine='python')

    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    print('n_users:', n_users, ' n_items:', n_items)

    train_data, test_data = train_test_split(df, test_size=0.2)
    train_data, validation_data = train_test_split(train_data, test_size=0.125)
    train_data = pd.DataFrame(train_data)
    validation_data = pd.DataFrame(validation_data)
    test_data = pd.DataFrame(test_data)

    traindata = []
    validdata = []
    testdata = []

    for line in train_data.itertuples():
        user = int(line[1]) - 1
        item = int(line[2]) - 1
        rating = float(line[3])
        traindata.append((user, item, rating))

    for line in validation_data.itertuples():
        user = int(line[1]) - 1
        item = int(line[2]) - 1
        rating = float(line[3])
        validdata.append((user, item, rating))

    for line in test_data.itertuples():
        user = int(line[1]) - 1
        item = int(line[2]) - 1
        rating = float(line[3])
        testdata.append((user, item, rating))


    # %% md

    # Utility Functions
    # %%
    def batchtoinput(batch, use_cuda):
        users = []
        items = []
        ratings = []
        for example in batch:
            users.append(example[0])
            items.append(example[1])
            ratings.append(example[2])
        users = torch.tensor(users, dtype=torch.int64)
        items = torch.tensor(items, dtype=torch.int64)
        ratings = torch.tensor(ratings, dtype=torch.float32)
        if use_cuda:
            users = users.cuda()
            items = items.cuda()
            ratings = ratings.cuda()
        return users, items, ratings


    # %%
    def getbatches(traindata, batch_size, use_cuda, shuffle):
        dataset = traindata.copy()
        if shuffle:
            random.shuffle(dataset)
        for batch_i in range(0, int(np.ceil(len(dataset) / batch_size))):
            start_i = batch_i * batch_size
            batch = dataset[start_i:start_i + batch_size]
            yield batchtoinput(batch, use_cuda)


    # %%
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)


    # eval_metric define
    def get_eval(ratlist, predlist, output=False):
        mae = np.mean(np.abs(ratlist - predlist))
        rmse = np.sqrt(np.mean(np.square(ratlist - predlist)))
        if output:
            maelist = np.abs(ratlist - predlist)
            with open('maelist.dat', 'w') as f:
                i = 0
                while i < len(maelist):
                    f.write(str(maelist[i]) + '\n')
                    i += 1
            mselist = np.square(ratlist - predlist)
            with open('mselist.dat', 'w') as f:
                i = 0
                while i < len(mselist):
                    f.write(str(mselist[i]) + '\n')
                    i += 1
        return mae, rmse


    # %% md
    def test_validation(validdata, batch_size, use_cuda):
        net.eval()  # switch to test/validate mode
        ratlist = []
        predlist = []
        for k, (users, items, ratings) in enumerate(getbatches(validdata, batch_size, use_cuda, False)):
            pred = net(users, items)
            predlist.extend(pred.tolist())
            ratlist.extend(ratings.tolist())
        mae, rmse = get_eval(np.array(ratlist), np.array(predlist))
        return rmse, mae


    # 差分进化算法
    def GenerateTrainVector(ID, maxID, lr_min, lr_max, reg_rate_min, reg_rate_max, lr_matrix, reg_rate_matrix):
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

        if Result[0][0] <= lr_min:
            Result[0][0] = lr_min
        if Result[0][0] >= lr_max:
            Result[0][0] = lr_max
        if Result[1][0] <= reg_rate_min:
            Result[1][0] = reg_rate_min
        if Result[1][0] >= reg_rate_max:
            Result[1][0] = reg_rate_max

        return Result


    # MainFunction
    min_RMSE_DE = 1e10
    min_RMSE_Error = 1e10
    min_mae_Error = 1e10
    min_round = 0
    delay_round = 30
    start_Time = 0
    over_Time = 0
    # 差分进化相关参数
    individual_num = 5
    lr_min = 0
    lr_max = 0.0001
    reg_rate_min = 0
    reg_rate_max = 0.01
    print('learning_rate:', lr_min, '~', lr_max, ' reg_rate:', reg_rate_min, '~', reg_rate_max)

    lr_matrix = np.empty(shape=(individual_num, 1))
    reg_rate_matrix = np.empty(shape=(individual_num, 1))

    for i in range(individual_num):
        xx = lr_min + np.random.rand() * (lr_max - lr_min)
        yy = reg_rate_min + np.random.rand() * (reg_rate_max - reg_rate_min)
        lr_matrix[i][0] = xx
        reg_rate_matrix[i][0] = yy

    # Train Model
    net = MetaMF(n_users, n_items)
    net.apply(weights_init)
    if use_cuda:
        net.cuda()
    # %%
    # optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.001)#for MetaMF
    # optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)#for NeuMF
    # %%
    batch_size = 64  # for MetaMF
    # batch_size = 256#for NeuMF
    epoches = 100
    # %%
    start_Time = time.time()

    for epoch in range(epoches):
        time1 = time.time()
        time2 = 0
        net.train()  # switch to train mode
        error = 0
        num = 0
        rmse_no_de, mae_no_de = test_validation(validdata, batch_size, use_cuda)
        # 差分进化和模型训练
        for ID in range(individual_num):
            # 差分算法进化寻优
            evolution = GenerateTrainVector(ID, individual_num, lr_min, lr_max, reg_rate_min, reg_rate_max, lr_matrix,
                                            reg_rate_matrix)
            learning_rate = evolution[0][0]
            reg_rate = evolution[1][0]
            # 模型优化器
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=reg_rate, momentum=0.99)
            # optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=reg_rate)
            # Model Training
            for k, (users, items, ratings) in enumerate(getbatches(traindata, batch_size, use_cuda, True)):
                optimizer.zero_grad()
                pred = net(users, items)
                loss = net.loss(pred, ratings)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5)
                optimizer.step()
                error += loss.detach().cpu().numpy() * len(users)  # loss is averaged
                num += len(users)
                # if (k+1)%1000 == 0:
                #     print(error/num)
            # print('Epoch {}/{} - Training Loss: {:.3f}'.format(epoch + 1, epoches, error / num))
            rmse_de, mae_de = test_validation(validdata, batch_size, use_cuda)

            if rmse_de <= rmse_no_de:
                lr_matrix[ID][0] = evolution[0][0]
                reg_rate_matrix[ID][0] = evolution[1][0]
                rmse_no_de = rmse_de
                mae_no_de = mae_de

        rmse_Err = rmse_no_de
        mae_Err = mae_no_de
        # 当前rmse误差小于最小误差时继续训练
        if rmse_Err < min_RMSE_DE:
            min_RMSE_DE = rmse_Err
            min_round = epoch + 1
        # 当前rmse误差开始上升，则在验证集上误差最小的时候进行测试集测试，并保存该结果
        else:
            rmse_test, mae_test = test_validation(testdata, batch_size, use_cuda)
            if min_RMSE_Error > rmse_test:
                min_RMSE_Error = rmse_test
                min_mae_Error = mae_test
                over_Time = time.time()

        # 当误差继续上升则达到delay_round停止训练
        if (epoch - min_round) >= delay_round:
            break

        time2 = time.time()
        # print('Epoch {}/{}'.format(epoch + 1, epoches), 'RMSE: {:.6f}'.format(rmse_Err), ' MAE: {:.6f}'.format(mae_Err),
        #       ' time: {:.2f}'.format(time2 - time1))
        print('Epoch {}/{}'.format(epoch+1, epoches), learning_rate, reg_rate)

    print('Final Result:\n', ' %.6f' % min_RMSE_Error, ' %.6f' % min_mae_Error, ' %.2f' % (over_Time - start_Time))
