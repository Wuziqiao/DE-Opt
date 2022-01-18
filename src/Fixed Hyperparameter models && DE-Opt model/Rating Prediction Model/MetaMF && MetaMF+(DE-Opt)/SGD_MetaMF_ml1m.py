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

print('dataset: ML1M')

for i in [0.000001]:
    for j in [10, 1, 0.1, 0.01, 0.001, 0]:
        learning_rate = i
        reg_rate = j
        print('lr:', learning_rate, ' reg:', reg_rate)

        # # %%
        # random.seed(1)
        # np.random.seed(1)
        # torch.manual_seed(1)  # set random seed for cpu
        # torch.cuda.manual_seed(1)  # set random seed for current gpu
        # torch.cuda.manual_seed_all(1)  # set random seed for all gpus
        # # %%
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '4'
        # %%conda
        if torch.cuda.is_available():
            use_cuda = True
        else:
            use_cuda = False
        use_cuda
        # %% md
        # Read Dataset
        # %%


        df = pd.read_csv("./data/ml1m/ratings.dat", sep="::",
                         names=['user_id', 'item_id', 'rating', 'category'],
                         engine='python')


        n_users = df.user_id.unique().shape[0]
        n_items = df.item_id.unique().shape[0]

        print('n_users:', n_users, ' n_items:', n_items)

        train_data, test_data = train_test_split(df, test_size=0.2)
        train_data, validation_data = train_test_split(train_data, test_size=0.125)
        # train_data, test_data = train_test_split(df, test_size=0.01)
        # train_data, validation_data = train_test_split(train_data, test_size=0.8)
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
            # mse = np.mean(np.square(ratlist-predlist))
            mse = np.sqrt(np.mean(np.square(ratlist - predlist)))
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
            return mae, mse


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


        # MainFunction
        min_RMSE = 1e10
        min_RMSE_Error = 1e10
        min_mae_Error = 1e10
        min_round = 0
        delay_round = 10
        start_Time = 0
        over_Time = 0


        # Train Model
        net = MetaMF(n_users, n_items)
        net.apply(weights_init)
        if use_cuda:
            net.cuda()
        # %%
        # optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=reg_rate)  # Adam Optimizer
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=reg_rate, momentum=0.99)  # SGD Optimizer
        # %%
        batch_size = 64  # for MetaMF
        # batch_size = 256 # for NeuMF
        epoches = 1000
        # %%
        start_Time = time.time()
        for epoch in range(epoches):
            time1 = time.time()
            net.train()  # switch to train mode
            error = 0
            num = 0
            # 模型训练步骤
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

            rmse_valid, mae_valid = test_validation(validdata, batch_size, use_cuda)

            # 当前rmse误差小于最小误差时继续训练
            if min_RMSE > rmse_valid:
                min_RMSE = rmse_valid
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
            # 每隔步长5输出当前训练结果
            if epoch % 5 == 0:
                print('Epoch {}'.format(epoch+1), ' RMSE: {:.6f}'.format(rmse_valid), ' MAE: {:.6f}'.format(mae_valid),
                      ' time: {:.2f}'.format(time2 - time1))
            # print('MAE: {:.6f}'.format(mae_valid))
        # 最终在testset上的测试结果
        print('Final Result:\n', ' %.6f' % min_RMSE_Error, ' %.6f' % min_mae_Error, ' %.2f' % (over_Time - start_Time))


