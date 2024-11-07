import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from dataloader import Dataset, Load_Dataset_Train, Load_Dataset_Test
from ViT_model import ViT
import os
import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class LabelSmoothing(torch.nn.Module):
    """NLL loss with label smoothing."""
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


if __name__ == "__main__":
    # Training epochs
    EPOCH = 200

    for seed in range(1, 11):
        curr_time = datetime.datetime.now()
        torch.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True



        # Select the specified path
        data_path = r'D:\气象项目'   # 数据集路径

        # Save file
        save_path = 'D:/hainan/save/' + 'Train_Result'

        # Load dataset, set flooding levels

        flooding_level = [0, 0, 0, 0]
        feature, label = Load_Dataset_Train(data_path)

        _, _, channels, sampling_points = feature.shape

        # 把数据分成训练数据集和验证数据集
        train_data, test_data, train_label, test_label = \
            train_test_split(feature, label, test_size=0.2, random_state=42)


        path = save_path + '/' + str(seed)
        assert os.path.exists(path) is False, 'path is exist'
        os.makedirs(path)

        writer = SummaryWriter("logs")
        # 打开tensorboard命令：tensorboard --logdir=logs --port=6007

        train_set = Dataset(train_data, train_label, transform=True)
        test_set = Dataset(test_data, test_label, transform=True)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)


# -------------------------------------------------------------------------------------------------------------------- #
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        net = ViT(n_class=2, sampling_point=sampling_points, dim=64, depth=6,
                  heads=8, mlp_dim=64, dropout=0, emb_dropout=0).to(device)     # ec为35通道输入



        # # 1.模型查看
        # print(net)#可以看出网络一共有3层，两个Sequential()+avgpool
        # model_features = list(net.children())
        # print(model_features[0][3])#取第0层Sequential()中的第四层
        # for index,layer in enumerate(model_features[0]):
        #     print(layer)

        # criterion = LabelSmoothing(0.1)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(net.parameters())
        lrStep = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

        last_improved = 0  # record global_step at best_val_accuracy
        require_improvement = 10  # break training if not having improvement over 10 epoch
        flag = False

        # #   保存模型
        # with torch.no_grad():
        #     temp_train_data = torch.tensor(train_data, dtype=torch.float).to(device)
        #     writer.add_graph(net, temp_train_data)
        # -------------------------------------------------------------------------------------------------------------------- #
        test_min_loss = 10
        for epoch in range(EPOCH):
            net.train()
            train_running_loss = 0
            train_running_acc = 0
            total = 0
            loss_steps = []
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs[1], labels.long())

                # Piecewise decay flooding. b is flooding level, b = 0 means no flooding
                if epoch < 30:
                    b = flooding_level[0]
                elif epoch < 50:
                    b = flooding_level[1]
                elif epoch < 70:
                    b = flooding_level[2]
                else:
                    b = flooding_level[3]

                # flooding
                loss = (loss - b).abs() + b
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_steps.append(loss.item())
                total += labels.shape[0]
                pred = outputs[1].argmax(dim=1, keepdim=True)
                train_running_acc += pred.eq(labels.view_as(pred)).sum().item()

            train_running_loss = float(np.mean(loss_steps))
            train_running_acc = 100 * train_running_acc / total
            writer.add_scalar("train_acc", train_running_acc, epoch, 1)
            writer.add_scalar("train_loss", train_running_loss, epoch, 2)
            print('[%d] Train loss: %0.4f' % (epoch, train_running_loss))
            print('[%d] Train acc: %0.3f%%' % (epoch, train_running_acc))

            # -------------------------------------------------------------------------------------------------------------------- #
            net.eval()
            test_running_loss = 0
            test_running_acc = 0
            total = 0
            loss_steps = []
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs[1], labels.long())

                    loss_steps.append(loss.item())
                    total += labels.shape[0]
                    pred = outputs[1].argmax(dim=1, keepdim=True)
                    test_running_acc += pred.eq(labels.view_as(pred)).sum().item()

                test_running_acc = 100 * test_running_acc / total
                test_running_loss = float(np.mean(loss_steps))
                writer.add_scalar("test_acc", test_running_acc, epoch, 3)
                writer.add_scalar("test_loss", test_running_loss, epoch, 4)

                if test_running_loss < test_min_loss and epoch > 15:
                    test_min_loss = test_running_loss
                    torch.save(net.state_dict(), path + '/test_max_acc.pt')
                    test_save = open(path + '/test_max_acc.txt', "w")
                    test_save.write("best_acc= %.3f" % (test_running_acc))
                    last_improved = epoch
                    improved_str = '*'
                    test_save.close()
                else:
                    improved_str = ''

                print('     [%d] Test loss: %0.4f %s' % (epoch, test_running_loss, improved_str))
                print('     [%d] Test acc: %0.3f%% %s' % (epoch, test_running_acc, improved_str))

                if epoch - last_improved > require_improvement and epoch > 15:
                    print("超过15步没有优化，停止训练")
                    flag = True
                    break
            if flag:
                break

            lrStep.step()

        writer.close()

