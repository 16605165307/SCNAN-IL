import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import spines
from scipy.io import loadmat
from sklearn import preprocessing
from scipy.stats import entropy
import pywt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
from torch import device
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from Models_Water import Embedding_Net, SClassifier, Generator, Discriminator, UClassifier, SClassifier_Online
from early_stopping_online import EarlyStoppingOnline
from early_stopping import EarlyStopping
from early_stopping_ucls import EarlyStoppingUCls


# 小波去噪函数
def wavelet_denoise_2d(data, wavelet='db1', level=1):
    denoised_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        signal = data[:, i]
        coefficients = pywt.wavedec(signal, wavelet, mode='per')
        coefficients[1:] = (pywt.threshold(j, value=0.1, mode='soft') for j in coefficients[1:])
        denoised_data[:, i] = pywt.waverec(coefficients, wavelet, mode='per')
    return denoised_data


def get_data_list(path, is_train):
    fault1 = pd.read_excel(path + 'fault1.xls').iloc[0:500] if is_train else pd.read_excel(path + 'fault1.xls').iloc[
                                                                             800:1280]
    fault2 = pd.read_excel(path + 'fault2.xls').iloc[0:500] if is_train else pd.read_excel(path + 'fault2.xls').iloc[
                                                                             800:1280]
    fault3 = pd.read_excel(path + 'fault3.xls').iloc[0:500] if is_train else pd.read_excel(path + 'fault3.xls').iloc[
                                                                             800:1280]
    fault4 = pd.read_excel(path + 'fault4.xls').iloc[0:500] if is_train else pd.read_excel(path + 'fault4.xls').iloc[
                                                                             800:1280]
    fault5 = pd.read_excel(path + 'fault5.xls').iloc[0:500] if is_train else pd.read_excel(path + 'fault5.xls').iloc[
                                                                             800:1280]
    fault6 = pd.read_excel(path + 'fault6.xls').iloc[0:500] if is_train else pd.read_excel(path + 'fault6.xls').iloc[
                                                                             800:1280]
    fault7 = pd.read_excel(path + 'fault7.xls').iloc[0:500] if is_train else pd.read_excel(path + 'fault7.xls').iloc[
                                                                             800:1280]

    fault1 = fault1.to_numpy()
    fault2 = fault2.to_numpy()
    fault3 = fault3.to_numpy()
    fault4 = fault4.to_numpy()
    fault5 = fault5.to_numpy()
    fault6 = fault6.to_numpy()
    fault7 = fault7.to_numpy()

    data_list = [fault1, fault2, fault3, fault4, fault5, fault6, fault7]
    return data_list


def creat_dataset_seen(is_train=True):
    path = r'D:\Projects\metaGAN\Datasets\Water_data\\'
    print("creat_dataset_seen loading data...")
    seen_index = [index - 1 for index in train_classes]
    attribute_matrix_ = pd.read_excel('D:/Projects/metaGAN/attribute_matrix_water.xlsx', index_col='no')
    attribute_matrix = attribute_matrix_.values
    seen_index.sort()
    print("seen classes: {}".format(train_classes))
    data_list = get_data_list(path, is_train)
    # 只关注已见类
    data = []
    attributelabel = []
    label = []
    for item in range(len(seen_index)):
        label += [item] * per_train_shots if is_train else [item] * per_test_shots
        attributelabel += [attribute_matrix[seen_index[item], :]] * per_train_shots if is_train else [attribute_matrix[
                                                                                                      seen_index[item],
                                                                                                      :]] * per_test_shots
        data.append(data_list[seen_index[item]])
    data = np.row_stack(data)
    label = np.row_stack(label)
    attributelabel = np.row_stack(attributelabel)
    return data, attributelabel, label


def creat_dataset_unseen(is_train=True):
    path = r'D:\Projects\metaGAN\Datasets\Water_data\\'
    print("creat_dataset_unseen loading data...")
    zero_test_index = [index - 1 for index in test_classes]  # test_index对应的是类别序号，-1才是索引
    attribute_matrix_ = pd.read_excel('D:/Projects/metaGAN/attribute_matrix_water.xlsx', index_col='no')
    attribute_matrix = attribute_matrix_.values
    zero_test_index.sort()
    print("test unseen classes: {}".format(test_classes))
    data_list = get_data_list(path, is_train)
    # 只关注未见类
    data = []
    attributelabel = []
    label = []
    for item in range(len(zero_test_index)):
        label += [item] * per_train_shots if is_train else [item] * per_test_shots
        attributelabel += [attribute_matrix[zero_test_index[item], :]] * per_train_shots if is_train else [
                                                                                                              attribute_matrix[
                                                                                                              zero_test_index[
                                                                                                                  item],
                                                                                                              :]] * per_test_shots
        data.append(data_list[zero_test_index[item]])
    data = np.row_stack(data)
    label = np.row_stack(label)
    attributelabel = np.row_stack(attributelabel)
    return data, attributelabel, label


def create_dataset_comp(is_train=True):
    path = r'D:\Projects\metaGAN\Datasets\Water_data\\'
    print("create_dataset_comp loading data...")
    zero_test_index = [index - 1 for index in test_classes]  # test_index对应的是类别序号，-1才是索引
    attribute_matrix_ = pd.read_excel('D:/Projects/metaGAN/attribute_matrix_water.xlsx', index_col='no')
    attribute_matrix = attribute_matrix_.values
    train_index = list(set(np.arange(7)) - set(zero_test_index))
    zero_test_index.sort()
    train_index.sort()
    print("test classes: {}".format(test_classes))
    print("train classes: {}".format(train_classes))
    data_list = get_data_list(path, is_train)
    train_attributelabel = []
    traindata = []
    train_label = []
    train_true_label = []
    # 遍历训练故障类型
    for item in range(len(train_index)):
        train_attributelabel += [attribute_matrix[train_index[item], :]] * per_train_shots if is_train else [
                                                                                                                attribute_matrix[
                                                                                                                train_index[
                                                                                                                    item],
                                                                                                                :]] * per_test_shots
        traindata.append(data_list[train_index[item]])
        train_label += [item] * per_train_shots if is_train else [item] * per_test_shots
        train_true_label += [train_classes[item]] * per_train_shots if is_train else [train_classes[
                                                                                          item]] * per_test_shots  # 真实标签就直接按1~15来吧

    train_label = np.row_stack(train_label)
    train_true_label = np.row_stack(train_true_label)
    train_attributelabel = np.row_stack(train_attributelabel)
    traindata = np.row_stack(traindata)
    # 将未见类样本加入
    data = []
    attributelabel = []
    label = []
    true_label = []
    for item in range(len(zero_test_index)):
        label += [len(train_classes)] * per_train_shots if is_train else [len(train_classes)] * per_test_shots
        attributelabel += [attribute_matrix[zero_test_index[item], :]] * per_train_shots if is_train else [
                                                                                                              attribute_matrix[
                                                                                                              zero_test_index[
                                                                                                                  item],
                                                                                                              :]] * per_test_shots
        data.append(data_list[zero_test_index[item]])
        true_label += [test_classes[item]] * per_train_shots if is_train else [test_classes[item]] * per_test_shots
    data = np.row_stack(data)
    label = np.row_stack(label)
    attributelabel = np.row_stack(attributelabel)
    true_label = np.row_stack(true_label)
    traindata = np.vstack((traindata, data))
    train_attributelabel = np.vstack((train_attributelabel, attributelabel))
    train_label = np.vstack((train_label, label))
    train_true_label = np.vstack((train_true_label, true_label))
    return traindata, train_attributelabel, train_label, train_true_label


def creat_dataset_online():
    path = r'D:\Projects\metaGAN\Datasets\Water_data\\'
    print("creat_dataset_online loading data...")
    fault1 = pd.read_excel(path + 'fault1.xls').iloc[500:800].to_numpy()
    fault2 = pd.read_excel(path + 'fault2.xls').iloc[500:800].to_numpy()
    fault3 = pd.read_excel(path + 'fault3.xls').iloc[500:800].to_numpy()
    fault4 = pd.read_excel(path + 'fault4.xls').iloc[500:800].to_numpy()
    fault5 = pd.read_excel(path + 'fault5.xls').iloc[500:800].to_numpy()
    fault6 = pd.read_excel(path + 'fault6.xls').iloc[500:800].to_numpy()
    fault7 = pd.read_excel(path + 'fault7.xls').iloc[500:800].to_numpy()

    # 假设有300样本用来增量训练，480用来测试
    unseen_index = [index - 1 for index in test_classes]  # test_classes对应的是类别序号，-1才是索引
    seen_index = [index - 1 for index in train_classes]
    attribute_matrix_ = pd.read_excel('D:/Projects/metaGAN/attribute_matrix_water.xlsx', index_col='no')
    attribute_matrix = attribute_matrix_.values
    seen_index.sort()
    unseen_index.sort()
    print("seen classes: {}".format(train_classes))
    print("unseen classes: {}".format(test_classes))
    # 用于增量训练的data_list
    new_data_list = [fault1, fault2, fault3, fault4, fault5, fault6, fault7]
    # 用于增量测试的数据就用create_dataset_comp里的（一开始的GZSL样本）即可
    # 先把300个增量训练样本处理好
    new_data = []
    new_attributelabel = []
    new_label = []
    new_true_label = []
    for item in range(len(seen_index)):
        new_label += [item] * per_new_shots
        new_attributelabel += [attribute_matrix[seen_index[item], :]] * per_new_shots
        new_data.append(new_data_list[seen_index[item]])
        new_true_label += [train_classes[item]] * per_new_shots
    for item in range(len(unseen_index)):
        new_label += [len(train_classes)] * per_new_shots
        new_attributelabel += [attribute_matrix[unseen_index[item], :]] * per_new_shots
        new_data.append(new_data_list[unseen_index[item]])
        new_true_label += [test_classes[item]] * per_new_shots
    new_data = np.row_stack(new_data)
    new_label = np.row_stack(new_label)
    new_attributelabel = np.row_stack(new_attributelabel)
    new_true_label = np.row_stack(new_true_label)
    # 得先对新样本进行归一化后经过sce得到提取后的样本，因为对于正确分类的样本都是提取特征后的！
    # 得先去噪！A,B[4, 7, 10]不适用
    # new_data = wavelet_denoise_2d(new_data)
    # 用训练S_Classifier的scaler？
    train_datas_comp_ = []
    for i in range(len(train_classes) + len(test_classes)):
        # i:i+每类新样本的数量
        train_datas_comp_.append(train_datas_comp[i:i + per_new_shots, :])
    train_datas_comp_ = np.row_stack(train_datas_comp_)
    # 检测漂移 反正肯定有！
    drift_detected, kl_divergence = detect_concept_drift(train_datas_comp_, new_data)
    print(f"每个特征维度是否检测到漂移: {drift_detected}")

    return new_data, new_attributelabel, new_label, new_true_label


def calculate_kl_divergence(p, q):
    """
    计算KL散度

    Parameters:
    - p: 第一个分布的概率分布
    - q: 第二个分布的概率分布

    Returns:
    - kl_divergence: KL散度值
    """
    kl_divergence = entropy(p, q)
    return kl_divergence


def detect_concept_drift(data_before_drift, data_after_drift, threshold=0.1):
    """
    检测概念漂移

    Parameters:
    - data_before_drift: 漂移前的数据
    - data_after_drift: 漂移后的数据
    - threshold: 判断漂移的阈值，默认为0.1

    Returns:
    - drift_detected: 是否检测到漂移（每个特征维度分别判断）
    - kl_divergence: 每个特征维度的KL散度值
    """
    num_features = data_before_drift.shape[1]
    drift_detected = np.zeros(num_features)
    kl_divergence = np.zeros(num_features)

    for i in range(num_features):
        kl_divergence[i] = calculate_kl_divergence(data_before_drift[:, i], data_after_drift[:, i])
        drift_detected[i] = kl_divergence[i] > threshold

    return drift_detected, kl_divergence


class MyData(data.Dataset):
    def __init__(self, datas, attribute_labels, labels):
        super(MyData, self).__init__()
        self.datas = datas
        self.attribute_labels = attribute_labels
        self.labels = labels.reshape(-1)

    def __getitem__(self, index):
        return self.datas[index], self.attribute_labels[index], self.labels[index]

    def __len__(self):
        return self.datas.shape[0]


class MyDataFour(data.Dataset):
    def __init__(self, datas, attribute_labels, labels, true_labels):
        super(MyDataFour, self).__init__()
        self.datas = datas
        self.attribute_labels = attribute_labels
        self.labels = labels.reshape(-1)
        self.true_labels = true_labels.reshape(-1)

    def __getitem__(self, index):
        return self.datas[index], self.attribute_labels[index], self.labels[index], self.true_labels[index]

    def __len__(self):
        return self.datas.shape[0]


class MyDataIndex(data.Dataset):
    def __init__(self, datas, attribute_labels, labels, true_labels):
        super(MyDataIndex, self).__init__()
        self.datas = datas
        self.attribute_labels = attribute_labels
        self.labels = labels.reshape(-1)
        self.true_labels = true_labels.reshape(-1)

    def __getitem__(self, index):
        return index, self.datas[index], self.attribute_labels[index], self.labels[index], self.true_labels[index]

    def __len__(self):
        return self.datas.shape[0]


# 知识蒸馏损失函数
class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=1)

    def forward(self, student_logits, teacher_logits):
        student_probs = self.softmax(student_logits / self.temperature)
        teacher_probs = self.softmax(teacher_logits / self.temperature)
        loss = nn.KLDivLoss()(student_probs, teacher_probs) * (self.temperature ** 2)
        return loss


def SCE():
    load_path = r"D:\Projects\metaGAN\Models\Online\SavedModels\SCE"
    lambda_ = [1, 1e-5, 1, 0.25]
    dim = [25, 50]  # 属性有12个，提取后为50个，则特征也是50个
    sce = Embedding_Net(dim, lambda_=lambda_)
    sce.load_state_dict(torch.load(load_path))
    print('Load Model successfully from [%s]' % load_path)
    return sce


def train_sce():
    lambda_ = [1, 1e-5, 1, 0.25]
    dim = [25, 50]
    LR = 1e-4
    # 不能用未见类样本，data和attrs对应不就是相当于知道标签吗！
    trainset = MyData(train_datas_seen, train_attrs_seen, train_labels_seen)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=256, shuffle=True
    )
    model = Embedding_Net(dim, lambda_=lambda_).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    L1, L2, L3, L = [], [], [], []
    model.train()
    for epoch in range(3000):
        print('=======================epoch: {}======================='.format(epoch))
        model.train()
        for batch_data, batch_attr, _ in train_loader:
            batch_data = batch_data.float().to(device)
            batch_attr = batch_attr.float().to(device)
            optimizer.zero_grad()
            package = model(batch_data, batch_attr)
            loss_R1, loss_R2, loss_CM, loss = package['r1'], package['r2'], package['cm'], package['loss']
            loss.backward()
            optimizer.step()
            L1.append(loss_R1.item())
            L2.append(loss_R2.item())
            L3.append(loss_CM.item())
            L.append(loss.item())
        print('loss:{}'.format(min(L)))
    filename = r"D:\Projects\metaGAN\Models\Online\SavedModels\SCE"
    torch.save(model.state_dict(), filename)
    print(f'Save model at: {filename}')
    # 再用t-sne看一下
    test_datas_sce = get_sce_data_seen(test_datas_comp, test_attrs_comp)
    tsne_classes = [1, 2, 3, 6, 7, 4, 5]  #
    tsne_data = []
    for item in range(len(tsne_classes)):
        tsne_data.append(test_datas_sce[item * 480:(item + 1) * 480, :])
    tsne_data = np.reshape(np.array(tsne_data), (-1, 50))
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(tsne_data)
    plt.figure(figsize=(10, 8))
    labels_ = test_true_labels.flatten('A').tolist()
    labels = []
    for i in tsne_classes:
        for j in labels_:
            if j == i:
                labels.append(j)
    labels = np.array(labels)
    x_min, x_max = tsne_results.min(0), tsne_results.max(0)
    X_norm = (tsne_results - x_min) / (x_max - x_min)  # 归一化
    # 处理一下，把不用的标签给删掉
    # labels = np.delete(labels,np.where(labels >= 7))
    for label in np.unique(labels):
        indices = labels == label  # 把labels中与label标签相等的索引所在的值置为true
        plt.scatter(X_norm[indices, 0], X_norm[indices, 1], label=f'Fault {label}', alpha=0.5)
    # plt.title('T-SNE Visualization of Centrifugal Chiller', fontsize=25)
    plt.legend(fontsize=20)
    # 设置 X 轴和 Y 轴刻度数字的字体大小
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()


def train_sclassifier(output_dim=12):
    early_stopping = EarlyStopping('')
    train_datas_sce = get_sce_data_seen(train_datas_comp, train_attrs_comp)
    test_datas_sce = get_sce_data_seen(test_datas_comp, test_attrs_comp)
    train_set = MyDataFour(train_datas_sce, train_attrs_comp, train_labels_comp, train_true_labels)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=256, shuffle=True
    )
    test_set = MyDataFour(test_datas_sce, test_attrs_comp, test_labels_comp, test_true_labels)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False
    )
    model = SClassifier(output_dim=output_dim).to(device)
    # optimizer = optim.AdamW(model.parameters(), lr=0.000005, weight_decay=0.00005)
    optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.000005)  # 对B不好？
    train_acc_SCls = []
    test_acc_SCls = []
    epoch_accS = []
    y_true = []
    y_pred = []
    epochs = 700
    for epoch in range(epochs):
        epoch_accS.append(epoch + 1)
        model.train()
        sum = 0
        count = 0
        for data, attr, label, true_label in train_loader:
            x = data.float().to(device)
            y = label.float().to(device)
            o = model(x)
            loss = torch.nn.CrossEntropyLoss()(o, y.long())
            output = o.data.cpu().numpy()
            pred = output.argmax(axis=1)
            target = label.numpy()
            sum += (pred == target).sum()
            count += target.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_acc = sum / count
        train_acc_SCls.append(train_acc)
        model.eval()
        sum = 0
        count = 0
        eval_loss = 0
        for data, attr, label, true_label in test_loader:
            x = data.float().to(device)
            y = label.to(device)
            o = model(x)
            loss = torch.nn.CrossEntropyLoss()(o, y.long()).to(device)
            eval_loss += loss.item()
            output = o.data.cpu().numpy()
            pred = output.argmax(axis=1)
            target = label.cpu().numpy()
            if epoch == epochs - 1:  # 只有最后一轮迭代才需要！
                y_pred.append(pred)
                y_true.append(target)
            sum += (pred == target).sum()
            count += target.shape[0]
        test_acc = sum / count
        test_acc_SCls.append(test_acc)
        print('{}\tepoch = {}\ttrain accuracy: {:0.3f}\ttest accuracy: {:0.3f}\ttest loss: {:0.3f}' \
              .format(datetime.now(), epoch, train_acc, test_acc, eval_loss))
        early_stopping(eval_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    # # 画出正确率曲线
    # plt.xlabel('Epochs')  # x轴表示
    # plt.ylabel('Classification Accuracy(%)')  # y轴表示
    # # plt.title("chart")  # 图标标题表示
    # plt.plot(epoch_accS, train_acc_SCls, color='r', label='train_acc')
    # plt.plot(epoch_accS, test_acc_SCls, color='green', label='test_acc')
    # plt.legend(fontsize=20)
    # # 设置 X 轴和 Y 轴刻度数字的字体大小
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.show()  # 显示图片

    # # 画出混淆矩阵
    # y_pred = np.concatenate(y_pred, dtype='int32')
    # y_true = np.concatenate(y_true, dtype='int32')
    # C = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5])  # 可将'1'等替换成自己的类别，如'cat'。
    # # plt.figure(figsize=(12, 12))
    # plt.matshow(C, cmap=plt.cm.Reds)  # 根据最下面的图按自己需求更改颜色
    # plt.colorbar()
    # for i in range(len(C)):
    #     for j in range(len(C)):
    #         plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center', fontsize=15)
    # # plt.tick_params(labelsize=15)  # 设置左边和上面的label类别如0,1,2,3,4的字体大小。
    # # plt.title('Confusion Matrix', fontsize=20)
    # plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 17})  # 设置字体大小。
    # plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 17})
    # plt.xticks(range(0, 6), labels=['f1', 'f2', 'f3', 'f4', 'f5', 'unk'], fontsize=15)  # 将x轴或y轴坐标，刻度 替换为文字/字符
    # plt.yticks(range(0, 6), labels=['f1', 'f2', 'f3', 'f4', 'f5', 'unk'], fontsize=15)
    # plt.show()


# 对传入数据进行sce再归一化
def get_sce_data_seen(data, attr):
    sce = SCE()
    # 初始数据都用seen_scaler归一化好了，但是sce后的数据还没有归一化器
    # 获得sce后的归一化器(归一化器必须是用只有已见类的数据)
    seen_datas_sce = sce(torch.from_numpy(train_datas_seen).float(), torch.from_numpy(train_attrs_seen).float())[
        'z1'].detach().numpy()
    global sce_scaler
    sce_scaler = preprocessing.StandardScaler().fit(seen_datas_sce)
    # 传入的数据也得先sce啊！！！
    data_sce = sce(torch.from_numpy(data).float(), torch.from_numpy(attr).float())['z1'].detach().numpy()
    # 归一化的数据是传过来的数据
    datas_sce = sce_scaler.transform(data_sce)
    return datas_sce


def get_sce_attr_seen(data, attr):
    sce = SCE()
    attrs_sce = sce(torch.from_numpy(data).float(), torch.from_numpy(attr).float())['z2'].detach().numpy()
    return attrs_sce


# def ploy_matrix_3d():
#     test_datas_sce = get_sce_data_seen(test_datas_comp, test_attrs_comp).astype('float32')[:800]
#     test_attrs_sce = get_sce_attr_seen(test_datas_comp, test_attrs_comp).astype('float32')[:800]
#     x, y = np.meshgrid(test_datas_sce, test_attrs_sce)
#     N, D = torch.from_numpy(test_datas_sce).shape
#     z = nn.BatchNorm1d(50)(torch.from_numpy(test_datas_sce)).T @ nn.BatchNorm1d(50)(torch.from_numpy(test_attrs_sce)) / N
#     z = z.detach().numpy()
#     fig = plt.figure(figsize=(14, 7))
#     ax = fig.add_subplot(111, projection='3d')
#     surf = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
#     # Set the angle of the camera
#     ax.view_init(30, 120)
#
#     # Set labels as in the original image
#     ax.set_xlabel('i')
#     ax.set_ylabel('j')
#     ax.set_zlabel('Bij')
#
#     # Show the plot
#     plt.show()

# 定义 Self-paced sample weighting 的权重调整函数
def self_paced_weighting(loss, lambda_param_2=1):
    return torch.exp(-loss / lambda_param_2)


def train_uclassifier(output_dim=3):
    attribute_matrix_ = pd.read_excel(r'D:\Projects\metaGAN\attribute_matrix_water.xlsx', index_col='no')
    attribute_matrix = attribute_matrix_.values
    early_stopping = EarlyStoppingUCls('')
    # 训练GAN用的是已见类
    train_datas_sce = get_sce_data_seen(train_datas_seen, train_attrs_seen)
    train_set = MyData(train_datas_sce, train_attrs_seen, train_labels_seen)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=256, shuffle=True
    )
    test_datas_sce = get_sce_data_seen(test_datas_unseen, test_attrs_unseen)
    test_set = MyData(test_datas_sce, test_attrs_unseen, test_labels_unseen)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False
    )
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    uclassifier = UClassifier(output_dim).to(device)
    # Define loss functions
    criterion = nn.BCELoss()
    # Define optimizers for the generator and discriminator
    optimizer_G = optim.Adam(generator.parameters(), lr=0.00001)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.00001)
    num_epochs = 1000
    if input('Train GAN? y/n\n').lower() == 'y':
        for epoch in range(num_epochs):
            for batch_idx, (real_faults, attrs, _) in enumerate(train_loader):
                real_faults = real_faults.view(-1, 50).to(device)
                attrs = attrs.view(-1, 12).to(device)
                batch_size = real_faults.size(0)
                real_labels = torch.ones(batch_size, 1).to(device)
                fake_labels = torch.zeros(batch_size, 1).to(device)
                # Train the discriminator
                optimizer_D.zero_grad()
                outputs = discriminator(real_faults, attrs)
                loss_real = criterion(outputs, real_labels)
                z = torch.randn(batch_size, latent_dim).to(device)
                fake_faults = generator(z, attrs)
                outputs = discriminator(fake_faults, attrs)
                loss_fake = criterion(outputs, fake_labels)
                loss_D = loss_real + loss_fake
                loss_D.backward()
                optimizer_D.step()
                # Train the generator
                optimizer_G.zero_grad()
                z = torch.randn(batch_size, latent_dim).to(device)
                fake_faults = generator(z, attrs)
                outputs = discriminator(fake_faults, attrs)
                loss_G = criterion(outputs, real_labels)
                loss_G.backward()
                optimizer_G.step()
                if batch_idx % 10 == 0:
                    print(f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(train_loader)} "
                          f"Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
        torch.save(generator.state_dict(), r'D:\Projects\metaGAN\Models\Online\SavedModels\Gen')
        torch.save(discriminator.state_dict(), r'D:\Projects\metaGAN\Models\Online\SavedModels\Disc')
        print(
            r'Save generator model at: D:\Projects\metaGAN\Models\Online\SavedModels\Gen ; Save discriminator model '
            r'at: '
            r'D:\Projects\metaGAN\Models\Online\SavedModels\Disc')
    # 生成样本  默认三个未见类
    gen_label = []
    gen_attributelabel = []
    for item in range(len(test_classes)):
        gen_label += [item] * 20  # 280好用
        gen_attributelabel += [attribute_matrix[test_classes[item] - 1, :]] * 20  # test_index是类别号，索引要-1
    gen_label = np.row_stack(gen_label)
    gen_attributelabel = np.row_stack(gen_attributelabel)
    z = torch.randn(gen_attributelabel.shape[0], latent_dim).to(device)
    gen_fault = generator(z, torch.Tensor(gen_attributelabel).to(device)).to(device)
    print('生成样本维度：', gen_fault.shape)  # 特征就是sce后的维度
    global sce_scaler
    gen_fault = sce_scaler.transform(gen_fault.cpu().detach().numpy())
    gen_data = torch.from_numpy(gen_fault).to(device)
    # # 用t-sne看看生成的样本什么样！！！一坨答辩！！！（可视化-start!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!）
    # real_num = 20  # 要画的真实样本的个数
    # num = 380
    # real_data = np.vstack((test_datas_sce[0 * per_test_shots+num:0 * per_test_shots+num+real_num, :],
    #                        test_datas_sce[1 * per_test_shots+num:1 * per_test_shots + num + real_num, :]
    #                        ))
    # # 把真实样本在t-sne的标签设为2，3
    # real_labels = np.hstack(([2] * real_num, [3] * real_num))
    # real_labels = np.reshape(real_labels, (-1, 1))
    # gen_labels = torch.from_numpy(gen_label).to(device)
    # all_data = np.vstack([real_data, gen_data.cpu().numpy()])
    # all_labels = np.vstack([real_labels, gen_labels.cpu().numpy()])
    # tsne = TSNE(n_components=2)
    # tsne_results = tsne.fit_transform(all_data)
    # x_min, x_max = tsne_results.min(0), tsne_results.max(0)
    # X_norm = (tsne_results - x_min) / (x_max - x_min)  # 归一化
    # maker = ['o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']      # 设置散点形状
    # colors = ['#FF0000', '#00FF00']      # 设置散点颜色（真实的设为红色，生成的设为绿色）
    # Label_Com = ['a', 'b', 'c', 'd']      # 图例名称
    # plt.figure(figsize=(10, 8))
    # # 在这改变想要查看的类别，第一个参数表示真实样本在all_labels里的标签，第二个参数表示生成样本在all_labels里的标签
    # plot_labels = [2, 0]
    # for i in range(len(X_norm)):
    #     if all_labels[i] == plot_labels[0]:
    #         plt.scatter(X_norm[i:, 0], X_norm[i:, 1], color=colors[0], alpha=0.5)
    #     elif all_labels[i] == plot_labels[1]:
    #         plt.scatter(X_norm[i:, 0], X_norm[i:, 1], color=colors[1], alpha=0.5)
    # plt.title('t-SNE Visualization of TEP')
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')
    # plt.legend()
    # plt.show()
    # # （可视化 - end!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!）
    gen_set = MyData(gen_data, gen_attributelabel, gen_label)
    gen_loader = torch.utils.data.DataLoader(
        gen_set, batch_size=256, shuffle=True
    )
    # test_datas_sce = sce(torch.from_numpy(test_datas_unseen).float(), torch.from_numpy(test_attrs_unseen).float())[
    #     'z1'].detach().numpy()
    # test_datas_sce = sce_scaler.transform(test_datas_sce)
    # test_set = MyData(test_datas_sce, test_attrs_unseen, test_labels_unseen)
    # 测试u_classifier只用未见类

    model = uclassifier
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0005)
    # optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.005)
    # optimizer = optim.AdamW(model.parameters(), lr=0.000001, weight_decay=0.00005)
    # 用生成样本训练未见类分类器
    for epoch in range(1000):
        model.train()
        sum = 0
        count = 0
        train_loss = 0
        for data, _, label in gen_loader:
            label = torch.Tensor(label).long()
            x = data.to(device)
            y = label.to(device)
            o = model(x)
            loss = criterion(o, y)
            output = o.data.cpu().numpy()
            pred = output.argmax(axis=1)
            target = label.numpy()
            sum += (pred == target).sum()
            count += target.shape[0]
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_acc = sum / count
        # 测试
        model.eval()
        sum = 0
        count = 0
        eval_loss = 0
        for data, _, label in test_loader:
            x = torch.reshape(data, (-1, 50)).to(device)
            y = label.long().to(device)
            o = model(x)
            # 早停
            loss = criterion(o, y)
            eval_loss += loss.item()
            output = o.data.cpu().numpy()
            pred = output.argmax(axis=1)
            target = label.numpy()
            sum += (pred == target).sum()
            count += target.shape[0]
        test_acc = sum / count
        print('\tepoch = {}\ttrain accuracy: {:0.3f}\ttest accuracy: {:0.3f}\ttest loss: {:0.3f}'
              .format(epoch, train_acc, test_acc, eval_loss))
        early_stopping(eval_loss, model)
        # 达到早停止条件时，early_stop会被置为True
        if early_stopping.early_stop:
            print("Early stopping")
            break  # 跳出迭代，结束训练
    torch.save(uclassifier.state_dict(), r"D:\Projects\metaGAN\Models\Online\SavedModels\U_Classifier")
    print(r'Saving model U_Classifier successfully to D:\Projects\metaGAN\Models\Online\SavedModels\U_Classifier')


train_classes = [1, 2, 3, 4, 5]
test_classes = [6, 7]
# [6, 7] [1, 2, 3, 4, 5] A
# [4, 5] [1, 2, 3, 6, 7] B
# [1, 3] [2, 4, 5, 6, 7] C
# [1, 2] [3, 4, 5, 6, 7] D
latent_dim = 30
# 用t0x_te训练
per_train_shots = 500
per_test_shots = 480
per_new_shots = 300
sce_scaler = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 注意：未见类样本跟属性也不能一一对应啊（可以知道属性但不能对应），一一对应属性不就相当于知道了标签吗！！！
# 虽然测试样本sce时也输入了样本--属性，但是其实样本嵌入和属性嵌入是分开的(只是为了方便写在了一起，其实没用到属性)，且测试样本也没用来计算损失更新梯度

# # 已见类（p类） + 未见类（标签都是p+1）（训练及测试s_classifier用），只是少了一个true_label，不如直接用create_dataset_comp，而且选择样本时也要用到所有东西
# train_datas, train_attrs, train_labels = creat_dataset(is_train=True)
# scaler = preprocessing.StandardScaler().fit(train_datas)
# train_datas = scaler.transform(train_datas)
# test_datas, test_attrs, test_labels = creat_dataset(is_train=False)
# test_datas = scaler.transform(test_datas)

# 仅有已见类【训练sce、gan用】
train_datas_seen, train_attrs_seen, train_labels_seen = creat_dataset_seen(is_train=True)
# 小波去噪（无敌）A,B,C不适用？
# train_datas_seen = wavelet_denoise_2d(train_datas_seen)
# 后面的数据归一化都要用这个scaler
seen_scaler = preprocessing.StandardScaler().fit(train_datas_seen)
train_datas_seen = seen_scaler.transform(train_datas_seen)
test_datas_seen, test_attrs_seen, test_labels_seen = creat_dataset_seen(is_train=False)
# 小波去噪
# test_datas_seen = wavelet_denoise_2d(test_datas_seen)
test_datas_seen = seen_scaler.transform(test_datas_seen)

# 已见类(p类标签p个) + 未见类(q类标签都是p+1、正常标签)【训练及测试s_classifier，测试GZSL、Online用】
train_datas_comp, train_attrs_comp, train_labels_comp, train_true_labels = create_dataset_comp(is_train=True)
# 小波去噪
# train_datas_comp = wavelet_denoise_2d(train_datas_comp)
train_datas_comp = seen_scaler.transform(train_datas_comp)
test_datas_comp, test_attrs_comp, test_labels_comp, test_true_labels = create_dataset_comp(is_train=False)
# 小波去噪
# test_datas_comp = wavelet_denoise_2d(test_datas_comp)
test_datas_comp = seen_scaler.transform(test_datas_comp)

# 仅有未见类【测试u_classifier用】
train_datas_unseen, train_attrs_unseen, train_labels_unseen = creat_dataset_unseen(is_train=True)
# 小波去噪
# train_datas_unseen = wavelet_denoise_2d(train_datas_unseen)
train_datas_unseen = seen_scaler.transform(train_datas_unseen)
test_datas_unseen, test_attrs_unseen, test_labels_unseen = creat_dataset_unseen(is_train=False)
# 小波去噪
# test_datas_unseen = wavelet_denoise_2d(test_datas_unseen)
test_datas_unseen = seen_scaler.transform(test_datas_unseen)

# 在线增量训练【测试sclassifier_online用】
train_datas_online, train_attrs_online, train_labels_online, train_true_labels_online = creat_dataset_online()
# 小波去噪
# train_datas_online = wavelet_denoise_2d(train_datas_online)
train_datas_online = seen_scaler.transform(train_datas_online)


# # 再用t-sne看一下
# tsne_classes = [1, 2, 3, 6, 7, 4, 5]  #
# tsne_data = []
# for item in range(len(tsne_classes)):
#     tsne_data.append(test_datas_comp[item * 480:(item + 1) * 480, :])
# tsne_data = np.reshape(np.array(tsne_data), (-1, 64))
# tsne = TSNE(n_components=2, random_state=42)
# tsne_results = tsne.fit_transform(tsne_data)
# plt.figure(figsize=(10, 8))
# labels_ = test_true_labels.flatten('A').tolist()
# labels = []
# for i in tsne_classes:
#     for j in labels_:
#         if j == i:
#             labels.append(j)
# labels = np.array(labels)
# x_min, x_max = tsne_results.min(0), tsne_results.max(0)
# X_norm = (tsne_results - x_min) / (x_max - x_min)  # 归一化
# # 处理一下，把不用的标签给删掉
# # labels = np.delete(labels,np.where(labels >= 7))
# for label in np.unique(labels):
#     indices = labels == label  # 把labels中与label标签相等的索引所在的值置为true
#     plt.scatter(X_norm[indices, 0], X_norm[indices, 1], label=f'Fault {label}', alpha=0.5)
# # plt.title('T-SNE Visualization of Centrifugal Chiller', fontsize=25)
# plt.legend(fontsize=20)
# # 设置 X 轴和 Y 轴刻度数字的字体大小
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.show()


# 仅用于GZSL和Online测试
def get_test_loader():
    test_data_sce = get_sce_data_seen(test_datas_comp, test_attrs_comp)
    # 创建数据加载器
    test_set = MyDataFour(test_data_sce, test_attrs_comp, test_labels_comp, test_true_labels)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False
    )
    return test_loader


def test():
    # 加载各模块模型
    S_Classifier = SClassifier(len(train_classes) + 1)
    S_Classifier.load_state_dict(torch.load(r"D:\Projects\metaGAN\Models\Online\SavedModels\S_Classifier"))
    print(r'Load Model successfully from D:\Projects\metaGAN\Models\Online\SavedModels\S_Classifier ')
    U_Classifier = UClassifier(len(test_classes))
    U_Classifier.load_state_dict(torch.load(r"D:\Projects\metaGAN\Models\Online\SavedModels\U_Classifier"))
    print(r'Load Model successfully from D:\Projects\metaGAN\Models\Online\SavedModels\U_Classifier ')
    # print(S_Classifier)
    # print(U_Classifier)
    test_loader = get_test_loader()
    S_Classifier.eval()
    U_Classifier.eval()  # 不加会报错
    num_accS = 0  # 预测正确的已见类个数
    num_accU = 0  # 预测正确的未见类个数
    for data, attr, label, true_label in test_loader:
        x = data.to(torch.float32)
        o = S_Classifier(x)
        outputs = o.data.cpu().numpy()
        pred = outputs.argmax(axis=1)
        for i in range(pred.shape[0]):
            pred_index = pred[i]
            if pred_index < 5:  # 预测为已见类
                if pred_index == label[i]:  # 已见类样本预测正确
                    # 不对，S_Classifier只能预测13类
                    num_accS += 1
            else:  # 预测为未见类，则使用U_Classifier进一步分类
                x_u = x[i:i + 1, :]
                output = U_Classifier(x_u)
                u_pred = output.argmax(axis=1)[0]
                if test_classes[u_pred] == true_label[i]:  # 未见类样本预测正确
                    num_accU += 1
        # print(pred)
    accS = num_accS / (5 * per_test_shots)
    accU = num_accU / (2 * per_test_shots)
    H = 2 * accU * accS / (accU + accS)
    print('\taccS: {:0.5f}\taccU: {:0.5f}\tH: {}'.format(accS, accU, H))


def test_online():
    # 假设在线阶段的未见类样本都是无标签的, 已见类都是有标签的
    # 只需要修改S_Classifier的参数！
    early_stopping = EarlyStoppingOnline('')
    sclassifier_online = SClassifier_Online(len(train_classes) + 1)
    # 在线分类器也得加载老模型的参数啊！
    s_classifier = SClassifier(len(train_classes) + 1).to(device)
    s_classifier.load_state_dict(torch.load(r"D:\Projects\metaGAN\Models\Online\SavedModels\S_Classifier"))
    sclassifier_online.load_state_dict(torch.load(r"D:\Projects\metaGAN\Models\Online\SavedModels\S_Classifier"))
    train_datas_sce = get_sce_data_seen(train_datas_online, train_attrs_online)
    # 定义数据集
    online_set = MyDataIndex(train_datas_sce, train_attrs_online, train_labels_online, train_true_labels_online)
    online_loader = torch.utils.data.DataLoader(
        online_set, batch_size=256, shuffle=True
    )
    test_loader = get_test_loader()
    # 初始化模型和优化器
    model = sclassifier_online.to(device)
    # optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.00005)
    optimizer = optim.AdamW(model.parameters(), lr=0.000005, weight_decay=0.00005)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss(reduction='none')
    # 定义知识蒸馏损失函数
    # kd_loss_fn = KnowledgeDistillationLoss(temperature=2.0)
    # 存储每个样本的权重？
    weights_all = []
    weights_index = []  # 每个样本的真实序号
    y_true = []
    y_pred = []
    # 训练循环
    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        sum = 0
        count = 0
        weights_all = []
        weights_index = []
        y_true = []
        y_pred = []
        for index, input, attr, label, true_label in online_loader:
            x = input.float().to(device)
            y = label.long().to(device)
            outputs_student = model(x)
            # outputs_teacher = s_classifier(x)
            loss = criterion(outputs_student, y)
            # 计算 Self-paced sample weighting 的权重
            weights = self_paced_weighting(loss, 1)
            weights_index.extend((index + 1).tolist())
            weights_all.extend(weights.tolist())
            # 计算加权损失
            weighted_loss = (weights * loss).mean() - 1 * torch.sum(weights)
            # 计算蒸馏损失
            # kd_loss = kd_loss_fn(outputs_student, outputs_teacher)
            # total_loss = 0.8 * kd_loss + 0.2 * weighted_loss
            total_loss = weighted_loss
            pred = outputs_student.argmax(axis=1).cpu().numpy()
            target = label.cpu().numpy()
            sum += (pred == target).sum()
            count += target.shape[0]
            optimizer.zero_grad()
            # 反向传播和优化
            total_loss.backward()
            optimizer.step()
        train_acc = sum / count
        model.eval()
        sum = 0
        count = 0
        eval_loss = 0
        for data, attr, label, true_label in test_loader:
            x = data.float().to(device)
            y = label.to(device)
            o = model(x)
            loss = torch.nn.CrossEntropyLoss()(o, y.long()).to(device)
            eval_loss += loss.item()
            output = o.data.cpu().numpy()
            pred = output.argmax(axis=1)
            target = label.cpu().numpy()
            y_pred.append(pred)
            y_true.append(target)
            sum += (pred == target).sum()
            count += target.shape[0]
        test_acc = sum / count
        print('{}\tepoch = {}\ttrain accuracy: {:0.3f}\ttest accuracy: {:0.3f}\ttest loss: {:0.3f}' \
              .format(datetime.now(), epoch, train_acc, test_acc, eval_loss))
        early_stopping(eval_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    # 画个所有样本权重的散点图？
    # plot_weights(weights_all, weights_index)
    plot_radar(weights_all)

    # 画出混淆矩阵
    y_pred = np.concatenate(y_pred, dtype='int32')
    y_true = np.concatenate(y_true, dtype='int32')
    C = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5])  # 可将'1'等替换成自己的类别，如'cat'。
    # plt.figure(figsize=(12, 12))
    plt.matshow(C, cmap=plt.cm.Reds)  # 根据最下面的图按自己需求更改颜色
    plt.colorbar()
    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center', fontsize=15)
    # plt.tick_params(labelsize=15)  # 设置左边和上面的label类别如0,1,2,3,4的字体大小。
    # plt.title('Confusion Matrix', fontsize=20)
    plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 17})  # 设置字体大小。
    plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 17})
    plt.xticks(range(0, 6), labels=['f1', 'f2', 'f3', 'f4', 'f5', 'unk'], fontsize=15)  # 将x轴或y轴坐标，刻度 替换为文字/字符
    plt.yticks(range(0, 6), labels=['f1', 'f2', 'f3', 'f4', 'f5', 'unk'], fontsize=15)
    plt.show()

    # 创建数据加载器
    U_Classifier = UClassifier(len(test_classes)).to(device)
    U_Classifier.load_state_dict(torch.load(r"D:\Projects\metaGAN\Models\Online\SavedModels\U_Classifier"))
    sclassifier_online.eval()
    U_Classifier.eval()  # 不加会报错
    num_accS = 0  # 预测正确的已见类个数
    num_accU = 0  # 预测正确的未见类个数
    for data, attr, label, true_label in test_loader:
        x = data.to(torch.float32).to(device)
        o = sclassifier_online(x)
        outputs = o.data.cpu().numpy()
        pred = outputs.argmax(axis=1)
        for i in range(pred.shape[0]):
            pred_index = pred[i]
            if pred_index < 5:  # 预测为已见类
                if pred_index == label[i]:  # 已见类样本预测正确
                    # 不对，S_Classifier只能预测13类
                    num_accS += 1
            else:  # 预测为未见类，则使用U_Classifier进一步分类
                x_u = x[i:i + 1, :]
                output = U_Classifier(x_u)
                u_pred = output.argmax(axis=1)[0]
                if test_classes[u_pred] == true_label[i]:  # 未见类样本预测正确
                    num_accU += 1
        # print(pred)
    accS = num_accS / (5 * per_test_shots)
    accU = num_accU / (2 * per_test_shots)
    H = 2 * accU * accS / (accU + accS)
    print('\taccS: {:0.5f}\taccU: {:0.5f}\tH: {}'.format(accS, accU, H))


def plot_radar(data_list):
    # 将区间分成 len(bins)-1个
    bins = [0.00, 0.70, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00]
    num_bins = len(bins)-1

    # 使用 numpy.histogram 函数统计每个区间的元素个数
    hist, edges = np.histogram(data_list, bins=bins)

    # 打印结果
    for i in range(num_bins):
        print(f"区间 {i + 1}: {hist[i]} 个元素")

    # 共有len(bins)-1个值
    num_variables = len(bins)-1
    values = [num for num in hist]

    # 计算角度
    theta = np.linspace(0, 2 * np.pi, num_variables, endpoint=False)

    # 创建雷达图
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=300)

    # 为每个变量指定不同颜色
    colors = plt.cm.rainbow(np.linspace(0, 1, num_variables))

    # 填充连线内部区域
    ax.fill_between(np.concatenate((theta, [theta[0]])), 0, np.concatenate((values, [values[0]])), color='#DCF1F9',
                    alpha=1)

    # 画点
    ax.scatter(theta, values, color=colors, zorder=3)

    # 画线连接不同点
    ax.plot(np.concatenate((theta, [theta[0]])), np.concatenate((values, [values[0]])), color='#0D4DA1', alpha=1, linewidth=1.5)


    # 添加标签
    # 将0~1分为25个区间
    intervals = bins
    # 将每个区间的字符串形式放入列表
    interval_strings = [f'{intervals[i]:.2f}-{intervals[i+1]:.2f}' for i in range(len(intervals)-1)]

    # 设置标签，设置标签离雷达图的圆的距离
    ax.set_rlabel_position(90)
    ax.set_xticks(theta)
    ax.set_xticklabels(interval_strings, fontsize=18)

    # 设置值的字体大小
    ax.set_yticklabels([])  # 先将默认的值标签清除
    ax.set_yticklabels([f'{int(val)}' for val in ax.get_yticks()], fontsize=18)

    # 显示雷达图
    plt.show()
    return


def plot_weights(weights, indexs):
    # 创建一个图形对象
    fig = plt.figure(dpi=250)
    # 在图形对象上添加一个三维子图
    ax = fig.add_subplot(111, projection='3d')
    # 合并为字典，并按照keys排序
    combined_dict = dict(sorted(zip(indexs, weights), key=lambda item: item[0]))
    # 注意weights, indexs都是乱序排列且一一对应的
    x = np.arange(1, 8, dtype=float)  # 对应7个类
    y = np.arange(1, 301, dtype=float)  # 对应每个类型的300个样本编号
    z = np.arange(0, 7 * 300, dtype=float)  # 表示7 * 300个元素的权重
    # 通过键遍历字典并获取对应的值
    index = 0
    for key in combined_dict:
        value = combined_dict[key]
        z[index] = value
        index += 1
    x, y = np.meshgrid(x, y)
    z = z.reshape((300, 7))
    # # 使用列表解析筛选值为1到300的元素
    # for i in range(len(indexs)):
    #     if 1 <= indexs[i] <= 300:
    #         # 将indexs[i]在300个元素中的位置对应元素设置为权重
    #         y_[indexs[i]-1] = weights[i]
    # y = y_
    # 绘制散点图
    # 自定义颜色列表，确保每个x值对应不同的颜色
    # 对 x 进行唯一化处理，为每个唯一的 x 值分配一个颜色
    unique_x, color_indices = np.unique(x, return_inverse=True)

    # 自定义颜色列表，确保与 unique_x 的长度一致
    custom_colors = ['#818181', '#2A5522', '#BF9895', '#E07E35', '#F2CCA0', '#A9C4E6', '#D1392B']

    # 根据索引获取每个 x 值对应的颜色
    colors = [custom_colors[i] for i in color_indices]
    ax.scatter(x, y, z, c=colors, cmap='viridis', s=13)
    # # 使用plot函数绘制三维线图，并指定颜色
    # for i in range(len(y)):
    #     ax.plot(x[i, :], y[i, :], z[i, :], color=plt.cm.viridis(i / len(y)))
    ax.set_xlabel('Faults', fontsize=18)
    ax.set_ylabel('Samples', fontsize=18)
    ax.set_zlabel('Weights', fontsize=18)
    # ax.xticks(fontsize=16)
    # ax.yticks(fontsize=16)
    # ax.zticks(fontsize=16)
    # plt.subplots_adjust(bottom=0.15)
    plt.show()

start_time = 0
end_time = 0
process = psutil.Process(os.getpid())
start_mem = 0
end_mem = 0
def train():
    start_time = time.time()
    start_mem = process.memory_full_info().rss
    if input('Train SCE? y/n\n').lower() == 'y':
        train_sce()
    # if input('Plot 3d Matrix? y/n\n').lower() == 'y':
    #     ploy_matrix_3d()
    if input('Train S_Classifier? y/n\n').lower() == 'y':
        train_sclassifier(len(train_classes) + 1)
    if input('Train U_Classifier? y/n\n').lower() == 'y':
        train_uclassifier(len(test_classes))
        end_time = time.time()
        end_mem = process.memory_info().rss
        print(f"NAN实际运行时间: {end_time - start_time} 秒")
        # 单位是字节，转换为兆字节
        print(f"内存占用: {(end_mem - start_mem) / (1024 * 1024)} MB")
    if input('Test GZSL? y/n\n').lower() == 'y':
        test()
    if input('Test Online? y/n\n').lower() == 'y':
        start_time = time.time()
        test_online()
        end_time = time.time()
        print(f"IL实际运行时间: {end_time - start_time} 秒")

train()
