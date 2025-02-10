import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
from sklearn import preprocessing
from scipy.stats import entropy
import pywt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
from torch import device
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from Models_NoSCE import Embedding_Net, SClassifier, Generator, Discriminator, UClassifier, SClassifier_Online
from early_stopping_online import EarlyStoppingOnline
from early_stopping import EarlyStopping
from early_stopping_ucls import EarlyStoppingUCls

def get_data_list(path, is_train):
    fault1 = loadmat(path + 'd01_te.mat')['data'][:, 160:660] if is_train else loadmat(path + 'd01.mat')['data'][:, :]
    fault2 = loadmat(path + 'd02_te.mat')['data'][:, 160:660] if is_train else loadmat(path + 'd02.mat')['data'][:, :]
    fault3 = loadmat(path + 'd03_te.mat')['data'][:, 160:660] if is_train else loadmat(path + 'd03.mat')['data'][:, :]
    fault4 = loadmat(path + 'd04_te.mat')['data'][:, 160:660] if is_train else loadmat(path + 'd04.mat')['data'][:, :]
    fault5 = loadmat(path + 'd05_te.mat')['data'][:, 160:660] if is_train else loadmat(path + 'd05.mat')['data'][:, :]
    fault6 = loadmat(path + 'd06_te.mat')['data'][:, 160:660] if is_train else loadmat(path + 'd06.mat')['data'][:, :]
    fault7 = loadmat(path + 'd07_te.mat')['data'][:, 160:660] if is_train else loadmat(path + 'd07.mat')['data'][:, :]
    fault8 = loadmat(path + 'd08_te.mat')['data'][:, 160:660] if is_train else loadmat(path + 'd08.mat')['data'][:, :]
    fault9 = loadmat(path + 'd09_te.mat')['data'][:, 160:660] if is_train else loadmat(path + 'd09.mat')['data'][:, :]
    fault10 = loadmat(path + 'd10_te.mat')['data'][:, 160:660] if is_train else loadmat(path + 'd10.mat')['data'][:, :]
    fault11 = loadmat(path + 'd11_te.mat')['data'][:, 160:660] if is_train else loadmat(path + 'd11.mat')['data'][:, :]
    fault12 = loadmat(path + 'd12_te.mat')['data'][:, 160:660] if is_train else loadmat(path + 'd12.mat')['data'][:, :]
    fault13 = loadmat(path + 'd13_te.mat')['data'][:, 160:660] if is_train else loadmat(path + 'd13.mat')['data'][:, :]
    fault14 = loadmat(path + 'd14_te.mat')['data'][:, 160:660] if is_train else loadmat(path + 'd14.mat')['data'][:, :]
    fault15 = loadmat(path + 'd15_te.mat')['data'][:, 160:660] if is_train else loadmat(path + 'd15.mat')['data'][:, :]
    data_list = [fault1, fault2, fault3, fault4, fault5, fault6, fault7, fault8, fault9, fault10, fault11, fault12,
                 fault13, fault14, fault15]
    return data_list


def creat_dataset(is_train=True):
    path = r'D:\Projects\metaGAN\Datasets\TE_mat_data\\'
    print("loading data...")
    zero_test_index = [index - 1 for index in test_classes]  # test_index对应的是类别序号，-1才是索引
    attribute_matrix_ = pd.read_excel('D:/Projects/metaGAN/attribute_matrix.xlsx', index_col='no')
    attribute_matrix = attribute_matrix_.values
    train_index = list(set(np.arange(15)) - set(zero_test_index))
    zero_test_index.sort()
    train_index.sort()
    print("test classes: {}".format(test_classes))
    print("train classes: {}".format(train_classes))
    data_list = get_data_list(path, is_train)
    train_attributelabel = []
    traindata = []
    train_label = []
    # 遍历训练故障类型(12个)
    for item in range(len(train_index)):
        train_attributelabel += [attribute_matrix[train_index[item], :]] * per_train_shots if is_train else [attribute_matrix[
                                                                                                 train_index[item],
                                                                                                 :]] * per_test_shots
        traindata.append(data_list[train_index[item]])
        train_label += [item] * per_train_shots if is_train else [item] * per_test_shots
    train_label = np.row_stack(train_label)
    train_attributelabel = np.row_stack(train_attributelabel)
    traindata = np.column_stack(traindata).T
    # 将未见类样本加入
    data = []
    attributelabel = []
    label = []
    for item in range(len(zero_test_index)):
        label += [len(train_classes)] * per_train_shots if is_train else [len(train_classes)] * per_test_shots
        attributelabel += [attribute_matrix[zero_test_index[item], :]] * per_train_shots if is_train else [attribute_matrix[
                                                                                               zero_test_index[item],
                                                                                               :]] * per_test_shots
        data.append(data_list[zero_test_index[item]])
    data = np.column_stack(data).T
    label = np.row_stack(label)
    attributelabel = np.row_stack(attributelabel)
    traindata = np.vstack((traindata, data))
    train_attributelabel = np.vstack((train_attributelabel, attributelabel))
    train_label = np.vstack((train_label, label))
    return traindata, train_attributelabel, train_label


def creat_dataset_seen(is_train=True):
    path = r'D:\Projects\metaGAN\Datasets\TE_mat_data\\'
    print("loading data...")
    seen_index = [index - 1 for index in train_classes]
    attribute_matrix_ = pd.read_excel('D:/Projects/metaGAN/attribute_matrix.xlsx', index_col='no')
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
    data = np.column_stack(data).T
    label = np.row_stack(label)
    attributelabel = np.row_stack(attributelabel)
    return data, attributelabel, label


def creat_dataset_unseen(is_train=True):
    path = r'D:\Projects\metaGAN\Datasets\TE_mat_data\\'
    print("loading data...")
    zero_test_index = [index - 1 for index in test_classes]  # test_index对应的是类别序号，-1才是索引
    attribute_matrix_ = pd.read_excel('D:/Projects/metaGAN/attribute_matrix.xlsx', index_col='no')
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
        attributelabel += [attribute_matrix[zero_test_index[item], :]] * per_train_shots if is_train else [attribute_matrix[
                                                                                               zero_test_index[item],
                                                                                               :]] * per_test_shots
        data.append(data_list[zero_test_index[item]])
    data = np.column_stack(data).T
    label = np.row_stack(label)
    attributelabel = np.row_stack(attributelabel)
    return data, attributelabel, label


def create_dataset_comp(is_train=True):
    path = r'D:\Projects\metaGAN\Datasets\TE_mat_data\\'
    print("loading data...")
    zero_test_index = [index - 1 for index in test_classes]  # test_index对应的是类别序号，-1才是索引
    attribute_matrix_ = pd.read_excel('D:/Projects/metaGAN/attribute_matrix.xlsx', index_col='no')
    attribute_matrix = attribute_matrix_.values
    train_index = list(set(np.arange(15)) - set(zero_test_index))
    zero_test_index.sort()
    train_index.sort()
    print("test classes: {}".format(test_classes))
    print("train classes: {}".format(train_classes))
    data_list = get_data_list(path, is_train)
    train_attributelabel = []
    traindata = []
    train_label = []
    train_true_label = []
    # 遍历训练故障类型(12个)
    for item in range(len(train_index)):
        train_attributelabel += [attribute_matrix[train_index[item], :]] * per_train_shots if is_train else [attribute_matrix[
                                                                                                 train_index[item],
                                                                                                 :]] * per_test_shots
        traindata.append(data_list[train_index[item]])
        train_label += [item] * per_train_shots if is_train else [item] * per_test_shots
        train_true_label += [train_classes[item]] * per_train_shots if is_train else [train_classes[
                                                                              item]] * per_test_shots  # 真实标签就直接按1~15来吧

    train_label = np.row_stack(train_label)
    train_true_label = np.row_stack(train_true_label)
    train_attributelabel = np.row_stack(train_attributelabel)
    traindata = np.column_stack(traindata).T
    # 将未见类样本加入
    data = []
    attributelabel = []
    label = []
    true_label = []
    for item in range(len(zero_test_index)):
        label += [len(train_classes)] * per_train_shots if is_train else [len(train_classes)] * per_test_shots
        attributelabel += [attribute_matrix[zero_test_index[item], :]] * per_train_shots if is_train else [attribute_matrix[
                                                                                               zero_test_index[item],
                                                                                               :]] * per_test_shots
        data.append(data_list[zero_test_index[item]])
        true_label += [test_classes[item]] * per_train_shots if is_train else [test_classes[item]] * per_test_shots
    data = np.column_stack(data).T
    label = np.row_stack(label)
    attributelabel = np.row_stack(attributelabel)
    true_label = np.row_stack(true_label)
    traindata = np.vstack((traindata, data))
    train_attributelabel = np.vstack((train_attributelabel, attributelabel))
    train_label = np.vstack((train_label, label))
    train_true_label = np.vstack((train_true_label, true_label))
    return traindata, train_attributelabel, train_label, train_true_label


def creat_dataset_online():
    path = r'D:\Projects\metaGAN\Datasets\TE_mat_data\\'
    print("loading data...")
    fault1 = loadmat(path + 'd01_te.mat')['data'][:, 660:960]
    fault2 = loadmat(path + 'd02_te.mat')['data'][:, 660:960]
    fault3 = loadmat(path + 'd03_te.mat')['data'][:, 660:960]
    fault4 = loadmat(path + 'd04_te.mat')['data'][:, 660:960]
    fault5 = loadmat(path + 'd05_te.mat')['data'][:, 660:960]
    fault6 = loadmat(path + 'd06_te.mat')['data'][:, 660:960]
    fault7 = loadmat(path + 'd07_te.mat')['data'][:, 660:960]
    fault8 = loadmat(path + 'd08_te.mat')['data'][:, 660:960]
    fault9 = loadmat(path + 'd09_te.mat')['data'][:, 660:960]
    fault10 = loadmat(path + 'd10_te.mat')['data'][:, 660:960]
    fault11 = loadmat(path + 'd11_te.mat')['data'][:, 660:960]
    fault12 = loadmat(path + 'd12_te.mat')['data'][:, 660:960]
    fault13 = loadmat(path + 'd13_te.mat')['data'][:, 660:960]
    fault14 = loadmat(path + 'd14_te.mat')['data'][:, 660:960]
    fault15 = loadmat(path + 'd15_te.mat')['data'][:, 660:960]
    # 假设有300样本用来增量训练，480用来测试
    unseen_index = [index - 1 for index in test_classes]  # test_classes对应的是类别序号，-1才是索引
    seen_index = [index - 1 for index in train_classes]
    attribute_matrix_ = pd.read_excel('D:/Projects/metaGAN/attribute_matrix.xlsx', index_col='no')
    attribute_matrix = attribute_matrix_.values
    seen_index.sort()
    unseen_index.sort()
    print("seen classes: {}".format(train_classes))
    print("unseen classes: {}".format(test_classes))
    # 用于增量训练的data_list
    new_data_list = [fault1, fault2, fault3, fault4, fault5, fault6, fault7, fault8, fault9, fault10, fault11, fault12,
                     fault13, fault14, fault15]
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
    new_data = np.column_stack(new_data).T
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
    dim = [20, 40]  # 属性有20个，提取后为40个，则特征也是40个
    sce = Embedding_Net(dim, lambda_=lambda_)
    sce.load_state_dict(torch.load(load_path))
    print('Load Model successfully from [%s]' % load_path)
    return sce


def train_sclassifier(output_dim=12):
    early_stopping = EarlyStopping('')
    train_set = MyDataFour(train_datas_comp, train_attrs_comp, train_labels_comp, train_true_labels)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=256, shuffle=True
    )
    test_set = MyDataFour(test_datas_comp, test_attrs_comp, test_labels_comp, test_true_labels)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False
    )
    model = SClassifier(output_dim=output_dim).to(device)
    # optimizer = optim.AdamW(model.parameters(), lr=0.000005, weight_decay=0.00005)
    optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.00005)  #对B不好？

    for epoch in range(1000):
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
            sum += (pred == target).sum()
            count += target.shape[0]
        test_acc = sum / count
        print('{}\tepoch = {}\ttrain accuracy: {:0.3f}\ttest accuracy: {:0.3f}\ttest loss: {:0.3f}' \
              .format(datetime.now(), epoch, train_acc, test_acc, eval_loss))
        early_stopping(eval_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break


# 定义 Self-paced sample weighting 的权重调整函数
def self_paced_weighting(loss, lambda_param_2=1):
    return torch.exp(-loss / lambda_param_2)


def train_uclassifier(output_dim=3):
    attribute_matrix_ = pd.read_excel(r'D:\Projects\metaGAN\attribute_matrix.xlsx', index_col='no')
    attribute_matrix = attribute_matrix_.values
    early_stopping = EarlyStoppingUCls('')
    # 训练GAN用的是已见类
    train_set = MyData(train_datas_seen, train_attrs_seen, train_labels_seen)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=256, shuffle=True
    )
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    uclassifier = UClassifier(output_dim).to(device)
    # Define loss functions
    criterion = nn.BCELoss()
    # Define optimizers for the generator and discriminator
    optimizer_G = optim.Adam(generator.parameters(), lr=0.00001)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.00001)
    num_epochs = 500
    if input('Train GAN? y/n\n').lower() == 'y':
        for epoch in range(num_epochs):
            for batch_idx, (real_faults, attrs, _) in enumerate(train_loader):
                real_faults = real_faults.view(-1, 52).to(device)
                attrs = attrs.view(-1, 20).to(device)
                batch_size = real_faults.size(0)
                real_labels = torch.ones(batch_size, 1).to(device)
                fake_labels = torch.zeros(batch_size, 1).to(device)
                # Train the discriminator
                optimizer_D.zero_grad()
                outputs = discriminator(real_faults.float(), attrs)
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
        gen_label += [item] * 280  # 280好用
        gen_attributelabel += [attribute_matrix[test_classes[item] - 1, :]] * 280  # test_index是类别号，索引要-1
    gen_label = np.row_stack(gen_label)
    gen_attributelabel = np.row_stack(gen_attributelabel)
    z = torch.randn(gen_attributelabel.shape[0], latent_dim).to(device)
    gen_fault = generator(z, torch.Tensor(gen_attributelabel).to(device)).to(device)
    print('生成样本维度：', gen_fault.shape)  # 特征就是sce后的维度
    global seen_scaler
    gen_fault = seen_scaler.transform(gen_fault.cpu().detach().numpy())
    gen_data = torch.from_numpy(gen_fault).to(device)
    # # 用t-sne看看生成的样本什么样！！！一坨答辩！！！
    # real_num = 280  # 要画的真实样本的个数
    # real_data = np.vstack((train_datas_sce[0 * per_test_shots:real_num, :],
    #                        train_datas_sce[1 * per_test_shots:1 * per_test_shots + real_num, :],
    #                        train_datas_sce[2 * per_test_shots:2 * per_test_shots + real_num, :]))
    # # 把真实样本在t-sne的标签设为3，4，5
    # real_labels = np.hstack(([3] * real_num, [4] * real_num, [5] * real_num)).T
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
    # plot_labels = [3, 1]
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
    gen_set = MyData(gen_data, gen_attributelabel, gen_label)
    gen_loader = torch.utils.data.DataLoader(
        gen_set, batch_size=256, shuffle=True
    )
    # test_datas_sce = sce(torch.from_numpy(test_datas_unseen).float(), torch.from_numpy(test_attrs_unseen).float())[
    #     'z1'].detach().numpy()
    # test_datas_sce = sce_scaler.transform(test_datas_sce)
    # test_set = MyData(test_datas_sce, test_attrs_unseen, test_labels_unseen)
    # 测试u_classifier只用未见类
    test_set = MyData(test_datas_unseen, test_attrs_unseen, test_labels_unseen)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False
    )
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
            x = torch.reshape(data, (-1, 52)).to(device)
            y = label.long().to(device)
            o = model(x.float())
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


train_classes = [1, 2, 3, 4, 5, 6, 7, 9, 10, 13, 14, 15]
test_classes = [8, 11, 12]
# [1, 6, 14] [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15]
# [4, 7, 10] [1, 2, 3, 5, 6, 8, 9, 11, 12, 13, 14, 15]
# [8, 11, 12] [1, 2, 3, 4, 5, 6, 7, 9, 10, 13, 14, 15]
# [2, 3, 5] [1, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] D
# [9, 13, 15] [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14] E
latent_dim = 20
# 用t00_te训练
per_train_shots = 500
per_test_shots = 480
per_new_shots = 300
sce_scaler = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 注意：样本跟属性也不能一一对应啊（可以知道属性但不能对应），一一对应属性不就相当于知道了标签吗！！！
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


# 仅用于GZSL和Online测试
def get_test_loader():
    # 创建数据加载器
    test_set = MyDataFour(test_datas_comp, test_attrs_comp, test_labels_comp, test_true_labels)
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
            if pred_index < 12:  # 预测为已见类
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
    accS = num_accS / (12 * per_test_shots)
    accU = num_accU / (3 * per_test_shots)
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
    # 定义数据集
    online_set = MyDataFour(train_datas_online, train_attrs_online, train_labels_online, train_true_labels_online)
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
    # 训练循环
    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        sum = 0
        count = 0
        for input, attr, label, true_label in online_loader:
            x = input.float().to(device)
            y = label.long().to(device)
            outputs_student = model(x)
            # outputs_teacher = s_classifier(x)
            loss = criterion(outputs_student, y)
            # 计算 Self-paced sample weighting 的权重
            weights = self_paced_weighting(loss, 1)
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
            sum += (pred == target).sum()
            count += target.shape[0]
        test_acc = sum / count
        print('{}\tepoch = {}\ttrain accuracy: {:0.3f}\ttest accuracy: {:0.3f}\ttest loss: {:0.3f}' \
              .format(datetime.now(), epoch, train_acc, test_acc, eval_loss))
        early_stopping(eval_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
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
            if pred_index < 12:  # 预测为已见类
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
    accS = num_accS / (12 * per_test_shots)
    accU = num_accU / (3 * per_test_shots)
    H = 2 * accU * accS / (accU + accS)
    print('\taccS: {:0.5f}\taccU: {:0.5f}\tH: {}'.format(accS, accU, H))


def train():
    if input('Train S_Classifier? y/n\n').lower() == 'y':
        train_sclassifier(len(train_classes) + 1)
    if input('Train U_Classifier? y/n\n').lower() == 'y':
        train_uclassifier(len(test_classes))
    if input('Test GZSL? y/n\n').lower() == 'y':
        test()
    if input('Test Online? y/n\n').lower() == 'y':
        test_online()


train()
