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
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
from torch import device
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from Models_Water_NoSCE import Embedding_Net, SClassifier, Generator, Discriminator, UClassifier, SClassifier_Online
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
    optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.000005)  # 对B不好？
    train_acc_SCls = []
    test_acc_SCls = []
    epoch_accS = []
    y_true = []
    y_pred = []
    epochs = 700
    for epoch in range(epochs):
        y_true = []
        y_pred = []
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

# 定义 Self-paced sample weighting 的权重调整函数
def self_paced_weighting(loss, lambda_param_2=1):
    return torch.exp(-loss / lambda_param_2)


def train_uclassifier(output_dim=3):
    attribute_matrix_ = pd.read_excel(r'D:\Projects\metaGAN\attribute_matrix_water.xlsx', index_col='no')
    attribute_matrix = attribute_matrix_.values
    early_stopping = EarlyStoppingUCls('')
    # 训练GAN用的是已见类
    train_set = MyData(train_datas_seen, train_attrs_seen, train_labels_seen)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=256, shuffle=True
    )
    test_set = MyData(test_datas_unseen, test_attrs_unseen, test_labels_unseen)
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
                real_faults = real_faults.view(-1, 64).to(device)
                attrs = attrs.view(-1, 12).to(device)
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
    print('生成样本维度：', gen_fault.shape)  # 特征维度
    global seen_scaler
    gen_fault = seen_scaler.transform(gen_fault.cpu().detach().numpy())
    gen_data = torch.from_numpy(gen_fault).to(device)
    gen_set = MyData(gen_data, gen_attributelabel, gen_label)
    gen_loader = torch.utils.data.DataLoader(
        gen_set, batch_size=256, shuffle=True
    )
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
            x = torch.reshape(data, (-1, 64)).to(device)
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


train_classes = [1, 2, 3, 6, 7]
test_classes = [4, 5]
# [6, 7] [1, 2, 3, 4, 5] A
# [4, 5] [1, 2, 3, 6, 7] B
# [1, 3] [2, 4, 5, 6, 7] C
# [1, 2] [3, 4, 5, 6, 7] D
latent_dim = 30
# 用t0x_te训练
per_train_shots = 500
per_test_shots = 480
per_new_shots = 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 注意：未见类样本跟属性也不能一一对应啊（可以知道属性但不能对应），一一对应属性不就相当于知道了标签吗！！！

# # 已见类（p类） + 未见类（标签都是p+1）（训练及测试s_classifier用），只是少了一个true_label，不如直接用create_dataset_comp，而且选择样本时也要用到所有东西
# train_datas, train_attrs, train_labels = creat_dataset(is_train=True)
# scaler = preprocessing.StandardScaler().fit(train_datas)
# train_datas = scaler.transform(train_datas)
# test_datas, test_attrs, test_labels = creat_dataset(is_train=False)
# test_datas = scaler.transform(test_datas)

# 仅有已见类【训练gan用】
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
    # 定义数据集
    online_set = MyDataIndex(train_datas_online, train_attrs_online, train_labels_online, train_true_labels_online)
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


def plot_weights(weights, indexs):
    # 类1
    # 注意weights, indexs都是乱序排列且一一对应的
    x = np.arange(1, 301, dtype=float)
    y_ = np.arange(1, 301, dtype=float)  # 表示300个元素中第1-300个元素的权重
    # 使用列表解析筛选值为1到300的元素
    for i in range(len(indexs)):
        if 1 <= indexs[i] <= 300:
            # 将indexs[i]在300个元素中的位置对应元素设置为权重
            y_[indexs[i]-1] = weights[i]
    y = y_
    # 绘制散点图
    plt.scatter(x, y, marker='o')
    plt.xlabel('Sample number', fontsize=17)
    plt.ylabel('Sample weight', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.subplots_adjust(bottom=0.15)
    plt.show()
    # 类2
    x = np.arange(1, 301, dtype=float)
    y_ = np.arange(1, 301, dtype=float)  # 表示300个元素中第1-300个元素的权重
    # 使用列表解析筛选值为1到300的元素
    for i in range(len(indexs)):
        if 301 <= indexs[i] <= 600:
            # 将indexs[i]在300个元素中的位置对应元素设置为权重
            y_[indexs[i] - 1 - 300] = weights[i]
    y = y_
    # 绘制散点图
    plt.scatter(x, y, marker='o')
    plt.xlabel('Sample number', fontsize=17)
    plt.ylabel('Sample weight', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.subplots_adjust(bottom=0.15)
    plt.show()
    # 类3
    x = np.arange(1, 301, dtype=float)
    y_ = np.arange(1, 301, dtype=float)  # 表示300个元素中第1-300个元素的权重
    # 使用列表解析筛选值为1到300的元素
    for i in range(len(indexs)):
        if 601 <= indexs[i] <= 900:
            # 将indexs[i]在300个元素中的位置对应元素设置为权重
            y_[indexs[i] - 1 - 600] = weights[i]
    y = y_
    # 绘制散点图
    plt.scatter(x, y, marker='o')
    plt.xlabel('Sample number', fontsize=17)
    plt.ylabel('Sample weight', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.subplots_adjust(bottom=0.15)
    plt.show()
    # 类4
    x = np.arange(1, 301, dtype=float)
    y_ = np.arange(1, 301, dtype=float)  # 表示300个元素中第1-300个元素的权重
    # 使用列表解析筛选值为1到300的元素
    for i in range(len(indexs)):
        if 901 <= indexs[i] <= 1200:
            # 将indexs[i]在300个元素中的位置对应元素设置为权重
            y_[indexs[i] - 1 - 900] = weights[i]
    y = y_
    # 绘制散点图
    plt.scatter(x, y, marker='o')
    plt.xlabel('Sample number', fontsize=17)
    plt.ylabel('Sample weight', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.subplots_adjust(bottom=0.15)
    plt.show()
    # 类5
    x = np.arange(1, 301, dtype=float)
    y_ = np.arange(1, 301, dtype=float)  # 表示300个元素中第1-300个元素的权重
    # 使用列表解析筛选值为1到300的元素
    for i in range(len(indexs)):
        if 1201 <= indexs[i] <= 1500:
            # 将indexs[i]在300个元素中的位置对应元素设置为权重
            y_[indexs[i] - 1 - 1200] = weights[i]
    y = y_
    # 绘制散点图
    plt.scatter(x, y, marker='o')
    plt.xlabel('Sample number', fontsize=17)
    plt.ylabel('Sample weight', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.subplots_adjust(bottom=0.15)
    plt.show()

def train():
    # if input('Plot 3d Matrix? y/n\n').lower() == 'y':
    #     ploy_matrix_3d()
    if input('Train S_Classifier? y/n\n').lower() == 'y':
        train_sclassifier(len(train_classes) + 1)
    if input('Train U_Classifier? y/n\n').lower() == 'y':
        train_uclassifier(len(test_classes))
    if input('Test GZSL? y/n\n').lower() == 'y':
        test()
    if input('Test Online? y/n\n').lower() == 'y':
        test_online()


train()
