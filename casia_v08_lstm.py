import sys
import os
import torch
import torch.nn as nn
import json
import linecache
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.mobile_optimizer import optimize_for_mobile
from tqdm import tqdm

# 设备选择
use_gpu = True     # 是否启用GPU
device = torch.device("cuda:0" if (torch.cuda.is_available() and use_gpu) else "cpu")

# 模型参数
epochs = 900       # 迭代次数
batch_size = 500  # 每个批次样本数
input_dim = 320    # 汉字笔画点最大总数
embedding_dim = 4  # 笔画点向量大小
output_dim = 3755  # 输出维度(识别汉字类别数量)
lr = 0.001         # 学习率


def load_label(filename):
    f = open(filename, 'r', encoding='utf-8')
    labels = {}
    set_label = set()
    for each in f:
        each = each.strip()
        labels[each] = len(labels)
        set_label.add(each)
    f.close()
    return labels

class LazyTextDataset(Dataset):
    def __init__(self, filename):
        self.filename = filename
        self.num_entries = self._get_n_lines(self.filename)

    @staticmethod
    def _get_n_lines(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            for line_idx, _ in enumerate(f, 1):
                pass

        return line_idx

    def __getitem__(self, idx):

        # linecache starts counting from one, not zero, +1 the given index
        idx += 1
        line = linecache.getline(self.filename, idx)
        map_json = json.loads(line)
        point_data = map_json['data'] + [[0, 0, 0, 0] for i in range(input_dim - len(map_json['data']))]
        label_id = g_map_label[map_json['label']]

        point_data = torch.tensor(point_data, dtype=torch.float32)
        point_data = point_data[:input_dim, :embedding_dim]
        # point_data = point_data.T

        return (point_data, label_id)

    def __len__(self):
        return self.num_entries


class BiLSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim):
        super(BiLSTM, self).__init__()
        self.num_layer = 6
        self.hidden_size = 64
        self.bi_num = 2

        self.lstm = nn.LSTM(embedding_dim, self.hidden_size, self.num_layer, bidirectional=True)
        self.fc1 = nn.Linear(self.hidden_size * self.bi_num, output_dim)

    def forward(self,x):

        x = x.permute(1, 0, 2)  # 进行轴交换
        h_0, c_0 = self.init_hidden_state(x.size(1))
        _, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        # 只要最后一个lstm单元处理的结果，取前向LSTM和后向LSTM的结果进行简单拼接
        out = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=-1)
        out = self.fc1(out)
        return F.log_softmax(out, dim=-1)

    def init_hidden_state(self, batch_size):
        h_0 = torch.rand(self.num_layer * self.bi_num, batch_size, self.hidden_size).to(device)
        c_0 = torch.rand(self.num_layer * self.bi_num, batch_size, self.hidden_size).to(device)
        return h_0, c_0


# 数据加载
g_map_label = load_label('./data/labels/label-gb2312-level1.txt')
g_train_loader = DataLoader(LazyTextDataset('./data/casia-sample/Pot1.0Train.v1.json'), batch_size=batch_size, shuffle=True)
g_test_loader = DataLoader(LazyTextDataset('./data/casia-sample/Pot1.0Test.v1.json'), batch_size=batch_size, shuffle=True)

# 模型定义
g_model = BiLSTM(input_dim, embedding_dim, len(g_map_label))
# if torch.cuda.device_count() > 1:
#     print("Use", torch.cuda.device_count(), 'gpus')
#     g_model = nn.DataParallel(g_model)

g_model.to(device)
g_optimizer = optim.Adam(g_model.parameters(), lr=lr)  # Adam 优化器
g_loss_fn = nn.CrossEntropyLoss()  # 多分类损失函数
g_best_train_acc = 0  # Train最佳准确率
g_best_test_acc = 0   # Test最佳准确率

def train(epoch, model_filename, log_filename):
    global g_best_train_acc
    g_model.train()  # 开启训练模式
    
    loss_count = 0        # 损失

    sample_correct = 0    # 正确样本数
    sample_count = 0      # 样本总数
    epoch_acc = 0.0       # 准确率 = 正确样本数 / 样本总数

    train_bar = tqdm(g_train_loader, ncols=80)  # 进度条
    for data in train_bar:
        hans, labels = data
        labels = labels.to(device)
        hans = hans.to(device)

        g_optimizer.zero_grad()
        labels_pred = g_model(hans)
        loss = g_loss_fn(labels_pred, labels)
        loss.backward()
        loss_count += loss.item()
        g_optimizer.step()

        sample_correct += (labels_pred.argmax(axis=1) == labels).sum()
        sample_count += len(hans)

    epoch_acc = sample_correct / sample_count

    if epoch_acc > g_best_train_acc:
        g_best_train_acc = epoch_acc

    print("[EPOCH-{}] 训练准确率={:.2%} LOSS={} 最高准确率={:.2%}".format(epoch + 1, epoch_acc.item(), loss_count, g_best_train_acc.item()))

    with open(log_filename, 'a', encoding='utf-8') as f:
        f.write('{},{},{},'.format(epoch + 1, epoch_acc.item(), loss_count))

    torch.save(g_model.state_dict(), model_filename)
    
def test(epoch, log_filename):
    global g_best_test_acc

    g_model.eval()

    loss_count = 0        # 损失

    sample_correct = 0    # 正确样本数
    sample_count = 0      # 样本总数
    epoch_acc = 0.0       # 准确率 = 正确样本数 / 样本数

    test_bar = tqdm(g_test_loader, ncols=80)
    for data in test_bar:
        hans, labels = data
        labels = labels.to(device)
        hans = hans.to(device)
        labels_pred = g_model(hans)
        _, pred = torch.max(labels_pred, 1)

        loss = g_loss_fn(labels_pred, labels)
        loss_count += loss.item()
        sample_correct += (pred == labels).sum()
        sample_count += len(hans)

    epoch_acc = sample_correct / sample_count
    if epoch_acc > g_best_test_acc:
        g_best_test_acc = epoch_acc

    print('[TEST] 测试准确率={:.2%} LOSS={} 最高准确率={:.2%}'.format(epoch_acc, loss_count, g_best_test_acc.item()))

    with open(log_filename, 'a', encoding='utf-8') as f:
        f.write('{},{}\n'.format(epoch_acc, loss_count))

def main(data_path, model_prefix):

    model_path = os.path.join(data_path, model_prefix)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    log_filename = os.path.join(model_path, '{}_train.csv'.format(model_prefix))
    if not os.path.isfile(log_filename):
        with open(log_filename, 'w', encoding='utf-8') as f:
            f.write('{},{},{},{},{}\n'.format('Epoch', 'Train_Acc', 'Train_Lose', 'Test_Acc', 'Test_Lose'))

    epoch_begin = epochs
    for i in range(epochs):
        filename = os.path.join(model_path, '{}-{}.pkl'.format(model_prefix, i))
        if not os.path.isfile(filename):
            epoch_begin = i
            break

    if epoch_begin < epochs:
        filename = os.path.join(model_path, '{}-{}.pkl'.format(model_prefix, epoch_begin - 1))
        if os.path.isfile(filename):
            print('load', filename)
            g_model.load_state_dict(torch.load(filename), strict=False)
    else:
        epoch_begin = 0

    for i in range(epoch_begin, epochs):
        filename = os.path.join(model_path, '{}-{}.pkl'.format(model_prefix, i))
        train(i, filename, log_filename)
        test(i, log_filename)

if __name__ == "__main__":
    
    main('model', os.path.basename(sys.argv[0]).split('.')[0])
