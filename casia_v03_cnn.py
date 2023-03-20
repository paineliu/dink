import sys
import os
import torch
import torch.nn as nn
import json
import linecache
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# 模型输入参数
epochs = 500       # 迭代次数
batch_size = 2000  # 每个批次样本数
input_dim = 320    # 每个汉字坐标总数，如果不够需要使用0进行填充
embedding_dim = 4  # 每个坐标向量大小
output_dim = 3755  # 输出维度，为识别汉字类别数量
lr = 0.001  # 学习率

# Decide which device we want to run on
ngpu= 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")



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
        point_data = point_data.T

        return (point_data, label_id)

    def __len__(self):
        return self.num_entries

# 一维卷积模块
class CNN(nn.Module):
    def __init__(self, output_dim, embedding_dim):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv1d(4, 32, 7, stride=1, padding=3), nn.BatchNorm1d(32), nn.LeakyReLU(), nn.AdaptiveMaxPool1d(output_size=160))
        self.conv2 = nn.Sequential(nn.Conv1d(32, 64, 7, stride=1, padding=3), nn.BatchNorm1d(64), nn.LeakyReLU(), nn.AdaptiveMaxPool1d(output_size=80))
        self.conv3 = nn.Sequential(nn.Conv1d(64, 128, 7, stride=1, padding=3), nn.BatchNorm1d(128), nn.LeakyReLU(), nn.AdaptiveMaxPool1d(output_size=40))
        self.conv4 = nn.Sequential(nn.Conv1d(128, 64, 7, stride=1, padding=3), nn.BatchNorm1d(64), nn.LeakyReLU(), nn.AdaptiveMaxPool1d(output_size=20))
        self.conv5 = nn.Sequential(nn.Conv1d(64, 32, 7, stride=1, padding=3), nn.BatchNorm1d(32), nn.LeakyReLU())
        self.conv6 = nn.Sequential(nn.Conv1d(32, 16, 7, stride=1, padding=3), nn.BatchNorm1d(16), nn.LeakyReLU())
        self.fc1 = nn.Sequential(nn.Linear(16 * 20, 64))
        self.fc2 = nn.Linear(64, output_dim)
        
    def forward(self, x):

        x = self.conv1(x) 
        x = self.conv2(x) 
        x = self.conv3(x) 
        x = self.conv4(x) 
        x = self.conv5(x) 
        x = self.conv6(x) 

        x = x.reshape(-1, x.shape[1] * x.shape[2])
        x = self.fc1(x)
        x = self.fc2(x)

        return x


# 将数据加载成迭代器
g_map_label = load_label('./data/labels/label-gb2312-level1.txt')
train_dataset = LazyTextDataset('./data/casia-sample/Pot1.0Train.v1.json')
test_dataset  = LazyTextDataset('./data/casia-sample/Pot1.0Test.v1.json')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = CNN(output_dim=len(g_map_label), embedding_dim=embedding_dim)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)  # 优化器
loss_fn = nn.CrossEntropyLoss()  # 多分类损失函数

best_acc = 0  # 保存最好准确率
best_model = None  # 保存对应最好准确率的模型参数

def train(epoch, model_filename, log_filename):
    global best_acc
    model.train()  # 开启训练模式
    epoch_acc = 0  # 每个epoch的准确率
    running_loss = 0  # 损失
    epoch_acc_count = 0  # 每个epoch训练的样本数
    train_count = 0  # 用于计算总的样本数，方便求准确率

    train_bar = tqdm(train_loader, ncols=80)  # 进度条
    for data in train_bar:
        x_input, label = data  # 解包迭代器中数据
        label = label.to(device)
        x_input = x_input.to(device)

        optimizer.zero_grad()

        # 形成预测结果
        label_pred = model(x_input)

        # 计算损失
        loss = loss_fn(label_pred, label)
        loss.backward()
        
        running_loss += loss.item()

        optimizer.step()

        # 计算每个epoch正确的个数
        epoch_acc_count += (label_pred.argmax(axis=1) == label).sum()
        train_count += len(x_input)

    # 每个epoch对应的准确率
    epoch_acc = epoch_acc_count / train_count

    # 保存模型及相关信息
    if epoch_acc > best_acc:
        best_acc = epoch_acc

    # 打印信息
    print("[EPOCH-{}] 训练准确率={:.2%} LOSS={} 最高准确率={:.2%}".format(epoch + 1, epoch_acc.item(), running_loss, best_acc.item()))

    with open(log_filename, 'a', encoding='utf-8') as f:
        f.write('EPOCH\t{}\t{}\t{}\t'.format(epoch + 1, running_loss, epoch_acc.item()))

    torch.save(model.state_dict(), model_filename)
    
def test(epoch, log_filename):
    model.eval()
    correct = 0
    test_bar = tqdm(test_loader, ncols=80)
    for data in test_bar:
        x_input, label = data
        label = label.to(device)
        x_input = x_input.to(device)
        y_pred=model(x_input)
        _, pred=torch.max(y_pred, 1)
        correct += (pred == label).sum()
    with open(log_filename, 'a', encoding='utf-8') as f:
        f.write('{}\n'.format(correct.item()/len(test_dataset)))

    print('[TEST] 测试准确率={:.2%}'.format(correct.item()/len(test_dataset)))


def main(data_path, model_prefix):

    model_path = os.path.join(data_path, model_prefix)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    log_filename = os.path.join(model_path, '{}_train.log'.format(model_prefix))

    epoch_begin = epochs
    for i in range(epochs):
        filename = os.path.join(model_path, '{}-{}.pkl'.format(model_prefix, i))
        if not os.path.isfile(filename):
            epoch_begin = i
            break

    if epoch_begin < epochs:
        filename = os.path.join(model_path, '{}-{}.pkl'.format(model_prefix, epoch_begin - 1))
        if os.path.isfile(filename):
            model.load_state_dict(torch.load(filename), strict=False)
    else:
        epoch_begin = 0

    for i in range(epoch_begin, epochs):
        filename = os.path.join(model_path, '{}-{}.pkl'.format(model_prefix, i))
        train(i, filename, log_filename)
        test(i, log_filename)

if __name__ == "__main__":
    
    main('model', os.path.basename(sys.argv[0]).split('.')[0])