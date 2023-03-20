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
        point_data = map_json['data']
        point_len = len(map_json['data'])

        label_id = g_map_label[map_json['label']] 
        point_data = torch.tensor(point_data, dtype=torch.float32)

        return label_id, point_data, point_len

    def __len__(self):
        return self.num_entries

def collate_fn(batch_data):
    batch_data.sort(key=lambda data: data[2], reverse=True)
    labels = [x[0] for x in batch_data]
    hans = [x[1] for x in batch_data]
    hans_len = [x[2] for x in batch_data]

    labels = torch.tensor(labels, dtype=torch.long)
    hans = nn.utils.rnn.pad_sequence(hans, batch_first=True, padding_value=0)
    hans_len = torch.tensor(hans_len, dtype=torch.long)
    
    return labels, hans, hans_len

class GRU(nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim):
        super(GRU, self).__init__()
        self.num_layer = 4
        self.hidden_size = 48
        self.bi_num = 2

        self.rnn = nn.GRU(embedding_dim, self.hidden_size, self.num_layer, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(self.hidden_size * self.bi_num, output_dim)

    def forward(self, x, x_len):

        x_packed = nn.utils.rnn.pack_padded_sequence(input=x, lengths=x_len, batch_first=True)
        output, h_n = self.rnn(x_packed)
        pade_outputs, others = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        out = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=-1)
        out = self.fc1(out)
        return F.log_softmax(out, dim=-1)


# 数据加载
g_map_label = load_label('./data/labels/label-gb2312-level1.txt')
g_train_loader = DataLoader(LazyTextDataset('./data/casia-sample/Pot1.0Train.v1.json'), batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
g_test_loader = DataLoader(LazyTextDataset('./data/casia-sample/Pot1.0Test.v1.json'), batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

# 模型定义
g_model = GRU(input_dim, embedding_dim, len(g_map_label))

# 并行计算
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
        labels, hans, hans_len = data
        
        labels = labels.to(device)
        hans = hans.to(device)
        # hans_len = hans_len.to(device)

        g_optimizer.zero_grad()

        labels_pred = g_model(hans, hans_len)
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
        labels, hans, hans_len = data
        labels = labels.to(device)
        hans = hans.to(device)

        labels_pred = g_model(hans, hans_len)
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

import torch.onnx 

#Function to Convert to ONNX 
def Convert_ONNX(model): 

    # set the model to inference mode 
    model.eval() 
    train_loader = DataLoader(LazyTextDataset('./data/casia-sample/Pot1.0Train.v1.json'), batch_size=1, collate_fn=collate_fn, shuffle=True)
    for i, item in enumerate(train_loader):
        print('i:', i)
        labels, hans, hans_len = item

        labels_pred = model(hans, hans_len)

        # Export the model   
        torch.onnx.export(model,         # model being run 
            (hans, hans_len),       # model input (or a tuple for multiple inputs) 
            "casia.onnx",       # where to save the model  
            export_params=True,  # store the trained parameter weights inside the model file 
            opset_version=10,    # the ONNX version to export the model to 
            do_constant_folding=True,  # whether to execute constant folding for optimization 
            input_names = ['hans', 'hans_len'],   # the model's input names 
            output_names = ['labels'], # the model's output names 
            dynamic_axes={'hans' : {1: 'len'}}) 
        print(" ") 
        print('Model has been converted to ONNX')
        break

def export_model():
    model = GRU(input_dim, embedding_dim, len(g_map_label))
    path = "casia_result.pkl" 
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu'))) 
    Convert_ONNX(model)

if __name__ == "__main__":
    export_model()
    # main('model', os.path.basename(sys.argv[0]).split('.')[0])
