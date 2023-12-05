import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from RNN_pre_process import TextDataset, ProcessData, BucketedDataLoader, save_pretrained
from RNN_Model import RNNModel
from tqdm import tqdm

# 设置超参数
embedding_dim = 50
vocab_size = 1000
hidden_dim = 128
batch_size = 256
num_epochs = 40
learning_rate = 0.001

# 设置文件路径
train_file_path = "Annex_02_PKU_Chinese_Corpus199801\ChineseCorpus199801.txt"
ovfitting_embedding_file_path = "rnn_ovfitting_embedding_layer.pth"
tested_embedding_file_path = "rnn_tested_embedding_layer.pth"

# 调用ProcessData类生成预处理数据和词典
data_processed = ProcessData(train_file_path)
sentences_train, sentences_test = data_processed.split_data()

# 导入训练数据
train_dataset = TextDataset(sentences_train, data_processed.word_to_index)
train_loader = BucketedDataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 导入测试数据
test_dataset = TextDataset(sentences_test, data_processed.word_to_index)
test_loader = BucketedDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 导入模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNNModel(vocab_size, embedding_dim, hidden_dim).to(device)

# 定义损失函数和优化器
ce_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def evaluate(model, test_loader, device):
    model.eval()
    total_loss = 0
    num_test = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs, targets, _ = batch
            inputs, targets = inputs.to(device), targets.to(device)
            log_probs = model(inputs)
            loss = ce_loss(log_probs, targets)
            total_loss += loss.item()
            num_test += 1
    model.train()
    return total_loss / num_test

model.train()
train_losses = []
test_losses = []
best_loss = float('inf')

for epoch in range(num_epochs):
    total_loss = 0
    num_train = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch{epoch}"):
        inputs, targets, _ = batch  # ignore lengths
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        log_probs = model(inputs)  # remove lengths
        loss = ce_loss(log_probs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_train += 1
    train_loss = total_loss / num_train
    train_losses.append(train_loss)

    test_loss = evaluate(model, test_loader, device)
    test_losses.append(test_loss)
    print(f"Train Loss: {train_loss:.2f}")
    print(f"Test Loss: {test_loss:.2f}")
    if test_loss < best_loss:
        best_loss = test_loss
        save_pretrained(data_processed.index_to_word, model.embeddings.weight.data, tested_embedding_file_path)
    print()

save_pretrained(data_processed.index_to_word, model.embeddings.weight.data, ovfitting_embedding_file_path)

# 绘制训练损失和测试损失的变化趋势图
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()