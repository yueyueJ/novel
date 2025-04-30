import os
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ====================
# 数据预处理
# ====================
def load_corpus(corpus_dir):
    """加载并清洗文本数据"""
    text = ""
    for filename in os.listdir(corpus_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(corpus_dir, filename), 'r', encoding='gb18030') as f:  # 处理中文编码
                raw_text = f.read()
                # 清洗文本：保留中文和基本标点
                cleaned = re.sub(r'[^\u4e00-\u9fa5，。！？、：；“”‘’—（）《》…]', '', raw_text)
                cleaned = re.sub(r'\s+', ' ', cleaned)
                text += cleaned
    return text

# 配置参数
corpus_dir = "D:\corpus"      # 语料库目录
seq_length = 100           # 输入序列长度
batch_size = 128           # 批次大小
embed_dim = 256            # 嵌入维度
hidden_size = 512          # LSTM隐藏层大小
num_layers = 2             # LSTM层数
epochs = 50                # 训练轮次
temperature = 0.8          # 生成温度
top_k = 10                 # Top-k采样参数

# 加载并处理数据
full_text = load_corpus(corpus_dir)
chars = sorted(list(set(full_text)))
vocab_size = len(chars)

# 创建字符映射
char_to_idx = {c:i for i, c in enumerate(chars)}
idx_to_char = {i:c for i, c in enumerate(chars)}

# 转换为数字序列
data = [char_to_idx[c] for c in full_text]

# ====================
# 数据加载器
# ====================
class TextDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, index):
        return (
            torch.tensor(self.data[index:index+self.seq_length], dtype=torch.long),
            torch.tensor(self.data[index+1:index+self.seq_length+1], dtype=torch.long)
        )

dataset = TextDataset(data, seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# ====================
# 模型定义
# ====================
class KungFuLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers,
                           dropout=0.2, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits, hidden

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = KungFuLSTM().to(device)
print(f"Using device: {device}")

# ====================
# 训练配置
# ====================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

# ====================
# 训练循环
# ====================
for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
    
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)  # 梯度裁剪
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(dataloader)
    scheduler.step(avg_loss)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

# ====================
# 文本生成
# ====================
def generate_text(model, seed, num_chars=500):
    model.eval()
    generated = list(seed)
    input_seq = torch.tensor([char_to_idx[c] for c in seed[-seq_length:]], 
                            dtype=torch.long, device=device).unsqueeze(0)
    hidden = None
    
    with torch.no_grad():
        for _ in range(num_chars):
            # 填充不足长度的序列
            if input_seq.size(1) < seq_length:
                padding = torch.zeros(1, seq_length - input_seq.size(1), 
                                    dtype=torch.long, device=device)
                input_seq = torch.cat([padding, input_seq], dim=1)
            
            # 前向传播
            outputs, hidden = model(input_seq, hidden)
            logits = outputs[0, -1, :] / temperature
            
            # Top-k筛选
            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, -1]] = -float('Inf')
                
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            
            generated.append(idx_to_char[next_idx])
            input_seq = torch.cat([input_seq[:, 1:], 
                                torch.tensor([[next_idx]], device=device)], dim=1)
    
    return ''.join(generated)

# 示例生成
seed_text = "只见张三丰"
print("\n生成文本：")
print(generate_text(model, seed_text, num_chars=500))