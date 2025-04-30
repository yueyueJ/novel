import os
import re
import time
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import Transformer
from tqdm import tqdm

# ======================
# 配置参数
# ======================
class Config:
    # 数据参数
    corpus_path = "./corpus"        # 语料库路径
    seq_length = 256                 # 输入序列长度
    batch_size = 128                 # 批次大小
    num_workers = 8                  # 数据加载线程数
    
    # 模型参数
    d_model = 384                    # 模型维度
    nhead = 6                        # 注意力头数
    num_layers = 4                   # Transformer层数
    dim_feedforward = 1536           # 前馈层维度
    dropout = 0.1                    # 丢弃率
    
    # 训练参数
    epochs = 30                      # 训练轮次
    lr = 3e-4                        # 学习率
    grad_accum_steps = 2             # 梯度累积步数
    max_grad_norm = 1.0              # 梯度裁剪阈值
    
    # 生成参数
    gen_length = 500                 # 生成文本长度
    temperature = 0.9                # 温度参数
    top_k = 10                       # Top-k采样
    
    # 系统参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = "wuxia_transformer.pth"  # 模型保存路径

# ======================
# 数据预处理
# ======================
class WuxiaDataset(Dataset):
    def __init__(self, text, seq_length):
        self.seq_length = seq_length
        self.text = text
        self.chars = self._build_vocab()
        self.data = self._process_text()
        
    def _build_vocab(self):
        chars = sorted(list(set(self.text)))
        self.char2idx = {'<pad>':0, '<bos>':1, '<eos>':2, **{c:i+3 for i,c in enumerate(chars)}}
        self.idx2char = {v:k for k,v in self.char2idx.items()}
        return chars
    
    def _process_text(self):
        encoded = [self.char2idx['<bos>']] 
        encoded += [self.char2idx[c] for c in self.text]
        encoded.append(self.char2idx['<eos>'])
        
        # 创建训练样本
        samples = []
        for i in range(0, len(encoded)-self.seq_length, self.seq_length//2):
            samples.append(encoded[i:i+self.seq_length])
        return samples
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)

def load_text():
    """加载并预处理文本"""
    text = ""
    for file in os.listdir(Config.corpus_path):
        if not file.endswith('.txt'):
            continue
        try:
            with open(os.path.join(Config.corpus_path, file), 'r', encoding='gb18030') as f:
                text += re.sub(r'\s+', ' ', f.read())
        except UnicodeDecodeError:
            with open(os.path.join(Config.corpus_path, file), 'r', encoding='utf-8') as f:
                text += re.sub(r'\s+', ' ', f.read())
    return text

# ======================
# 模型定义
# ======================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]

class WuxiaTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, Config.d_model)
        self.pos_encoder = PositionalEncoding(Config.d_model)
        self.transformer = Transformer(
            d_model=Config.d_model,
            nhead=Config.nhead,
            num_encoder_layers=0,
            num_decoder_layers=Config.num_layers,
            dim_feedforward=Config.dim_feedforward,
            dropout=Config.dropout,
            batch_first=True
        )
        self.fc = nn.Linear(Config.d_model, vocab_size)
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, src, tgt_mask=None):
        src = self.embed(src) * np.sqrt(Config.d_model)
        src = self.pos_encoder(src)
        
        tgt = src  # Decoder-only
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        return self.fc(output)

# ======================
# 训练流程
# ======================
def train():
    # 准备数据
    text = load_text()
    dataset = WuxiaDataset(text, Config.seq_length)
    dataloader = DataLoader(dataset, 
                          batch_size=Config.batch_size,
                          shuffle=True,
                          num_workers=Config.num_workers,
                          pin_memory=True)
    
    # 初始化模型
    model = WuxiaTransformer(len(dataset.char2idx)).to(Config.device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scaler = torch.cuda.amp.GradScaler()
    
    # 训练循环
    best_loss = float('inf')
    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{Config.epochs}')
        
        for step, batch in enumerate(progress_bar):
            inputs = batch[:, :-1].to(Config.device)
            targets = batch[:, 1:].to(Config.device)
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), 
                               targets.view(-1)) / Config.grad_accum_steps
                
            scaler.scale(loss).backward()
            
            if (step+1) % Config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            total_loss += loss.item() * Config.grad_accum_steps
            progress_bar.set_postfix({'loss': f"{loss.item()*Config.grad_accum_steps:.3f}"})
            
        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), Config.save_path)
            
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.3f}")

# ======================
# 文本生成
# ======================
def generate(seed, model_path=Config.save_path):
    # 加载模型
    text = load_text()[:1000]  # 用于重建词汇表
    dataset = WuxiaDataset(text, Config.seq_length)
    model = WuxiaTransformer(len(dataset.char2idx)).to(Config.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 生成准备
    generated = [dataset.char2idx['<bos>']]
    input_seq = torch.tensor(generated, dtype=torch.long, device=Config.device).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(Config.gen_length):
            outputs = model(input_seq)
            logits = outputs[0, -1, :] / Config.temperature
            
            # Top-k筛选
            values, indices = torch.topk(logits, Config.top_k)
            probs = torch.softmax(values, dim=-1)
            next_idx = indices[torch.multinomial(probs, 1)].item()
            
            generated.append(next_idx)
            input_seq = torch.cat([input_seq[:, 1:], 
                                torch.tensor([[next_idx]], device=Config.device)], dim=1)
            
            if next_idx == dataset.char2idx['<eos>']:
                break
                
    return ''.join([dataset.idx2char[idx] for idx in generated if idx not in [0,1,2]])

if __name__ == "__main__":
    # 训练模型
    train()
    
    # 示例生成
    print("\n生成示例：")
    print(generate("只见他长剑一抖"))