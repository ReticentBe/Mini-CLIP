import torch
from torch.utils.data import DataLoader, TensorDataset
from model import CLIP
from loss import CLIPLoss
from engine import train_one_epoch, evaluate

def test_engine():
    print("--- 开始测试 Engine ---")
    # 如果有 GPU 用 GPU，没有就用 CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 1. 制造假数据 (Mock Data)
    # 我们造 8 个样本，Batch Size 设为 4，刚好能跑 2 个 Batch
    vocab_size = 100
    max_len = 16
    batch_size = 4
    num_samples = 16
    
    images = torch.randn(num_samples, 3, 224, 224)
    input_ids = torch.randint(0, vocab_size, (num_samples, max_len))
    attention_mask = torch.ones(num_samples, max_len)
    
    # 使用 PyTorch 原生的 TensorDataset 快速打包
    dataset = TensorDataset(images, input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    # 2. 初始化模型、损失函数、优化器
    model = CLIP(vocab_size=vocab_size, max_len=max_len).to(device)
    criterion = CLIPLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 3. 测试训练环节
    print("\n--- 测试 train_one_epoch ---")
    train_loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
    print(f"返回的 Train Loss: {train_loss:.4f}")
    
    # 4. 测试评估环节 (看看 Recall 能不能算出来)
    print("\n--- 测试 evaluate ---")
    val_loss = evaluate(model, dataloader, criterion, device)
    print(f"返回的 Val Loss: {val_loss:.4f}")
    
    print("\n--- Engine 测试全部通过！ ---")

if __name__ == "__main__":
    # 如果你之前写了 test_model()，可以先把它注释掉，只跑 test_engine()
    test_engine()
