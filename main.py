import torch.nn as nn
import torch
import os

from model_worker import torch_worker

class Simple3LayerDNN(nn.Module):
    def __init__(self, input_dim=10, hidden1_dim=64, hidden2_dim=32, output_dim=5):
        super().__init__()
        # 3层全连接（DNN核心）：输入层→隐藏层1→隐藏层2→输出层
        self.dnn = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),  # 第一层：输入→隐藏1
            nn.ReLU(),                          # 激活函数
            nn.Linear(hidden1_dim, hidden2_dim),# 第二层：隐藏1→隐藏2
            nn.ReLU(),                          # 激活函数
            nn.Linear(hidden2_dim, output_dim)  # 第三层：隐藏2→输出
        )

    def forward(self, x):
        return self.dnn(x)

def main():
    model = Simple3LayerDNN(
        input_dim=10,    # 输入维度：10个特征
        hidden1_dim=64,  # 隐藏层1维度
        hidden2_dim=32,  # 隐藏层2维度
        output_dim=5     # 输出维度：5分类/回归
    )
    model.eval()  # 禁用训练态行为（Dropout/BatchNorm）

    example_input = torch.randn(1, 10)  # batch_size=1, input_dim=10
    input_shape = (1, 10)  # 用于模型结构可视化

    MODEL_DIR = "./models"
    # 若文件夹不存在则创建，exist_ok=True避免重复创建报错
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"模型存储目录：{os.path.abspath(MODEL_DIR)}（已确保存在）\n")

    torch_worker1 = torch_worker()
    torch_worker1.write_state_dict(model, os.path.join(MODEL_DIR, "model_weights.pt"))
    torch_worker1.write_state_dict(model, os.path.join(MODEL_DIR, "model_weights.pt"))
    torch_worker1.write_model(model, os.path.join(MODEL_DIR, "model_weights_all.pt"))

    model_rd = Simple3LayerDNN()
    ret = torch_worker1.read_state_dict(model_rd, os.path.join(MODEL_DIR, "model_weights.pt"), input_shape)


if __name__ == '__main__':
    main()
