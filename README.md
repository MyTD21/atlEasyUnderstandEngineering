# different model format
## 格式总结
| ID |  格式 | 框架 | 特点&用途 |
| ----- | ----- | ----- | ----- |
| 0 | .pt / .pth | PyTorch |  本质都是 Pickle 序列化格式（pth 是传统后缀，pt 更通用）；可保存 state_dict （仅权重）或整个模型（含结构）；但 Pickle 有安全风险（可能执行恶意代码）。|
| 1 | .ckpt | PyTorch Lightning/TensorFlow（早期）/MM 系列（MMDetection/MMagic）| 「检查点（Checkpoint）」格式，不仅含权重，还包含优化器状态、训练步数、超参数等，用于断点续训；PyTorch Lightning 和 TF 早期均常用。 |
| 2 | .safetensors |全框架（LLM/Stable Diffusion 为主）|替代 Pickle 的安全格式，二进制序列化、无代码执行风险，加载速度更快，跨框架兼容性更好（现在成为 LLM / 文生图模型的主流格式）。|
| 3 | .saved_model |TensorFlow 2.x（推荐）|文件夹形式（含 asset  s/ 、 varia  bles/ 、 saved  _model.pb ），TF 官方推荐格式，适配 TF Serving/TensorRT 部署，支持动态图 / 静态图。|
||
| 4 | .onnx |全框架跨平台（PyTorch/TF/Paddle→部署）|Open Neural Network Exchange 标准，工业界最主流的跨框架格式；支持 ONNX Runtime 推理，可转换为端侧格式（TFLite/RKNN）。|
| 5 | .torchscript |PyTorch 部署（C++/ 移动端）|PyTorch 编译后的静态图格式（后缀仍为 .pt），剥离 Python 依赖，适配 C++ 推理引擎，是 PyTorch 部署的核心格式。|
||
| 6 | .pkl / .pickle | 通用 Python 模型（自定义 ML 流程）| 基础序列化格式，可保存任意 Python 对象（如 Pipeline、自定义模型），但跨版本兼容性差，有安全风险。 |

## 说明
    torchscript和pt/pth虽然都是以pt结尾，但是区别明显：
      1 pt/pth，要么仅保存了权重，要么存储了结构，权重和训练状态信息，只有python环境可以运行；
      2 TorchScript的pt，保存了计算图和权重，无训练状态信息，cpp，python均可加载；

    Pickle 是Python 内置的序列化 / 反序列化模块（对应的文件后缀常为 .pkl ），核心作用是把「内存中的 Python 对象」（比如模型、列表、类实例、numpy
    数组等）转换成二进制字节流（序列化，pickling），就是把在内存里的对象专程二进制文件，存到硬盘，要用的时候再 “解封” 还原，是 Python 专属的对象持
    久化方案；
