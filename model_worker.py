import os
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file
from torchinfo import summary  # 可视化模型结构（比原生print更清晰）

class torch_worker:
    def __init__(self):
        pass

    def read_state_dict(self, model, path, input_shape):
        model.load_state_dict(torch.load(path))
        summary(model, input_size=input_shape)
        return model

    def write_state_dict(self, model, save_path):
        torch.save(model.state_dict(), save_path)

    def write_model(self, model, save_path):
        torch.save(model, save_path)

class safetensors_worker:
    def __init__(self):
        pass

    def read(self, model, path):
        model.load_state_dict(load_file(path))
        return model

    def write(self, model, save_path):
        save_file(model.state_dict(), save_path)


class torchscript_worker:
    def __init__(self):
        pass

    def read(self, path):
        model = torch.jit.load(path)
        print(model.graph)  # 打印静态计算图
        print(model.code)   # 打印编译后的代码

        return model

    def write(self, model, example_input, save_path):
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save(save_path)

class onnx_worker:
    def __init__(self):
        pass

    def read(self, path):
        # 加载ONNX模型 + 输出结构
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
        print(f"- 输入节点：{[node.name for node in onnx_model.graph.input]}")
        print(f"- 输出节点：{[node.name for node in onnx_model.graph.output]}")
        print(f"- 算子数量：{len(onnx_model.graph.node)}")
        # 打印ONNX模型层信息（简化版）
        print("- 核心算子列表：")
        for i, node in enumerate(onnx_model.graph.node[:5]):  # 只打印前5个算子
            print(f"  [{i}] 类型：{node.op_type} | 输入：{node.input} | 输出：{node.output}")

    def write(self, model, example_input, save_path):
        torch.onnx.export(
            model,
            example_input,
            save_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=12
        )

