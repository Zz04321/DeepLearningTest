# 基于等价融合算子的差分测试 (基于 PyTorch)
## 详细设计请见设计文档
## 项目简介
本项目旨在开发一种基于等价融合算子的深度学习框架差分测试工具，帮助识别 PyTorch 框架中的潜在缺陷。通过构造等价的深度学习模型，并比较其在不同场景下的行为一致性，检测框架算子的异常表现。

## 项目目标
等价模型生成：基于 PyTorch 实现多组常用融合算子（如 Conv-BN 融合、Add-Mul 融合）。
差分测试：利用生成的等价模型，对框架的行为进行比较分析，捕获潜在的不一致问题。
结果分析与可视化：生成测试报告，分析框架中的异常表现。

## 项目结构
project/  
├── main.py                # 主程序入口  
├── operators/             # 融合算子模块  
│   ├── base_operator.py   # 基础算子定义  
│   ├── fusion_operator.py # 融合算子定义  
├── model_builder.py       # 等价模型生成工具  
├── tester.py              # 差分测试工具        
├── datasets/              # 数据集模块  
│   ├── data_loader.py     # 数据加载工具  
├── utils/                 # 工具模块  
│   ├── visualizer.py      # 可视化工具 
├── log/                   # 日志模块  
├── plot/                  # 图表模块  
└── README.md              # 项目文档

## 功能概述
### 1. 融合算子模块
融合算子模块包含了一系列的融合算子，这些算子是深度学习模型中常见的操作组合，通过将它们融合在一起，可以减少模型的计算量和内存占用，同时提高模型的运行效率。本项目实现了以下融合算子：

·ConvBnFusion：卷积和批归一化融合算子。将卷积层和批归一化层的操作融合为一个单一的操作，减少了模型的计算量。
·AddMulFusionOperator：加法和乘法融合算子。将加法层和乘法层的操作融合为一个单一的操作，提高了模型的运行效率。
·MatMulAddFusionOperator：矩阵乘法和加法融合算子。将矩阵乘法层和加法层的操作融合为一个单一的操作，减少了模型的计算量。
·ReluDropoutFusionOperator：ReLU和Dropout融合算子。将ReLU激活层和Dropout层的操作融合为一个单一的操作，提高了模型的运行效率。
·PoolingFcFusionOperator：池化和全连接层融合算子。将池化层和全连接层的操作融合为一个单一的操作，减少了模型的计算量。
这些融合算子通过继承 BaseOperator 类，并实现 apply 方法来定义具体的融合操作。apply 方法接受两个参数：需要融合的两个层，并返回融合后的权重和偏置。

### 2. 等价模型生成
等价模型生成模块负责生成包含或不包含融合算子的等价深度学习模型。通过调用 ModelBuilder 类中的方法，可以生成不同结构的模型。这些模型在逻辑上是等价的，但实现方式不同，一些模型使用了融合算子，而另一些则没有。
例如，ModelBuilder 类中的 build_model_one 方法生成了一个包含卷积层、批归一化层、ReLU激活层、池化层、展平层、全连接层和Dropout层的模型。通过设置 use_fusion 参数，可以控制是否使用融合算子。
### 3. 差分测试工具
比较多个等价模型在不同输入和环境下的输出。
捕获潜在的不一致性，帮助定位框架缺陷。
### 4. 数据加载工具
提供随机生成的测试数据。
### 5. 结果分析与可视化
记录测试过程中的差异，生成清晰的测试报告。
可视化模型行为差异，便于分析。
Visualizer 类负责可视化测试结果。通过调用 visualize_difference 和 plot_differences_box 方法，可视化参数差异和箱线图，model1示例如下：
![difference_between_models_0](https://github.com/user-attachments/assets/ccdf823e-e4ae-467c-b92d-466a2a9ddfc2) 
![box_plot_of_differences_0](https://github.com/user-attachments/assets/dad4e2fa-000d-41f3-955a-1baa87803c86) 
通过differences值以及可视化可见，融合算子与非融合算子在示例模型下输出结果差异在可接受范围内。


