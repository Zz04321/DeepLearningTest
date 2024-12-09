深度学习框架差分测试工具 (基于 PyTorch)
项目简介
本项目旨在开发一种基于等价融合算子的深度学习框架差分测试工具，帮助识别 PyTorch 框架中的潜在缺陷。通过构造等价的深度学习模型，并比较其在不同场景下的行为一致性，检测框架算子的异常表现。

项目目标
等价模型生成：基于 PyTorch 实现多组常用融合算子（如 Conv-BN 融合、Add-Mul 融合）。
差分测试：利用生成的等价模型，对框架的行为进行比较分析，捕获潜在的不一致问题。
结果分析与可视化：生成测试报告，分析框架中的异常表现。
项目结构
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
│   ├── logger.py          # 日志工具
│   ├── visualizer.py      # 可视化工具
├── tests/                 # 测试模块
│   ├── test_operators.py  # 融合算子单元测试
│   ├── test_tester.py     # 差分测试单元测试
└── README.md              # 项目文档
功能概述
1. 融合算子模块
定义基础算子和常用融合算子（如 Conv-BN 融合）。
提供算子接口以便生成等价深度学习模型。
2. 等价模型生成
自动生成包含或不包含融合算子的等价深度学习模型。
确保生成的模型逻辑上等价，但实现方式不同。
3. 差分测试工具
比较多个等价模型在不同输入和环境下的输出。
捕获潜在的不一致性，帮助定位框架缺陷。
4. 数据加载工具
提供随机生成的测试数据。
支持扩展到标准化数据集（如 ImageNet、CIFAR）。
5. 结果分析与可视化
记录测试过程中的差异，生成清晰的测试报告。
可视化模型行为差异，便于分析。

待完成任务
至少实现5组常用融合算子：
Conv-BN 融合
Add-Mul 融合
MatMul-Add 融合
Relu-Dropout 融合
Pooling-FC 融合
完善等价模型生成逻辑：
支持更多复杂的模型结构。
验证融合后的模型等价性。
差分测试逻辑优化：
