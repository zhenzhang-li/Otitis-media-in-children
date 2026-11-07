# 儿童耳镜骨干比较与LayerFusion研究代码

本项目整理了十个骨干网络的冻结特征训练、不同层累积读出与融合、小样本适配、校准与固定模型外部评价、鲁棒性、区域保留/替换及层级解释。核心代码不固定四个诊断名称、图像数量、文件名或目录结构。

数据至少需要“图像资源＋参考类别”。可直接映射CSV/TSV、Excel、JSON/JSONL列名，或扫描任意目录；数据库、压缩包、DICOM/HDF5等可通过自定义元数据/图像读取函数接入。项目处理的是RGB二维图像分类，并不声称无需解码即可自动理解所有医学数据格式。

## 安装与先行检查

Python 3.10及以上，先按设备安装匹配的PyTorch/torchvision，再在项目根目录运行：

```bash
python -m pip install -e ".[dev]"
otoscopy-ai --help
python examples/generate_synthetic.py
otoscopy-ai --config configs/smoke.yaml run
otoscopy-ai --config configs/smoke.yaml robustness --all-seeds
otoscopy-ai --config configs/smoke.yaml explain
otoscopy-ai --config configs/smoke.yaml summarize
otoscopy-ai --config configs/smoke.yaml plot
python -m pytest -q
```

示例全部为程序生成的彩色图案和未预训练的小网络，用于检查流程，不是医学图像或论文结果。完整配置为`configs/study.yaml`；快速示例使用较少轮数、较低出图分辨率，不能与论文配置混用。

## 使用自己的数据

复制`configs/study.yaml`为`configs/my_data.yaml`，修改：

| 配置 | 用途 |
|---|---|
| `data.sources` | 任意来源表格、目录扫描、自定义读取函数；每来源独立映射列名 |
| `data.label_map`、可选`classes` | 自己的类别/别名及类别顺序；不固定AOM/OME |
| `split.mode` | 图像分层、按患者/内容组分组、或采用已有划分 |
| `split.fractions`、`seeds` | 四用途比例与任意数量外层种子 |
| `models` | 模型提供器、自己的预训练权重路径及DINOv3代码位置 |
| `training`、`fusion` | 学习率、批大小、早停、层位、融合头及辅助损失 |
| `fewshot` | 每类训练预算、内层种子和比较模型 |
| `robustness`、`explain` | 扰动参数、解释病例、区域数和计算预算 |
| `plot` | 默认190 mm、Arial、600 dpi、PDF/PNG；配色/字号/线宽等集中调整 |

支持的元数据、任意目录、自定义图像解码示例见[数据适配说明](docs/DATA.md)。路径相对于配置文件目录解析，也支持绝对路径和环境变量；无需改变自己的数据文件夹来匹配作者目录。重复影像/患者跨用途默认报错；若明确要保留原始目录的重叠敏感性分析，可设`overlap_policy: report`，并查看审计结果。

```bash
otoscopy-ai --config configs/my_data.yaml prepare
otoscopy-ai --config configs/my_data.yaml inspect
otoscopy-ai --config configs/my_data.yaml split
otoscopy-ai --config configs/my_data.yaml cache
otoscopy-ai --config configs/my_data.yaml train --section backbones
otoscopy-ai --config configs/my_data.yaml train --section layers
otoscopy-ai --config configs/my_data.yaml train --section fewshot
otoscopy-ai --config configs/my_data.yaml evaluate --all-internal
otoscopy-ai --config configs/my_data.yaml evaluate --selected
otoscopy-ai --config configs/my_data.yaml robustness --all-seeds
otoscopy-ai --config configs/my_data.yaml explain
otoscopy-ai --config configs/my_data.yaml summarize
otoscopy-ai --config configs/my_data.yaml plot
```

也可用`run`执行至内部/所选模型外部评价、汇总和绘图；鲁棒性与解释需另行执行。程序将所有拟合完成后再开展内部评价；外部部分只使用内部轮次选择集宏平均F1最高的一套融合权重及其校准参数。各候选选择集成员不同，最高观测选择分数不等于已证明独立共同验证集上的最优。

## 训练与调试

论文设置冻结编码器，仅训练新分类头/融合头；小样本“适配/微调”同样针对头部，当前发布版不支持端到端骨干微调。线性特征标准化只使用当前训练样本；小样本只用当次每类K张样本，不使用选择、校准或评价数据估计均值/标准差。

每个头输出`history.csv`（逐轮损失、选择集F1/准确率、学习率）、最优轮次、选择预测和`checkpoint.pt`。模型记录类别顺序、训练/选择ID、缓存及代码/参数指纹。不兼容的缓存或权重不会被默认为有效；改变数据或配置后用新的输出目录，修改原始图片后先重新`prepare`。内存不足可设`training.feature_storage: stream`。

```bash
otoscopy-ai --config configs/my_data.yaml --set output=../outputs/new_lr --set training.lr=0.0003 train --section backbones --seed 42
python scripts/tune.py --config configs/my_data.yaml --space configs/search.yaml --encoder dinov3_vitb16 --kind tokens
```

调参脚本按选择集分数排序，不使用校准/内部评价/外部标签排序。它是读者可选扩展，不应写成原论文曾执行的参数搜索过程。

## 与论文对应

默认十骨干、74层配置、五模型×六小样本预算×五内层种子，每个外层234次拟合、五外层共1,170次。原始图像数量与类别名称不参与代码断言。3.2、3.3、3.5保留五次内部均值和95% t CI；小样本先在每个外层内平均内层运行，再计算外层CI。最新稿3.4采用一个选择分数最高的融合模型；其Wilson/图像bootstrap区间不等于五次t区间。论文中这一单模型报告属于既有重复评价后的修订选择，代码不能让既有外部数据恢复为首次盲评。

鲁棒性在源RGB图像上逐个施加扰动，再统一预处理；原图重推断须与缓存预测一致。解释区域是特征聚类而非临床病灶。灰色/模糊参考之间的贡献范围不是统计CI。解释未满足删除条件时保留回退案例，不只报告成功示例。

## 发布

所有代码、配置、示例、测试及GitHub Actions已在本目录。当前尚未上传GitHub；真实仓库网址建立后，按[发布说明](docs/PUBLISHING.md)填写引用信息，并将[代码可用性模板](docs/CODE_AVAILABILITY.md)的占位链接替换为实际网址。项目不包含内部数据、病人表格或真实训练权重，第三方DINOv3代码/权重需单独获取并遵守原许可。
