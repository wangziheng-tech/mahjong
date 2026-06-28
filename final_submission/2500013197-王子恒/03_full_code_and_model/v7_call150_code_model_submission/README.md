# V7+call150 国标麻将智能体代码与模型说明

## 1. 最终版本

最终提交版本为 `V7+call150`。策略网络使用 `model_v6.CNNModel`，权重为 `models/mahjong_v7_calibrated_best.pkl`；课程要求的 `.pt` 模型副本为 `models/mahjong_v7_calibrated_best.pt`。两者内容相同，均为 PyTorch state_dict，区别只在扩展名。Botzone 最终入口默认读取 `data/mahjong_v7_calibrated_best.pkl`。

推理阶段的 call150 策略位于 `botzone_final/__main__.py`：

```text
CALL_BIAS = 1.50
CHI_EXTRA_BIAS = 0.35
PENG_EXTRA_BIAS = 0.35
```

当存在 Pass 且存在吃/碰/明杠响应机会时，程序对 Chi、Peng、Gang logits 加偏置，然后仍通过 action_mask 保证动作合法。

## 2. 目录结构

```text
botzone/
  mahjong_v7_call150_friendstyle_storage_bot.zip  # Botzone 代码包，不含模型
botzone_final/
  __main__.py agent.py feature_v2.py model_v6.py  # 可直接打包上传的最终代码
models/
  mahjong_v7_calibrated_best.pkl                  # Botzone 实际使用
  mahjong_v7_calibrated_best.pt                   # 课程提交模型副本
training_code/
  source_v7/                                      # V7 训练、评估、自对战代码
  training_support/                               # 预处理和训练依赖补充
configs/
  run_config.json best_metrics.json               # 最终训练配置和验证指标
eval_results/
  *.json                                          # 关键本地对战结果
requirements.txt
README.md
```

## 3. 安装环境

Botzone 部署使用平台 Python 3.6.5，勾选“长时运行”和“简单交互”。本地训练/评估建议使用 Ubuntu 18+ 与 Python 3.10+。

推荐本地环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` 中列出最小依赖：

```text
numpy
torch>=1.4
PyMahjongGB
```

如果只在 Botzone 上运行最终 bot，不需要安装训练脚本的额外环境。

## 4. Botzone 提交方法

1. 在 Botzone 用户存储空间 `/data` 上传：

```text
models/mahjong_v7_calibrated_best.pkl
```

上传后文件路径应为：

```text
/data/mahjong_v7_calibrated_best.pkl
```

2. 在 Botzone bot 代码处上传：

```text
botzone/mahjong_v7_call150_friendstyle_storage_bot.zip
```

3. 选择 Python 3.6.5，勾选长时运行、简单交互。

4. 若需要重新打包代码，可在 `botzone_final/` 目录下执行：

```bash
zip -j mahjong_v7_call150_friendstyle_storage_bot.zip __main__.py agent.py feature_v2.py model_v6.py
```

## 5. 本地运行最终入口

在本地有 torch/numpy 的环境中，可以这样启动交互入口：

```bash
cd botzone_final
export MAHJONG_MODEL_PATH=../models/mahjong_v7_calibrated_best.pkl
python __main__.py
```

Windows PowerShell：

```powershell
cd botzone_final
$env:MAHJONG_MODEL_PATH = "..\models\mahjong_v7_calibrated_best.pkl"
python __main__.py
```

程序会从标准输入读取 Botzone 协议请求，并向标准输出写动作。完整对局需要由 Botzone 或本地模拟器驱动。

## 6. 训练脚本执行方法

原始训练数据不包含在本 zip 中。课程提供的 `data.txt` 应放到项目根目录的 `data/data.txt`，其 README 说明包含 98209 场对局。

基础监督学习预处理：

```bash
cd training_code/training_support
mkdir -p data
cp /path/to/data.txt data/data.txt
python preprocess_v2.py
```

这会生成 `data_v2/*.npz` 与 `data_v2/count.json`。V7 最终训练使用的配置保存在：

```text
configs/run_config.json
```

核心训练命令形态如下：

```bash
python training_code/source_v7/train_distill_v6.py \
  --train-end 0.95 \
  --val-begin 0.95 \
  --epochs 4 \
  --batch-size 1536 \
  --lr 5e-5 \
  --model-module model_v6 \
  --init-ckpt model/checkpoint/v6_refine2_095/mahjong_v6_best.pkl \
  --teacher-module model_v6 \
  --teacher-ckpt model/checkpoint/v6_refine2_095/mahjong_v6_best.pkl \
  --reward-dir data_reward_v5 \
  --distill-coef 0.35 \
  --type-coef 0.08 \
  --value-coef 0.04 \
  --save-prefix mahjong_v7_calibrated
```

说明：最终 V7 是从 V6 refine2 checkpoint 继续微调，并使用 `data_reward_v5` 的收益样本训练价值头。本提交包保留训练脚本和最终配置；大体积中间数据 `data_v2/`、`data_reward_v5/` 和 V6 中间 checkpoint 未放入压缩包。

## 7. 评估脚本

本地自对战脚本在 `training_code/source_v7/` 中：

```bash
python selfplay_arena_v2.py --help
python selfplay_fourway_v2.py --help
python eval_v7_diagnostics.py --help
```

关键评估结果保存在 `eval_results/`：

- `v7_call150_vs_base_2v2_r40_p24_npu.json`
- `v8a_vs_v7base_2v2_r20_p24_npu.json`
- `v8a_vs_v7call150_2v2_r20_p24_npu.json`

主要结论：960 局 2v2 中，V7+call150 相对 V7 base 团队小分 +480，排名分 208 vs 192，点炮率 16.46% vs 17.08%。V8a 虽离线准确率更高，但自对战变弱，因此未采用。

## 8. 模型校验

`models/mahjong_v7_calibrated_best.pkl` 与 `models/mahjong_v7_calibrated_best.pt` 的 SHA256 相同，说明它们是同一个 state_dict 的两份命名：

```text
E2A62BD155EA298C14E7ED758CFC5824629DAD6E6A5C07E2756B66226B5098E6
```
