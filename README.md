# 国标麻将智能体 V7+call150

本仓库保存国标麻将大作业最终版本 V7+call150 的代码、模型、评估材料和研究性报告。

## 最终提交目录

最终整理后的上交内容位于：

`final_submission/2500013197-王子恒/`

其中包含三部分：

- `01_botzone_agent_without_model/`：Botzone 可上传的无模型智能体代码包。
- `02_research_report/`：最终研究性报告 PDF。
- `03_full_code_and_model/`：完整代码、训练/评估脚本、配置、评估结果和模型权重。

## Botzone 部署

无模型 Botzone 代码包：

`final_submission/2500013197-王子恒/01_botzone_agent_without_model/mahjong_v7_call150_friendstyle_storage_bot.zip`

模型文件：

`final_submission/2500013197-王子恒/03_full_code_and_model/v7_call150_code_model_submission/models/mahjong_v7_calibrated_best.pkl`

Botzone 用户存储空间中应上传为：

`/data/mahjong_v7_calibrated_best.pkl`

代码运行时默认读取：

`data/mahjong_v7_calibrated_best.pkl`

## 版本说明

V7+call150 继承 V7 calibrated 的残差 CNN 策略网络，并在推理阶段对 Chi/Peng/Gang 响应动作加入轻量 logits 偏置，以缓解原 V7 略保守的问题。最终报告中记录了 Baseline、V2、V3/V4、V6 refine2、V7 calibrated、V7+call150 和 V8a 的迭代关系、实验结果与失败实验分析。

根目录下保留的 `source/`、`botzone_storage/` 和 `metadata/` 是较早整理的 V7 代码与元数据；最终提交请以 `final_submission/2500013197-王子恒/` 为准。
