# llm-omop-trans

[English](README.md)

## 项目简介

`llm-omop-trans` 通过兼容 OpenAI Chat Completions 的接口，为已确定身份的 OMOP Concept 生成**仅供审核的简体中文候选名**。程序从七列输入中按 `vocabulary_id` 精确选取词表，校验整份输入，再用同一 Athena 快照的三个文件补全合格记录。候选不会写入正式字段 `concept_name_zh`。

仅在术语来源许可和数据外发范围均已明确授权时处理真实数据。仓库内 `DEMO` 记录为虚构样例，可用于少量连通性测试。真实数据运行前请阅读[来源与外发政策](docs/SOURCE_POLICY.md)。

## 功能特性

- 严格校验 UTF-8、波浪号分隔输入；精确选择词表；外发前核对同版 Concept 身份、有效期及已有中文名。
- 每次请求只包含一个 Concept；使用固定 JSON 结构、严格解析和 14 类医学属性确定性后检；全部候选仍须专家审核。
- 独立 Pilot 运行，每次抽取 1–20 条可发送记录；机器门禁、明确选择已审核 Pilot、正式运行复用已验证的 Pilot 成功结果。
- Pilot 写入 `outputs/test/<run-id>/`，正式运行写入 `outputs/<run-id>/`；`outputs/run-index.json` 可按运行编号检索。
- DuckDB 调用台账、逐次提交、CSV 导出校验和带指纹的断点续跑；已成功请求不重发。仅明确识别额度耗尽且后备模型合成探测成功时切换模型。
- 汇总兼容的正式运行，区分 API 成功、候选生成、后检结果和处理完成。

## 环境要求

- Python 3.12（`>=3.12,<3.13`）与 [uv](https://docs.astral.sh/uv/)。
- 真实请求需要兼容 OpenAI Chat Completions 的端点及本地 API Key。
- 真实术语需要七列输入，以及来自同一获许可 Athena 快照的 `CONCEPT.csv`、`CONCEPT_SYNONYM.csv`、`VOCABULARY.csv`。仓库不分发真实数据。

依赖在 `pyproject.toml` 声明，并由 `uv.lock` 锁定；运行不依赖原课程项目。

## 安装指南

```powershell
git clone https://github.com/17ayyy/llm-omop-trans.git
cd llm-omop-trans
uv sync --locked
Copy-Item .env.example .env
```

在 `.env` 中填写已获授权的端点、模型及密钥。初始路径指向仓库内合成样例。Git 会忽略 `.env`、真实数据、运行状态、日志和缓存。`uv.toml` 指定 PyPI 与项目内缓存；当前终端继承的 `UV_DEFAULT_INDEX` 环境变量可能覆盖该索引。

## 参数配置

| `.env` 配置项 | 作用 |
| --- | --- |
| `OMOP_CONCEPT_PATH`、`OMOP_CONCEPT_SYNONYM_PATH`、`OMOP_VOCABULARY_PATH` | UTF-8、TAB 分隔的三个快照文件；相对路径从运行工作目录解析。 |
| `OMOP_ATHENA_RELEASE`、`OMOP_EFFECTIVE_DATE` | 快照标识及有效日（`YYYY-MM-DD`），用于来源记录和有效性核对。 |
| `LLM_BASE_URL` | 兼容 OpenAI 的纯 HTTP(S) 基础地址，通常以 `/v1` 结尾；不能包含凭据、查询或片段。 |
| `LLM_MODEL`、`LLM_API_KEY` | 默认模型与密钥；`LLM_MODEL_API_KEY` 是旧别名，两者同时设置时优先使用 `LLM_API_KEY`。 |
| `LLM_TIMEOUT_SECONDS` | 单次请求超时，默认 `60` 秒，有效范围 `(0, 3600]`。 |
| `LLM_TRANS_THINKING_MODE` | 可选的端点专有 `thinking.type`；仅在端点支持时设置。`adaptive` 还会启用 MiniMax 思考内容分离。 |

执行 `uv run llm-omop-trans translate --help` 可查看全部参数。主要参数如下：

| 参数 | 行为 |
| --- | --- |
| `--input`、`--vocabulary-id` | 必填输入路径与大小写敏感的目标词表 ID。 |
| `--concurrency`、`--batch-size` | 正整数并发数（默认 `1`）与调度批次大小（默认 `20`）；每条 API 请求仍只含一个 Concept。 |
| `--pilot-only`、`--pilot-size`、`--pilot-label` | 独立 Pilot；样本数默认 `3`，范围 `1..20`；标识可区分重试。 |
| `--require-pilot`、`--approved-pilot-run-id` | 正式运行需要通过机器门禁的 Pilot，并在必要时指定人工查看过的 Pilot。 |
| `--resume-run-id`、`--model-name`、`--fallback-model` | 续跑、覆盖模型或按优先顺序追加后备模型（最后一项可重复）。 |
| `--allow-code-hash-drift`、`--allow-endpoint-switch`、`--retry-auth-failures` | 显式审计恢复选项；跨端点需要在新端点通过有效 Pilot。 |

输入、Athena、Prompt、后检、端点或代码变化均可能阻止普通续跑。切换模型不会覆盖已成功记录及其实际模型名。

## 快速开始

配置 `.env` 后，向已授权模型发送**一条虚构记录**：

```powershell
uv run --locked llm-omop-trans translate --input .\examples\synthetic\input.csv --vocabulary-id DEMO --pilot-only --pilot-size 1 --concurrency 1 --batch-size 1
```

命令会打印 `outputs/test/<run-id>/omop-trans.csv`。请查看 CSV 和复核文件；Pilot 机器通过不等于专家批准。

## 使用示例

```powershell
# 对获授权输入执行小样本 Pilot，查看输出和 run-id。
uv run --locked llm-omop-trans translate --input .\rawdata\authorized-input.csv --vocabulary-id AUTHORIZED_ID --pilot-only --pilot-size 3 --concurrency 1 --batch-size 3

# 核对范围、端点、来源许可和 Pilot 后，再执行正式运行。
uv run --locked llm-omop-trans translate --input .\rawdata\authorized-input.csv --vocabulary-id AUTHORIZED_ID --require-pilot --approved-pilot-run-id run-PASTE_REAL_PILOT_ID --concurrency 1 --batch-size 10

# 同一 run-id 续跑，不重复发送已提交的成功请求。
uv run --locked llm-omop-trans translate --input .\rawdata\authorized-input.csv --vocabulary-id AUTHORIZED_ID --resume-run-id run-PASTE_REAL_RUN_ID --require-pilot --concurrency 1 --batch-size 10

# 待请求记录切换模型；额度耗尽时，后备模型先接受合成探测。
uv run --locked llm-omop-trans translate --input .\rawdata\authorized-input.csv --vocabulary-id AUTHORIZED_ID --resume-run-id run-PASTE_REAL_RUN_ID --model-name NEW_MODEL --fallback-model BACKUP_MODEL --require-pilot

# 合并兼容的正式运行并生成 Markdown 报告。
uv run --locked llm-omop-trans report --input .\rawdata\authorized-input.csv
```

请将占位符替换为获授权的真实路径、词表、模型和 run-id。合成样例的单条 Pilot 及续跑已经验证；真实数据命令是使用模板，不代表已获得真实数据外发授权。

## 输入输出

输入为 UTF-8、`~` 分隔 CSV，表头及**七列顺序必须完全一致**：

```text
concept_id~concept_name_zh~concept_name~domain_id~vocabulary_id~concept_class_id~model
900000001~~Left kidney~Condition~DEMO~Clinical Finding~
```

`concept_id` 必须是唯一的非负整数。程序先校验整份输入再选择目标词表，保留目标行原始字段。已有中文名、Athena 缺失或失效、词表身份冲突的记录不会发送。

正式运行写入 `outputs/<run-id>/omop-trans.csv`，Pilot 写入 `outputs/test/<run-id>/omop-trans.csv`；两者均按目标输入顺序生成以下九列：

```text
concept_id~concept_name_zh~concept_name~domain_id~vocabulary_id~concept_class_id~candidate_name_zh~review_status~model_name
```

`candidate_name_zh` 是候选，`concept_name_zh` 保持原始正式字段。非空候选在后检通过或需要复核时标为 `pending_expert`；后检失败仍保留证据，不自动接受。`model_name` 优先记录响应报告的实际模型，否则记录请求模型。每个运行目录还有 `run_manifest.json`、`state.duckdb` 和复核/异常 CSV。`report` 只汇总兼容的正式运行，不合并 Pilot，生成 `outputs/omop-trans-all.csv` 与 `outputs/llm-trans-report.md`。

`outputs/run-index.json` 按 run-id 记录运行类型、词表、用途、相对路径、状态及模型。仅在实际模型唯一时填写 `model_name`；`models_used` 保留调用台账与已保存成功行中的全部模型。程序在创建、续跑和结束时加锁并原子更新索引；索引缺失或损坏时可依据清单与 DuckDB 状态恢复：

```powershell
uv run --locked python -m llm_omop_trans.run_index
```

旧版位于 `outputs/<run-id>/` 的 Pilot 仍可读取和续跑。缺少类型字段的旧运行标为 `unknown`，不凭目录名推测。输出与索引均为 Git 忽略的本地运行产物。

## 项目结构

```text
llm-omop-trans/
├── llm_omop_trans/
│   ├── config.py       # 字段契约、Prompt、环境配置、外发拒绝清单
│   ├── llm_trans.py    # CLI、预检、调用、Pilot、状态与导出
│   ├── validator.py    # 14 类确定性医学属性检查
│   ├── report.py       # 正式运行合并与质量报告
│   ├── run_index.py    # 加锁运行索引与 Pilot 路径解析
│   └── utils.py        # 严格 JSON 与哈希
├── llm_trans_main.py   # 兼容的 Python CLI 入口
├── examples/synthetic/ # 虚构输入与三个小型快照表
├── tests/             # 合成单元测试与本地 HTTP 集成测试
├── docs/SOURCE_POLICY.md
├── .env.example        # 无密钥配置模板
├── pyproject.toml
├── uv.toml
└── uv.lock
```

运行输出结构：

```text
outputs/
├── test/<run-id>/      # Pilot 清单、DuckDB 状态、CSV
├── <run-id>/           # 正式运行清单、DuckDB 状态、CSV
└── run-index.json      # 可重建的元数据索引
```

## 许可证

**本仓库尚未添加或授予软件许可证。** 仓库公开可见本身不代表允许复制、修改或再分发代码；复用前须获得所有者许可。真实 OMOP/Athena 等术语数据未包含在仓库内，仍受其来源许可证和[外发政策](docs/SOURCE_POLICY.md)约束。生成的候选仅供内部审核，不自动成为获批准译名，也不获得公开再分发许可。
