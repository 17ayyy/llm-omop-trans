# 使用指南：从 Pilot 到报告

以下命令均在本仓库根目录的 PowerShell 中执行。`llm-omop-trans` 是独立程序；命令中的路径、运行编号和授权范围不能从课程项目直接复制。本文只提供操作模板，不代表真实术语已获外发许可。真实数据使用前请核对[来源与外发政策](SOURCE_POLICY.md)。

## 1. 准备环境与模型配置

```powershell
uv sync --locked
Copy-Item .env.example .env
```

仓库的 `uv.toml` 使用 PyPI。若当前终端设置了不同的 `UV_DEFAULT_INDEX`，`uv --locked` 可能提示锁文件需要更新；确认要使用仓库索引后，可仅在当前终端执行 `Remove-Item Env:UV_DEFAULT_INDEX`，再重试。不要为解决索引覆盖而盲目重写 `uv.lock`。

在本机 `.env` 中填写 `LLM_BASE_URL`、`LLM_MODEL`、`LLM_API_KEY`，并按同一获许可 Athena 快照设置三个 `OMOP_*_PATH`、发布版和有效日。不要提交 `.env`。`--model-name` 只覆盖模型名；切换服务商时还需配置匹配的端点和密钥，并重新核对数据外发许可。

下面的占位符仅在**已经确认来源许可、外发范围及端点**后替换：

```powershell
$InputCsv = '.\rawdata\authorized-input.csv'
$Vocabulary = 'AUTHORIZED_ID'
$Model = 'AUTHORIZED_MODEL'
```

## 2. 连通性与 Pilot

先用仓库内虚构的 `DEMO` 记录检查端点、凭据与完整调用链。此命令会向 `.env` 配置的端点发送一条合成术语；它生成独立 Pilot，不启动真实数据正式运行。

```powershell
uv run --locked llm-omop-trans translate `
  --input .\examples\synthetic\input.csv --vocabulary-id DEMO `
  --model-name $Model --pilot-only --pilot-size 1 `
  --pilot-label demo-connectivity --concurrency 1 --batch-size 1
```

通过连通性检查后，对已授权的目标词表做 1–20 条独立 Pilot。首次尝试建议从 3 条及低并发开始；重复测试时使用新的 `--pilot-label`，避免把不同尝试误认成同一个运行。

```powershell
uv run --locked llm-omop-trans translate `
  --input $InputCsv --vocabulary-id $Vocabulary --model-name $Model `
  --pilot-only --pilot-size 3 --pilot-label initial-pilot `
  --concurrency 1 --batch-size 3
```

命令打印结果 CSV 路径。记录实际 `run-...`，查看 `outputs/test/<run-id>/` 中的 `omop-trans.csv`、`state.duckdb` 和 `run_manifest.json`，并核对 `outputs/run-index.json`。检查请求是否成功、候选是否非空、`model_name` 是否符合预期，以及部位、侧别、入路等医学含义。机器后检通过的候选仍是 `pending_expert`，需人工审核。`batch-size` 只控制调度批次，每条术语仍是一次独立请求。

## 3. 首次正式运行

仅在 Pilot 机器门禁通过、人工查看结果且数据外发范围确认后，填入该 Pilot 的实际运行编号：

```powershell
$PilotRunId = 'run-替换为实际Pilot编号'
uv run --locked llm-omop-trans translate `
  --input $InputCsv --vocabulary-id $Vocabulary --model-name $Model `
  --require-pilot --approved-pilot-run-id $PilotRunId `
  --concurrency 1 --batch-size 10
```

程序生成新的正式 `run-...`，写入 `outputs/<run-id>/`。并发与批次按 Pilot 的延迟、错误率和质量调整。不要把 Pilot 的 run-id 当作正式 run-id。

## 4. 断点续跑与切换模型

中断后用**正式运行**的 run-id、相同输入和词表续跑。已提交的成功行会按缓存和请求指纹复核，不会重发。

```powershell
$FullRunId = 'run-替换为实际正式编号'
uv run --locked llm-omop-trans translate `
  --input $InputCsv --vocabulary-id $Vocabulary --model-name $Model `
  --resume-run-id $FullRunId --concurrency 1 --batch-size 10
```

同一端点切换模型时，先为新模型单独做 Pilot 并人工检查。确认 `.env` 中的端点与密钥对新模型有效，再在续跑命令中把 `--model-name` 换成新模型。旧成功行保留原模型记录，待请求行使用新模型；模型名变化本身不更换端点或密钥。

跨端点续跑还要在新端点配置匹配的 `LLM_BASE_URL`、密钥及模型，重新确认外发许可，并在新端点做合格 Pilot。人工查看新 Pilot 后，才可明确指定它和端点切换选项：

```powershell
$NewPilotRunId = 'run-替换为新端点实际Pilot编号'
uv run --locked llm-omop-trans translate `
  --input $InputCsv --vocabulary-id $Vocabulary --model-name $Model `
  --resume-run-id $FullRunId --allow-endpoint-switch `
  --approved-pilot-run-id $NewPilotRunId `
  --concurrency 1 --batch-size 10
```

源码变化导致旧运行拒绝续跑时，先核对输入、Athena、Prompt、响应结构和后检规则是否仍兼容；只有确认属于允许的实现哈希变化，才显式使用 `--allow-code-hash-drift`。不要把它作为常规参数。

## 5. 汇总与报告

对同一输入生成汇总 CSV 和 Markdown 报告。程序只汇总兼容的正式运行，排除 Pilot；报告不会把候选自动写入正式中文名。

```powershell
uv run --locked llm-omop-trans report --input $InputCsv
```

查看 `outputs/omop-trans-all.csv` 与 `outputs/llm-trans-report.md`。两者及运行目录属于本地输出，不应提交到 Git。
