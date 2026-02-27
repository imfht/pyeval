# PyEval

Python 编程能力评测框架。支持标准代码生成、Bug 修复、多轮纠错三种评测模式，通过 OpenAI 兼容 API 对接任意大语言模型。

## 特性

- **三种评测模式**：标准 / Bug 修复 / 多轮纠错，覆盖"写代码"和"改代码"两大核心能力
- **60 道题目**：45 道代码生成 + 15 道 Bug 修复，涵盖 10 个分类 × 3 个难度
- **沙箱执行**：代码在隔离子进程中运行，通过 unittest 验证正确性
- **并发评测**：API 调用和沙箱执行均支持可配置的并发数
- **零外部依赖**：纯 Python 标准库实现，无需安装第三方包
- **灵活配置**：JSON 配置文件 + CLI 参数，支持按分类/难度/题号筛选

## 快速开始

```bash
# 标准模式
python -m pyeval --api-base https://api.example.com/v1 --api-key YOUR_KEY --model gpt-4

# Bug 修复模式
python -m pyeval --mode bugfix --api-base https://api.example.com/v1 --api-key YOUR_KEY --model gpt-4

# 多轮纠错模式（最多 3 次尝试）
python -m pyeval --mode multiturn --max-attempts 3 --api-base https://api.example.com/v1 --api-key YOUR_KEY --model gpt-4

# 验证题目（用参考答案跑，不调 API）
python -m pyeval --dry-run
python -m pyeval --mode bugfix --dry-run
python -m pyeval --mode multiturn --dry-run
```

## 三种评测模式

### 标准模式（Standard）

给模型函数签名 + docstring，让它写完整实现。45 道题覆盖：

| 分类 | Easy | Medium | Hard | 合计 |
|------|------|--------|------|------|
| Basic Syntax | 3 | 1 | 1 | 5 |
| Data Structures | 2 | 2 | 1 | 5 |
| Algorithms | 2 | 3 | 2 | 7 |
| Standard Library | 2 | 2 | 1 | 5 |
| OOP | 1 | 2 | 2 | 5 |
| Exceptions | 1 | 1 | 1 | 3 |
| File I/O | 1 | 1 | 1 | 3 |
| String Processing | 2 | 1 | 1 | 4 |
| Functional | 1 | 2 | 1 | 4 |
| Concurrency | 1 | 2 | 1 | 4 |
| **合计** | **16** | **17** | **12** | **45** |

### Bug 修复模式（Bugfix）

给模型一段有 bug 的代码 + bug 描述，让它找到并修复 bug。15 道题覆盖经典 Python 陷阱：

| 难度 | 题目 |
|------|------|
| Easy | off-by-one、比较符写反、缺 return、变量拼错、条件逻辑错误 |
| Medium | 浅拷贝 vs 深拷贝、可变默认参数、错误异常类型、闭包变量捕获、整除 vs 浮点除 |
| Hard | 线程竞态条件、生成器耗尽、装饰器丢 wraps、钻石继承 MRO、async 上下文管理器 |

### 多轮纠错模式（Multiturn）

模拟真实开发中的"写代码 → 跑测试 → 看报错 → 改代码"循环：

- **第 1 次**：标准调用，通过得 **1.0** 分
- **第 2 次**：把报错反馈给模型重试，通过得 **0.6** 分
- **第 3 次**：再次反馈再试，通过得 **0.3** 分
- **全部失败**：0 分

加权分 = Σ(尝试权重 × 难度权重) / Σ(难度权重)

其中难度权重：easy=1, medium=2, hard=3

## 评分体系

| 模式 | 指标 |
|------|------|
| 标准 / Bugfix | 通过率、加权分（按难度） |
| 多轮纠错 | 首次通过率（Strict）、重试后通过率（With Retries）、加权分（综合尝试次数 + 难度） |

## CLI 参数

```
python -m pyeval [OPTIONS]

选项:
  --config PATH          配置文件路径（JSON）
  --api-base URL         OpenAI 兼容 API 地址
  --api-key KEY          API Key
  --model NAME           模型名称
  --mode MODE            评测模式：standard | bugfix | multiturn
  --max-attempts N       多轮模式最大尝试次数（默认 3）
  --temperature FLOAT    采样温度（默认 0.0）
  --max-tokens INT       最大生成 token 数（默认 2048）
  --timeout INT          API 超时秒数（默认 30）
  --sandbox-timeout INT  沙箱执行超时秒数（默认 10）
  --categories LIST      逗号分隔的分类筛选
  --difficulties LIST    逗号分隔的难度筛选（easy,medium,hard）
  --problem-ids LIST     逗号分隔的题目 ID 筛选
  --extra-body JSON      额外 API 请求参数
  --verbose              显示详细输出
  --dry-run              用参考答案验证题目
  --output-dir PATH      报告输出目录（默认 pyeval_results）
  --problems-dir PATH    题库目录
```

## 配置文件

创建 JSON 配置文件避免每次传参：

```json
{
    "api_base": "https://api.example.com/v1",
    "api_key": "YOUR_KEY",
    "model": "gpt-4",
    "temperature": 0.0,
    "max_tokens": 2048,
    "timeout": 30,
    "sandbox_timeout": 10,
    "max_concurrent_api": 4,
    "max_concurrent_sandbox": 8,
    "output_dir": "pyeval_results"
}
```

```bash
python -m pyeval --config my_config.json --mode multiturn
```

## 项目结构

```
pyeval/
├── __init__.py
├── __main__.py        # CLI 入口
├── config.py          # 配置加载
├── client.py          # LLM API 调用（标准/bugfix/multiturn）
├── runner.py          # 评测流程编排
├── scorer.py          # 评分与聚合
├── reporter.py        # 终端 + Markdown 报告生成
├── sandbox.py         # 隔离子进程执行代码
├── pyeval.json        # 默认配置
└── problems/
    ├── loader.py      # 题目加载与筛选
    └── bank/          # 题库（JSON）
        ├── 01_basic_syntax.json
        ├── 02_data_structures.json
        ├── 03_algorithms.json
        ├── 04_stdlib.json
        ├── 05_oop.json
        ├── 06_exceptions.json
        ├── 07_file_io.json
        ├── 08_string_processing.json
        ├── 09_functional.json
        ├── 10_concurrency.json
        ├── bugfix_easy.json
        ├── bugfix_medium.json
        └── bugfix_hard.json
```

## 评测结果（8 模型对比）

所有模型通过阿里云百炼 DashScope API 调用，temperature=0。

### 全景对比

| 模型 | 标准模式 | Bug 修复 | 多轮加权 |
|------|---------|----------|---------|
| qwen3.5-plus | 100% | 100% | **99.1%** |
| qwen3-max | 95.6% | 100% | **99.1%** |
| kimi-k2.5 | 93.3% | 100% | **99.1%** |
| qwen3.5-flash | 100% | 100% | **98.1%** |
| deepseek-v3.2 | 95.6% | 100% | **98.1%** |
| glm-5 | 97.8% | 93.3% | **96.3%** |
| MiniMax-M2.5 | 91.1% | 86.7% | **94.9%** |
| glm-4.7 | 93.3% | 93.3% | **86.0%** |

### 多轮纠错详情

| 模型 | 首次通过 | 重试后通过 | 加权分 |
|------|---------|-----------|--------|
| qwen3.5-plus | 97.8% | 100% | 99.1% |
| qwen3-max | 97.8% | 100% | 99.1% |
| kimi-k2.5 | 97.8% | 100% | 99.1% |
| qwen3.5-flash | 95.6% | 100% | 98.1% |
| deepseek-v3.2 | 95.6% | 100% | 98.1% |
| glm-5 | 95.6% | 97.8% | 96.3% |
| MiniMax-M2.5 | 88.9% | 100% | 94.9% |
| glm-4.7 | 80.0% | 93.3% | 86.0% |

### 区分度

- **标准模式**：91.1% — 100%（差 9pp）
- **多轮加权**：86.0% — 99.1%（差 13pp）

多轮模式清晰分出三个梯队：第一梯队（99%+）、第二梯队（94-98%）、第三梯队（86%）。

## License

MIT
