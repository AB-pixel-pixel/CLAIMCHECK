# ClaimCheck

## 概述

ClaimCheck 是一个事实核查系统，它会处理声明，并使用各种模块和模型来验证这些声明的真实性。

## 前置条件

1. 在 Google 中创建一个新的 Programmable Search Engine：

   * 前往 [Programmable Search Engine](https://cse.google.com/cse/) 并创建一个新的搜索引擎。
   * 记下 CSE ID。
   * 在 [Google Cloud Console](https://console.cloud.google.com/) 中启用 Custom Search JSON API。
   * 记下 API key。

2. 从 [SerpAPI](https://serper.dev/) 获取你的 API key。

## 安装

1. 克隆仓库：

   ```bash
   git clone https://github.com/idirlab/ClaimCheck.git
   cd ClaimCheck
   ```

2. 安装所需依赖：

   ```bash
   pip install -r requirements.txt
   ```

3. 在代码中更新 API key：

   * 打开 `claim_matching.py`，并更新 Google Cloud API key 和 CSE ID。
   * 打开 `search.py`，并更新 SerpAPI key。

## 使用方法

要从命令行运行事实核查系统，请使用 `fact-check.py` 脚本。它接收两个参数：包含声明的 JSON 文件路径，以及要处理的记录数量。

### 命令行参数：

* `json_path`：AVeriTeC JSON 文件的路径。
* `num_records`：要运行的声明数量。

### 示例：

```bash
python fact-check.py /path/to/json/file.json 5
```

将 `/path/to/json/file.json` 替换为你的 AVeriTeC JSON 文件的实际路径，并将 `5` 替换为你想要处理的记录数量。你可以在[这里](https://fever.ai/dataset/averitec.html)找到 AVeriTeC JSON 文件。
