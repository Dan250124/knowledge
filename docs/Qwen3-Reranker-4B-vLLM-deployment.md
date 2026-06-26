# Qwen/Qwen3-Reranker-4B · vLLM 部署总结

> 本文记录 Qwen3-Reranker-4B 在 vLLM 上正确部署与调用的踩坑结论，已在本环境验证可用。

---

## 一、最重要的坑（先看这个）

Qwen3-Reranker-4B 的 `config.json` 里 `architectures` 是 **`Qwen3ForCausalLM`**（生成式模型）。它的相关度分数来自模型在特定位置生成 `yes`/`no` 的概率，**不是** embedding 的余弦相似度。

如果直接用：

```bash
# ❌ 错误写法：会被当成 bi-encoder
vllm serve Qwen/Qwen3-Reranker-4B --runner pooling --host 0.0.0.0 --port 80
```

vLLM 会把 CausalLM **默认转成 embedding 模型（bi-encoder）**，`/score`、`/v1/rerank` 实际算的是 **query 与 doc 向量的余弦相似度**，而不是 cross-encoder 分类打分。

| 现象（错误部署时） | 说明 |
|---|---|
| 分数全挤在 **0.5 ~ 0.8** | 余弦相似度的典型区间，没有区分度 |
| 最相关的文档反而垫底 | 余弦抓不到“是非问句 vs 长多主题文档”的相关性 |
| 零相关文档扎堆偏高 | bi-encoder 只反映领域/表层相似 |

**正确分数形态**：相关项 → **0.9+**，无关项 → **<0.1**，两极分化。

---

## 二、部署命令（已验证可用）

```bash
vllm serve Qwen/Qwen3-Reranker-4B \
  --host 0.0.0.0 --port 80 \
  --gpu-memory-utilization 0.9 \
  --hf_overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' \
  --max-model-len 4096
```

### 参数说明

| 参数 | 作用 |
|---|---|
| `--hf_overrides`（核心） | 见下表 |
| `--gpu-memory-utilization 0.9` | 显存占用比例 |
| `--max-model-len 4096` | 最大序列长度（4B 原始支持 32k，按需调；调小可省显存） |

`--hf_overrides` 三个字段：

- **`architectures: ["Qwen3ForSequenceClassification"]`** — 把 CausalLM 转成序列分类模型，启用 cross-encoder 打分路径。
- **`classifier_from_token: ["no","yes"]`** — 从 `no`/`yes` 两个 token 的 logit 取分；顺序为 index0=`no`、index1=`yes`，`score = softmax[1] = P(yes)`。
- **`is_original_qwen3_reranker: true`** — 标记是官方原版 Qwen3-Reranker，触发 vLLM 内置的 `qwen3_reranker.jinja` score 模板（自动套官方 system prompt + `<think>` 结尾）。

> ✅ **不再需要 `--runner pooling`**：hf_overrides 把架构改成了 SequenceClassification，vLLM 会自动识别为 scoring 模型。

---

## 三、调用方式

### 方式一：`/score` + 手动构造官方模板（已验证可用）

把官方 prompt 模板拆成 query 侧（prefix）和 document 侧（suffix），分别传给 `text_1` / `text_2`，vLLM 拼接后取 yes/no logit。

```python
import requests

url = "http://<your-vllm-host>:80/score"
MODEL_NAME = "Qwen/Qwen3-Reranker-4B"

prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

query_template    = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
document_template = "<Document>: {doc}{suffix}"

instruction = "Given a web search query, retrieve relevant passages that answer the query"

queries = ["王晓明喜欢拍照吗"]
documents = [
    "王晓明喜欢拍照。",
    "王晓明不喜欢拍照。",
    "低端局abc喜欢撤回消息。",
    "苹果是一种水果。",
]

queries = [query_template.format(prefix=prefix, instruction=instruction, query=q) for q in queries]
documents = [document_template.format(doc=d, suffix=suffix) for d in documents]

response = requests.post(url, json={
    "model": MODEL_NAME,
    "text_1": queries,
    "text_2": documents,
    "truncate_prompt_tokens": -1,
}).json()

print(response)
```

期望结果（cross-encoder 正常工作）：

- `"王晓明喜欢拍照。"` → 高分（≈ 0.9+）
- `"王晓明不喜欢拍照。"` / 其它 → 低分（< 0.1）

### 方式二：`/v1/rerank` + 显式 jinja 模板（让高级接口可用）

> 第二节的默认部署仅靠 `is_original_qwen3_reranker:true` **不会自动套用 score 模板**（vLLM 已知行为，见 issue #39784）。`/v1/rerank` 依赖该模板，直接调会失真；要让它可用，需**显式提供 jinja**。

**① 新建 `qwen3_reranker.jinja`**（与启动脚本同目录），内容即官方模板：

```jinja
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {{ instruction | default(instruct | default(messages | selectattr("role", "eq", "system") | map(attribute="content") | first | default("Given a web search query, retrieve relevant passages that answer the query", true), true), true) }}
<Query>: {{ messages | selectattr("role", "eq", "query") | map(attribute="content") | first }}
<Document>: {{ messages | selectattr("role", "eq", "document") | map(attribute="content") | first }}<|im_end|>
<|im_start|>assistant
<think>

</think>
```

**② 部署命令追加 `--chat-template`**：

```bash
vllm serve Qwen/Qwen3-Reranker-4B \
  --host 0.0.0.0 --port 80 \
  --gpu-memory-utilization 0.9 \
  --hf_overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' \
  --chat-template ./qwen3_reranker.jinja \
  --max-model-len 4096
```

**③ 用 `/v1/rerank` 调用**（传**纯** `query`/`documents`，不要再手写 prefix/suffix）：

```bash
curl -X POST "http://<your-vllm-host>:80/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Reranker-4B",
    "query": "王晓明喜欢拍照吗",
    "documents": [
      "王晓明喜欢拍照。",
      "王晓明不喜欢拍照。",
      "低端局abc喜欢撤回消息。",
      "苹果是一种水果。"
    ],
    "top_n": 4
  }'
```

> ⚠️ 若加了 `--chat-template` 后分数反而变差（模板被套了两次），把 `hf_overrides` 里的 `is_original_qwen3_reranker` 去掉，只留 `--chat-template`。

---

## 四、打分原理

1. vLLM 把 `[system: Judge whether…] + <Instruct>/<Query>/<Document> + <think></think>` 拼成完整 prompt；
2. 对**最后一个 token 位置**取 `no` / `yes` 两个 token 的 logit；
3. `score = exp(yes) / (exp(yes) + exp(no))`，得到 `[0, 1]` 区间的相关度概率。

所以**应看相对排序和两极分化程度**，而不是绝对数值。健康表现是“相关 ≈ 1、无关 ≈ 0”。

---

## 五、验证 & 排查清单

部署后先确认是否真正走 cross-encoder：

- [ ] **看启动日志**：应出现 `task: classify` / `score_type: cross-encoder`；若出现 `task: embed` / `score_type: bi-encoder`，说明还是 bi-encoder，检查 `--hf_overrides` 是否生效。
- [ ] **跑个强对比用例**（如上面的拍照例子）：相关项应 ≈ 0.9+，无关项应 < 0.1。若分数仍挤在 0.5~0.8，多半还是被当成了 bi-encoder。
- [ ] **核对 `--hf_overrides` JSON**：注意引号转义，shell 里建议用单引号包外面、双引号包里面。
- [ ] **vLLM 版本**：建议较新版本（≥ 0.8.5 起 vLLM 才较好支持 Qwen3-Reranker）。已知相关 issue：vllm #35412（Qwen3-Reranker 分数错误）。

---

## 六、可选优化

- **加 instruction**：`/v1/rerank` 和 `/score` 都支持 `instruction` 字段；官方建议 instruction 用英文，可带来 1%~5% 的提升。
- **max_model_len**：4B 原生支持 32k，长文档场景可调大，注意显存。
- **prefix caching**：批量打分时可加 `--enable-prefix-caching` 提速（system/instruct 前缀可复用）。

---

## 七、参考

- 模型卡：<https://huggingface.co/Qwen/Qwen3-Reranker-4B>
- vLLM Scoring 文档（score_type 映射、hf_overrides、score template 规则）：<https://docs.vllm.ai/en/stable/models/pooling_models/scoring.html>
- vLLM 在线示例：<https://github.com/vllm-project/vllm/blob/main/examples/pooling/score/qwen3_reranker_online.py>
- 技术报告：<https://arxiv.org/html/2506.05176v1>
