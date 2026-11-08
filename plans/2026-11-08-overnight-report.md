# 2026-11-08 夜间工作报告

承接指令：继续 MTP 采样投机，无人推进到早上 10 点，多扫几遍 vLLM 源码优化并发下的显卡占空比。
全程钉住 **S=2**（4 会话按指示排除）。

---

## 一、一句话结论

**`STRATA_Q36_READ_WINDOWS` 这个旋钮是真有效的，代价线性、可外推、不饱和。**S=2 下每毫秒提示词时间换到的
解码 token 比单槽多 39%，混合流量再高 16%。但它是**延迟重分配，不是吞吐**——窗口时间加在读取上，不相交。

---

## 二、测出来的数（全部新测，盒子 192.168.1.98，每臂先重置三张 P104 + 预热）

### 2.1 双槽窗口成本曲线（`.backup/s2_interleave_2slot.py` + `s2_interleave_share.py`）

| `rw` | 窗口数 | 提示词 | tok/s | 槽0 | 槽1 | 合计 |
|---:|---:|---:|---:|---:|---:|---:|
| 0（对照） | 0 | 4,142 ms | 845 | 0 | 0 | **0** |
| 1（默认） | 6 | 4,401 | 795 | 11 | 11 | 22 |
| 8 | 48 | 5,996 | 584 | 108 | 108 | 216 |
| 16 | 96 | 7,781 | 450 | 218 | 218 | 436 |
| 32 | 192 | 11,505 | 304 | 460 | 461 | **921** |

**三个边际点相差 4%**：38.0 / 37.2 / 38.8 ms 每窗。→ 公平份额公式可外推：`windows = chunk_ms / 38`。
chunk 512 的半份额 ≈ **8 窗**，默认是 **1**。

### 2.2 混合提示词 vs 克隆提示词（`.backup/s2_interleave_mixed*.py`）

| `rw` | 克隆（同提示词） | 混合（不同提示词） | 优势 | 提示词成本 |
|---:|---:|---:|---:|---:|
| 1 | 22 | 28 | **+27.3%** | −0.5% |
| 8 | 216 | 251 | **+16.2%** | +0.2% |
| 16 | 436 | 505 | **+15.8%** | +1.2% |

**三条**：
1. 提示词耗时不变（噪声内），产出升 16–27% → 克隆 fixture 是**下界**，不是乐观值。
2. `rw≥8` 稳定在 **~16%**（两点差 0.4pp），`rw=1` 的 27% 是追赶窗口。
3. **`dec0` 三点全部逐 token 相同**（11 / 108 / 218）→ 多出的全归槽 1，窗口授予**没有抢一个槽给另一个**。

---

## 三、vLLM 源码六轮扫描（全部带 file:line，源码在 `/d/work/vllm-src`）

| 轮 | 问题 | 结论 | 能否移植 |
|---|---|---|---|
| 1-2 | 独立 prefill 线程 / 双计算流？ | **没有**。async scheduling 是主机侧重叠 | **不能**——q36 解码是主机阻塞的，且 `main_hybrid` 下没有捕获图 |
| 3 | chunked prefill 是 token 还是时间预算？ | **token**（`scheduler.py:741`），decode 先花（`:815-817`），prefill 吃剩余（`:1097`/`:1102`）；`long_prefill_token_threshold` 默认 0 | **无可抄设计**——比 `generate.cpp` 弱，比 q36 强 |
| 4 | 投机解码并发 | 每请求 `draft_len`，K 只按 **batch size** 调（`scheduler.py:1787-1791`），从不按接受率 | **不用**——q36 的 `slack = vrows-S-nc` 是同一件事的精确算法 |
| 5 | MoE all-to-all 重叠 | **DBO 存在**：2 microbatch、各带 `compute_stream`、共享 `comm_stream`（`ubatching.py:16`、`gpu_ubatch_wrapper.py:76`），**默认关** | **唯一活假设**——q36 两窗交错失败（−7.3%/−41.5%）的根因是共用一个 `stream_`，vLLM 不共用 |
| 6 | prefix caching | **默认开**（`config/cache.py:260`）；命中的 token **不占**每步预算（`scheduler.py:1362`/`:1638`） | **打的正是读取本身**——q36 只有会话内 `reuse`，没有跨请求块缓存 |

**要点**：你贴的那份方案（独立 prefill 线程 + 双 CUDA 流）**不是 vLLM 的做法**。vLLM 的三个答案是：
token 预算 decode 优先、DBO（默认关）、prefix cache 把读取消掉。只有第一个能对上本树，而本树
`generate.cpp` 已经有更强的版本。

---

## 四、最值钱的一条：q36 缺一个按时间比例的预算

`src/program/generate.cpp:17235` + `:17281-17286`：

```cpp
const auto tq = Clock::now();                       // 本块开始
...
const double budget = decode_share * elapsed_ms_since(tq);
do { batch_step(); } while (batch_on() && elapsed_ms(td) < budget);
```

`STRATA_BATCH_DECODE_SHARE` 默认 **0.5**——把本块耗时的**一半**还给解码槽。

而 `src/qwen35/q36_engine.cu` 只有固定计数：

```
9375:    int read_windows = 1;
9378:        if (const char* v = std::getenv("STRATA_Q36_READ_WINDOWS")) read_windows = ...;
10136:                for (int w = 0; w < read_windows && n_active > 0; ++w) window();
```

**同一个树里两个引擎，一个有时间预算一个没有。** 固定计数表达不了公平份额：chunk 512 半份额 ≈ 8，
chunk 2048 ≈ 33，chunk 4096 超过 64 的上限。默认 1 在每个 chunk 尺寸下都差一个数量级。

**没有实现**，原因写在计划文档里：① 改默认会改行为，`CONTRIBUTING.md` 要求默认路径逐字节不变，只能
opt-in；② `q36_engine.cu` 有**不是我做的**未提交改动，此时编辑会缠在一起；③ 一次一臂的测量还不足以
改默认值。

---

## 五、两个脚本 bug（都不是引擎行为），都修了

1. **条件变量漏 notify `BT`**：`wait_bt` 数 `BT` 行，但 reader 只在 `BADM`/`BDONE`/`DONE` 时唤醒。
   于是那道门睡到 slot 0 打 `BDONE`（47013.9 ms），长提示词推迟 42 秒提交，读到的是解码器跑完之后。
   指纹：`BDONE` 的 ms 字段**小于** `prompt_ms`，或第一个 `PP` 出现在 `BDONE` 之后。
   **`rw=0` 对照当时照样通过**，是靠对称性控制抓到的。
2. **host 侧等待把请求往后推而不是排序**：引擎 stdin 严格按序，`request()` 读完一个提示词才 pop 下一行
   （`:10371-10372`），所以三条 `BGEN` 一次 `write` 即可，中间任何等待都是纯延迟。

---

## 六、Reviewer

| 对象 | VERDICT | 处理 |
|---|---|---|
| `.backup/s2_interleave.py` | **FAIL** | 按 blockers 修复，**未复验** |
| `.backup/s2_interleave_2slot.py`（首版） | **FAIL** | 按 blockers 修复，**未复验** |
| `.backup/s2_interleave_2slot.py`（终版） | **PASS_WITH_NITS**，无 blocker | 见下 |

终版 Reviewer 的 NIT 1 预言对称性控制"会在好数据上触发"，理由是两槽接受率不同应漂移。**实测没触发**
（108/108、218/218、460/461）——它反倒让我发现对称来自 fixture（同提示词 → 同路由），并直接催生了
混合提示词那一组测量，结论是混合流量**更好**不是更差。其余 NIT 记录在计划文档的 review 表里。

---

## 七、改动清单

| 文件 | 改动 |
|---|---|
| `docs/QWEN36.md` | +101/−1：双槽表、阶梯表、混合提示词表、每窗成本口径 |
| `plans/2026-11-08-vllm-concurrency-occupancy.md` | 新建，~1,610 行：六轮 vLLM 扫描、全部测量、移植设计、review 记录 |
| `.backup/s2_interleave*.py` | 4 个探针脚本（未进仓库，本地+盒子各一份） |
| `src/qwen35/q36_engine.cu` | **未动**（工作区里原有的改动不是我做的） |

---

## 八、还没做的

- **时间预算的 A/B**：设计写好了，代码没写。要动 `q36_engine.cu`，得先确认工作区改动归属。
- **`q36_engine.cu:5635` 的 IMA**：有真实 Xid 31 / FAULT_PDE READ，已知缺陷，未修，不在本次范围。
- **prefix cache 的可行性**：六轮里唯一打读取本身的机制，比旋钮大得多，未评估。
