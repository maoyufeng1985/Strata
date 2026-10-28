# Whittle-45B 在 Flash-Next 引擎上：怎么跑（2026-10-28）

## 最佳命令（P100 主卡 + 三张 P104 作专家缓存层，serve 模式）

```sh
M=/home/models/Whittle-Qwen-3.8-45B-A3B-Q4_K_M.gguf
PACK=/home/Strata/packs/whittle45b
B=/home/Strata/build-flash
PROF=/home/Strata/.run/whittle-20261028/profile.bin      # 必须是真路由轨迹建的

printf 'GEN 256 %s\n' "$IDS" | "$B/strata" --pack "$PACK" --native "$M" \
    --ple-gguf "$M" --ple-io ram \
    --max-context 4096 --spec 2 --suffix-draft 0 \
    --expert-profile "$PROF" --expert-cache auto \
    --gpu 2,1,0,3 --kv int8 --prefill 512 \
    --expert-cache-device1 auto --expert-cache-device2 auto --expert-cache-device3 auto \
    --conversation-cache-mib 0 \
    --remote-expert-opt --expert-cache-remote-placement layer --serve
```

> `--conversation-cache-mib 0`：serve 模式默认按空闲内存留对话缓存（最多 8 GiB）。这台机器上
> 引擎已经占了 23.8 GiB（专家 arena 18.16 + PLE 表 5.62），31 GiB 里剩不下多少，测速时关掉。

实测（276 token 提示、128 token 生成，P100）：**decode 47.8 tok/s**；不开 `--remote-expert-opt`
是 42.1 tok/s。

> `--remote-expert-opt` 只在 `--serve` 下允许（`generate.cpp:3371`）。serve 模式就是引擎读 stdin 的
> `GEN <max_new> <ids>` 行、写 stdout 的 `T <id>` 行；桌面 app 和 Terminal 走的就是这条路。

## 每条参数为什么在那里

| 参数 | 为什么 |
|---|---|
| `--native "$M"` | pack 是 native（IQ）格式，必须给 `--native` |
| `--ple-gguf "$M"` + `--ple-io ram` | PLE 表（3904 万行 Q4_K，5.62 GiB）不在 pack 里，直接读 GGUF；`ram` 把它**锁进内存**（"PLE table locked in RAM in 0.5 s"），每个 token 约 0.04 ms，不在关键路径上 |
| `--spec 2` | native pack 的硬性要求（`--spec T`，T ≥ 2）。它只决定捕获的图宽与缓冲区；**没有 drafter 时一轮只跑 1 行**（引擎已改） |
| `--suffix-draft 0` | Whittle 没有 in-file MTP 头；suffix drafter 在这个模型上接受 0 个 draft。开着只是白算 |
| `--expert-profile` + `--expert-cache auto` | `--spec`/verify 路径的硬性要求；同时决定 4778 个专家槽（8.5 GiB）放谁 |
| `--gpu 2,1,0,3` | **nvidia-smi 下标**，不是 CUDA ordinal：这台机器 P100 = 2，三张 P104 = 1/0/3 |
| `--kv int8` | 和 fp32/k8v4 在 decode 上打平，int8 的 KV 更小 ⇒ 上下文更省 |
| `--prefill 512` | 276 token 的提示只有一块；大块对长提示更好（1799 token：468 tok/s） |
| `--expert-cache-device1..3 auto` | 三张 P104 各放 1820 个专家，CPU 每层未驻留专家从 1.36 降到 0.05 |
| `--remote-expert-opt` | **本轮最大的一笔，+12%**：helper 在自己卡上把专家输出加权归约成一个向量再回传，PCIe 上的回传量降到 1/8（每 helper 16.4 MiB 而不是 131.5 MiB），主卡等 helper 的时间从 223 ms 降到 129 ms（128 token） |
| `--expert-cache-remote-placement layer` | 每一层归一个 helper：层启动次数 6703 → 4098，staging 239 → 161 ms（128 token），+1.2% |

## 三条实测出来的规律

1. **profile 必须是真路由轨迹建的。** `--dump-routing` 跑一次，再
   `python tools/make_profile.py TRACE --no-base --reorder --n-layer 40 --n-expert 256 --out profile.bin`。
   用交错填充的 profile：decode 18.4 tok/s；用轨迹的：29.8 tok/s（**+62%**）。
2. **没有 drafter 时窗口必须是 1 行。** `--spec 2` 会白跑一行，代价 5.2–6.5 ms/轮（T1 22.58、
   T2 29.13、T3 34.30、T4 41.67 ms）。引擎已改为自动取 1 行 ⇒ 32.4 → 40.3 tok/s（**+24%**）。
3. **helper 的回传量是 decode 的瓶颈，不是专家的算力。** 三个 helper 每个每层都要回传
   `(专家数, token)` 行的 2048 个 float。`--remote-expert-opt` 把它变成每 token 一个加权向量。

## 不要用的

| 配置 | 结果 |
|---|---|
| `--spec 3/4/5` | 窗口更宽、drafts 0 接受 ⇒ 更慢（14.9 / 22.7 tok/s） |
| `--expert-cache 512`（槽太少）| 10.2 tok/s |
| `--peer-device 1` | 33.8 tok/s（比三张 helper 缓存慢），且与 `--expert-cache-device*` 互斥 |
| `STRATA_PF_F16_CUDA=1` | 与 MMQ 打平；且只在 streamed-all 路径里生效 |
| `STRATA_DENSE_F16_ACC=1` | decode 36.95 vs 36.87 —— 噪声内 |
| `STRATA_HC_PERSIST=1` | **CUDA 侧是空实现**（`fused_gr.cu:3042`），只有 HIP 有这个核；gfx1030 的数不能搬 |
| `STRATA_HC_UPMIX=1` / `STRATA_HCD_EXACT=1` | 只在 prefill 路径上 |
| `STRATA_GDN_CHUNKED=1/2` | 47.8 / 47.7 对基准 47.2 —— 噪声内（±1%） |
| `STRATA_TSUM=1` | 47.7 —— 噪声内 |
| `--gpu` 四卡跑 prefill | 340.5 vs 单卡 340.1（699 token）、468.5 vs 472.3（1799 token）⇒ **没有收益**，PCIe gen1 太慢 |

## 构建（盒子上）

```sh
cmake -S /home/Strata/src-flash -B /home/Strata/build-flash \
    -DCMAKE_BUILD_TYPE=Release -DSTRATA_ISA_FLOOR=avx \
    -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON \
    -DCMAKE_CUDA_ARCHITECTURES="60;61" \
    -DSTRATA_GGML_DIR=/home/Strata/build-avx/_deps/strata_llamacpp-src
cmake --build /home/Strata/build-flash --target strata -j 12
```

- `STRATA_ISA_FLOOR=avx`：盒子 CPU 是 Ivy Bridge（无 AVX2/FMA）。
- `STRATA_GGML_DIR` 指向已取好的 pinned llama.cpp，configure 不需要联网。
- 全量 ≈ 45 分钟；增量（改几个文件）≈ 20 秒–10 分钟。

## 本地 → 盒子的同步

`git archive HEAD` 取基线 → 覆盖工作树里改过的文件（行尾统一 LF）→ tar → 推 → 解到 `/home/Strata/src-flash`。
单个文件的小改动可以直接推（`.run/push98.sh`），但推完必须 `touch` 源文件，否则 make 不重编。

## 一次跑一个引擎

盒子 31 GiB RAM，专家 arena 18.16 GiB + PLE 表 5.62 GiB —— 同时跑两个引擎会 OOM。
**而且一个残留的引擎会把 load average 顶到 50+**，之后每一臂的 prefill 数都不可信
（本次就出现过：`best` 一臂 prefill 只有 103.6 tok/s，因为上一个 32K 上下文的进程还在）。
测量前必须 `pgrep -x strata` 为 0 且 `loadavg` < 4。

**扫描脚本**（`.run/`，不进仓库）：`R-flash-bench.sh`（generate 模式的配置矩阵）、
`R-flash-serve.sh` / `R-flash-serve2.sh` / `R-flash-serve3.sh`（serve 模式的矩阵，
`--remote-expert-opt` 只能用 serve）、`R-flash-opt.sh`（generate 模式的旋钮扫描）、
`R-flash-decode-serve.py`（把 `T <id>` 行解回文本）、`R-mktok.py`、`R-flash-profile.sh`。
写 bash 的 `case` 时**每个分支都要写 `;;`**（这台机器的 bash 5.2 不接受靠换行结束分支）。
