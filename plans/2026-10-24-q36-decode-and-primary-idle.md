# q36：decode 的时间去哪了，和「主卡闲着」这件事的账

日期：2026-10-24。目标机：192.168.1.98（1× Tesla P100-12GB 主卡 + 3× P104-100 8GB helper，16 核）。
二进制：`bin/strata-q36.final`（md5 `93595e17d8943c072d10ad342bfdd6f4`）。

## 0. 硬件的一个硬事实（先核过）

| 卡 | 架构 | compute cap | `__dp4a` |
|---|---|---|---|
| 主卡 Tesla P100-12GB | GP100 | **6.0** | **没有** → 走 `dp4a.hpp` 的 4×vmad 回退（实测慢 2.2×） |
| 3× P104-100 | GP104 | **6.1** | **有真硬件指令** |

远端 `CMAKE_CUDA_ARCHITECTURES=60;61` —— 同一份源码为两张卡编两份，
**helper 跑的是有 `__dp4a` 的那份，主卡跑的是回退的那份**。

所以「helper 是不是充分利用了 DP4A」这个问题的答案是：**helper 已经在用它能用的最快指令了**，
问题不在 helper 侧，在**主卡侧**（它比 helper 慢 2.2×），以及**主卡和 helper 的负载分配**。

## 1. decode 的分解（MEASURED）

`--bench-pp 512 --bench-tg 64`，3 helper，`STRATA_Q36_HELPER_TIMING=1`。
warm（bench 1/2）：**tg64 = 51.3 tok/s** = 19.5 ms/token。

| 段 | ms/token | 在谁身上 |
|---|---|---|
| **主卡 host 同步** | **6.258** | **主卡 CPU**（7680 calls = 40 层 × 192 token） |
| router | 0.501 | 主卡 GPU |
| helper CUDA1 `ms_begin` | 1.568 | helper GPU |
| helper CUDA2 `ms_begin` | 1.476 | helper GPU |
| helper CUDA3 `ms_begin` | 1.504 | helper GPU |
| helper CUDA1 `ms_wait` | 0.233 | helper |
| helper CUDA2 `ms_wait` | 0.285 | helper |
| helper CUDA3 `ms_wait` | **0.751** | helper |
| route_spin | 0.169 | 主卡 CPU |

**头号瓶颈是主卡那 6.258 ms/token 的 host 同步**，占 19.5 ms/token 的 **32%**。
它是 `moe_helpers()` 里那句 `cudaStreamSynchronize(st)`（`q36_engine.cu:3675`）——
每层每 token 一次，等的是主卡自己那条流（attention + SSM + router）。

三张 helper 的 `ms_begin` 加起来 4.55 ms/token，但它们**互相并行**，
所以 decode 的关键路径是 **主卡的 6.258 同步 + 主卡的 GPU 活**，helper 是侧翼。

## 2. 「主卡闲着」这件事，账已经算清

**prompt 时**（P4 形状，稳态 bench 2，每层 32.12 ms）：

```
mixer 7.144 | router_route 0.740 | route_d2h 0.062 | plan 9.124
primary_work 8.205 | helper_wait 6.595 | combine 0.234 | post 0.018
```

主卡 GPU 的活 = 7.144 + 0.740 + 8.205 + 0.234 = **16.3 ms/层**，
主卡 host 的活 = plan 里的 **6.9 ms**（`peer_plan` 的墙）。

**主卡的「专家算术」是 0** —— 3 张 helper 按 `expert % 3` 持走了每层每一个专家
（3440 + 3400 + 3400 = **10240 = 40 层 × 256**），所以 `peer_plan` 数出 `n_local = 0`，
主卡的 MMQ 在 served 层上无行可算。

**但主卡 VRAM 里那 39 个常驻专家（2.86 GiB）的权重白占着**：
`stream_kick` 把主卡常驻的专家从 `L.blobs`（VRAM）拷进 staging 缓冲（每层 73 MiB 的 D2D），
而主卡的 MMQ 一行都不读它们 —— **每层 73 MiB、40 层共 2.86 GiB 的无用拷贝**。

**后来（2026-10-24）发现：主卡那 8.205 ms/层的「专家算术」其实不是 0，
而是一遍 helper 已经算过的、会越界写崩主卡的垃圾** —— 见
`plans/2026-10-24-q36-peer-owner-encoding.md`（归属表编码 bug）。
它不只是浪费：dmesg 实锤主卡 P100 打出 6 次 `Xid 13`（Out Of Range Address）
+ `Xid 43`（崩溃进程 `strata-q36.fina`），随后三张 P104 相继 `Xid 79` 掉总线，
重启后主卡（PCI 05:00）不再枚举。§1 表里的 `primary_work 8.205` 现在归因到那个 bug。

## 3. 用户的两个提议，逐一算账

### 提议①：「保持全部权重放 3 helper 的前提下，主卡从内存流式计算 40% 权重，3 helper 各计算 20%」

**在这个盒子上不成立**，原因是 PCIe：

| 量 | 数 |
|---|---|
| 一个专家的权重 | 1.88 MiB（18.75 GiB / (40 层 × 256)） |
| 主卡算 40% = 每层 102 个专家 | 192 MiB/层 |
| PCIe 实测有效带宽 | **0.85 GB/s**（4 MiB 下腿 4.892 ms，见 `plans/2026-10-23-q36-max-rows-cache.md`） |
| 搬 192 MiB 要 | **226 ms/层** |

整个 prompt 才 32 ms/层。**这条路比现状慢 7 倍。**

**但提议背后的直觉是对的**，而且有一个便宜得多的落点：

> **主卡不闲着，它的「专家算术」是 0，而它 VRAM 里白占 39 个专家的权重，
> 而 `stream_kick` 每层还在白拷 73 MiB。**
>
> **让 helper 不接主卡常驻的那 39 个专家，主卡就变成「算 39/256」**
> —— 不用新增数值路径（主卡还是 MMQ int8）、不用改 `peer_owner` 的语义（helper 只是少接一些）、
> 代价是 helper 少接 39/256 的量（15%）。

这是 `plans/2026-10-23-q36-cpu-share-with-peer.md` 的「替代方向」，
那里明确标注**未验证**。本轮把它做出来。

### 提议②：「decode 有没有充分利用 helper 的 DP4A 算力」

**helper 侧已经用满了它能用的**：P104 是 sm_61，跑的是真 `__dp4a`。
decode 的瓶颈**不在 helper 的算力**，在：

1. **主卡那 6.258 ms/token 的 host 同步**（32%）；
2. **主卡自己的 sm_60 回退**（比 helper 慢 2.2×）；
3. helper 的负载不均：CUDA3 的 `ms_wait` 0.751 是 CUDA1 的 0.233 的 **3.2 倍**。

## 4. 待做（按优先级）

- [ ] **提议①的落点**：helper 不接主卡常驻的 39 个专家 → 主卡算它们。实测 pp512 与 tg。
- [ ] **decode 的 6.258 ms 同步**：每层每 token 一次 `cudaStreamSynchronize(st)`，
      看能不能不每次都排空主卡的整条流（`moe_helpers` 的注释里说：改成只等那两条 D2H，
      实测 5.4 → 5.7 ms/token，**更慢**，所以被删了）—— 那条路已证实是死路，得换别的。
- [ ] helper 的负载不均（CUDA3 慢 3.2×）：`expert % 3` 的分法是否让某个 helper 拿到更多行。
