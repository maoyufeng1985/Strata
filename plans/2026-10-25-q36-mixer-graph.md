# q36：decode 的墙钟模型，和 mixer 的 CUDA graph（2026-10-25）

状态：mixer graph 测完，结论是**没用**（本机），代码回退；墙钟模型留下，后面按它排序。
机器 192.168.1.98（P100-12G 主卡 + 3×P104-100 helper，CUDA only）。

## 1. 墙钟模型（MEASURED，2026-10-25，旗标开 `STRATA_Q36_HELPER_PARALLEL_MARSHAL=1`，tg128）

一 token 16.58 ms = 414 µs/层 × 40 层 + 1.34 ms（head/embedding/采样）。每层的 414 µs：

| 部分 | µs/层 | 是谁的时间 |
|---|---:|---|
| mixer 的 GPU 活（dense GEMV + GDN/注意力 + norm） | 135 | GPU（**主机那 55 µs 的派发藏在这里面**）|
| router | 20 | GPU |
| 派发段（helper 的 begin + 本地分组 + shared expert + CPU 池）| 133 | 主机（**GPU 在这段是闲的**）|
| 收 helper 的行（PCIe 往返 + 主机侧 scatter）| 58 | 主机（PCIe）|
| GPU 的其它活（本地分组、shared expert、merge）| 44 | GPU |
| head / embedding / 采样 | 27/token | 主机 |

**这条模型是本轮最有用的一条：主机的活只有在 GPU 闲着的时候才算墙钟。**
mixer 的派发（55 µs）落在 GPU 的 135 µs 里面 → 不要钱；派发段（133 µs）落在 GPU 的空窗里
→ 全价。marshal 旗标之所以 +10.4%，就是因为它砍的是派发段。

## 2. mixer 的 CUDA graph：测了，没用（回退）

做法：把 layer() 的 mixer 段（GDN/注意力分支、norm、router 的 GEMV、top-k）抽成方法，
启动时每层 capture 一个 graph，decode 时 `cudaGraphLaunch` 代替那 11 次 launch。
capture 的边界在两次 D2H 之前（pageable 的 D2H 不能进 capture）；位置是设备上的值
（pos0），所以 graph 每次 replay 读新的位置；注意力的 chunk 数在 capture 里是常数
n_chunk（越界的 chunk 立刻返回，归并核自己按位置算要归并几个）→ 结果与逐 token 变
chunk 数的那条路逐位相同。

**测到的（MEASURED，2026-10-25）：**
- 旗标开：15480 次 replay（40 层 × 387 token），**主机每次 graph launch 0.90 µs**。
- 文本：旗标开/关、与 control 三方**逐字节相同**。
- **速度：54.8 → 54.8 tok/s（单独开），60.3 → 60.4 tok/s（和 marshal 一起开）= 没有变化。**

**为什么没用（模型给出的）：** 主机省下的 55 µs 并没有变成墙钟——它变成了等 D2H 的时间
（sync 变长），因为那 55 µs 本来就藏在 GPU 的 135 µs 里面。graph 也不缩短 GPU 的
kernel-to-kernel 间隔（本机实测 graph 里每个节点约 2.4 µs，和 eager launch 的间隔一样；
.launch_overhead.cu 量的 0.77 µs/节点是空 kernel、没有依赖的情形）。

**本机的 graph launch 代价（.run/graph_launch_cost.cu，2026-10-25）：**

| graph 的节点数 | 主机：eager 每次 launch | 主机：graph 每次 launch | GPU：eager | GPU：graph |
|---:|---:|---:|---:|---:|
| 11 | 2.49 µs | 6.34 µs | 2.42 µs | 8.52 µs |
| 31 | 2.44 µs | 21.08 µs | 2.39 µs | 24.15 µs |
| 60 | 2.43 µs | 43.38 µs | 2.45 µs | 46.32 µs |

即 graph 的主机代价约 0.6–0.7 µs/节点（比 eager 便宜 4 倍），但**GPU 那侧没有便宜**：
真 kernel 的节点间隔和 eager 一样。

**结论：CUDA graph 在这台机器上换不到墙钟**（decode 的墙钟是 GPU 的时间线 + 主机在 GPU
空窗里的活）。代码回退，测量留档。

## 2b. shared expert 的 CUDA graph：也测了，也没有（保留代码，默认关）

做法：把 shared_expert() 抽成方法，启动时每层 capture 一个 graph（`STRATA_Q36_SHARED_GRAPH=1`），
decode 时 replay 代替那 4 次 launch。capture 在 `T == 1`、非 verify、非 rows 模式时成立。

**测到的（MEASURED，2026-10-25，旗标开 = marshal 也开，tg128）：**

| | 墙钟 tok/s | host 的 shared 段 | host 的 finish 段 |
|---|---:|---:|---:|
| 旗标关 | 61.2 / 61.4 / 61.2 | 0.908 ms/tok（22.7 µs/层）| 4.169 ms/tok |
| 旗标开 | 60.9 / 61.2 / 61.1 | **0.265 ms/tok（6.6 µs/层）** | **4.820 ms/tok** |

**host 的 shared 段确实掉了 16 µs/层（55 → 22.7 → 6.6），墙钟没动**：省下的 16 µs 原样
出现在 finish 段里（+0.65 ms/tok）。**⇒ 那 22.7 µs 根本不是 host 的 launch 开销，是 host
在等 GPU 算 shared expert。** graph 把 launch 换掉了，GPU 的活一点没少。

**这一条把墙钟模型改对了：host 各段的时间大多是 GPU 的活，不是 host 的活。**
所以再往前推的方向不是「把 host 的 launch 合并掉」，而是「少算」或者「少一个来回」。

代码保留（旗标 opt-in，默认关，默认路径逐字节不变），结论是没用。


## 3. 下一个：把 helper 的 begin 整段挪出主机线程

派发段（133 µs/层）是现在最贵的一段主机活。begin（60.6 µs）已经在 worker 上了，但主机
线程还在等它；等的时候主机本来可以派发本地分组 + shared expert + CPU 池（72.2 µs），
那段活跟 begin 没有数据依赖（只要 mask，而 mask 是路由回来就能算的）。

改法（`STRATA_Q36_HELPER_PARALLEL_MARSHAL=1` 的新语义）：**每个 helper 的 begin 都在
自己的 worker 上**，主机线程不 begin，它去派发本地那段活；收 helper 的行时先等 begin
完成再收。

**踩到的坑（已修）：** mask 原来是从 helper 的 `owned_` 数组算的（`owns(i)`），而
`owned_` 是 begin_slot 写的。begin 挪到 worker 之后，主机算 mask 就在和 worker 写
`owned_` 打架 → mask 漏掉 12.8% 的 entry → CPU 池替 helper 算了 3.5 s → decode 从
60 掉到 40 tok/s。**改成从 helper 的 cache 和路由算**（`holds(layer, expert)`，和
begin_slot 的规则一致：kind 和 primary_res 都是空的，cache 就是全部规则），在 begin
之前算完。
