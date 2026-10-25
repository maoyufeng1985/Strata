# q36：peer prompt 路径的验收门（2026-10-25）

目的：把「这条路径能不能开、能不能进默认」变成一条能跑的、有数有据的门。

工具：`tools/qwen35/q36_peer_gate.py`（跟别的门一样走 serve 协议，一次装模型、逐条 prompt 走 GEN）。

## 1. 写门之前的三个事实（都已在 plans/ 里）

- peer 路径**只对宽于 128 token 的 prompt chunk 生效**（`peer_window_fits`：`n_ent > hybrid_t * top_k`，
  `q36_engine.cu:4453`）。**那 10 条 parity prompt 只有 20-40 token ⇒ 它们永远碰不到 peer 的算术**
  ⇒ 拿 parity prompt 做这道门会**空跑通过**。门自己拼宽 prompt（≥129 token，默认 256）。
- peer 路径与默认路径**逐字节相同是做不到的**：prompt 的专家由 helper 算（q8_1 + `native_expert_grouped`），
  默认路径由主卡算（MMQ int8）⇒ 两条是两种各自自洽的算术。仓库里早就写着
  「the token stream diverges accordingly」（`q36_engine.cu:4081-4092`）。
  （那条注释里的 RMS 差 1.5e-1~3.3e-1 是用**36 token** 的 prompt 量的——那时 peer 路径根本没参与，
  量的是 helper 与 CPU 池的差别，不是 peer 路径的。）
- 2026-10-20 已经定过：**helper 路径与主卡路径的差异记录在案，parity 门只跑默认路径**
  （`2026-10-20-q36-1p3h-optimal.md` §3.5）。

⇒ **门的判据不能是「与默认路径逐字节相同」**（那是「进默认」的条件，不是「这条路径是否可用」的条件）。
⇒ 门判的是**这条路径自己能承诺的东西**，并把**与默认路径的差距量出来**给「进默认」的判断用。

## 2. 门判什么

| # | 判据 | 判定 |
|---|---|---|
| C1 | **门没有空跑**：每一趟 peer 臂的 stderr 必须有 `serve prompt chunks`；默认臂必须**没有** | 不满足 ⇒ **FAIL**（不是 pass）|
| C2 | **确定性**：peer 臂跑两遍 ⇒ 逐 token 相同 | 不同 ⇒ FAIL |
| C3 | **默认路径没被动过**：默认臂与参考（`--golden`）逐 token 相同 | 不同 ⇒ FAIL |
| C4 | **没有崩**：没有 illegal memory access、没有 NaN、退出码 0 | 有 ⇒ FAIL |
| C5 | **速度地板**：peer 臂的 prompt tok/s ≥ 默认臂的 | 更慢 ⇒ FAIL |
| C6 | **差距（只报不判）**：每条 prompt 首个不同的 token、与默认路径相同的 token 数 | 报告；这是「进默认」要用的数 |

退出码：C1..C5 全过才 0。

## 3. 实测（MEASURED，2026-10-25，192.168.1.98）

机器：Tesla P100-12GB 主卡 + 3× P104-100 helper。模型：Qwen3.6-35B-A3B-UD-Q4_K_XL。CUDA。
二进制：`bin/strata-q36.flag1`（= HEAD，`8264242f` 之后）。
旗标：`STRATA_REMOTE_ZEROCOPY=0 STRATA_Q36_HELPER_ASYNC=1 STRATA_Q36_HELPER_PARALLEL=1 STRATA_Q8_SM60=1
STRATA_Q36_HELPER_PARALLEL_MARSHAL=1 STRATA_Q36_FIXED_SLOTS=1`，peer 臂再加
`STRATA_Q36_PREFILL_PEERS=1 STRATA_Q36_PREFILL_PEERS_HOST=1 STRATA_Q36_PEER_F16=1`。
命令（serve 协议）：

```
python3 src/tools/qwen35/q36_peer_gate.py --exe /home/Strata/bin/strata-q36.flag1 \
  --args '--model /home/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --max-context 2048 --no-warmup \
          --expert-cache-device1 auto --expert-cache-device2 auto --expert-cache-device3 auto' \
  --base-env '<上面那批>' --count 6 --n 64 --width 256 --out .../peer.json
```

prompt：parity prompt 的 ids 拼起来的 **6 条 256 token** 的 prompt（`parity_prompts.ids` 已在仓里）。

| # | 判据 | 实测 | 判定 |
|---|---|---|---|
| C1 | 门没有空跑 | 两趟 peer 臂都有 `3 helper(s) serve prompt chunks (8680 experts held by helpers, 40 layers served …)`；默认臂没有 | ✓ |
| C2 | 确定性 | 两趟 6 条 prompt 的 token **逐个相同** | ✓ |
| C3 | 默认路径没被动过 | 默认臂与上一趟的默认臂（`--golden`）逐 token 相同：引擎打印「the default path matches the reference on 6 prompts」 | ✓ |
| C4 | 没有崩 | 三趟 rc=0，无 illegal memory access，无 NaN | ✓ |
| C5 | 速度地板 | prompt 的 tok/s：peer **454.7** vs 默认 **77.9**（5.84×）| ✓ |
| C6 | 差距（只报不判） | 6 条里 **4 条逐 token 相同**；p01 在第 **20** 个 token 分叉（共 26 个），p05 在第 **29** 个（共 64 个）| 报告 |

**⇒ 门通过（C1..C5），差距报告如上。**

门跑了两趟（第二趟带 `--golden` 指第一趟的输出）：**两趟的每一个 token 都一样**，
prompt 的速度 454.7 / 455.0 tok/s（peer）、77.9 / 78.1 tok/s（默认）。

### 3.1 这些数怎么读

- **C5 的 454.7 / 77.9 tok/s 是门自己的量法**（serve 协议、256 token 的 prompt、装完模型后的第一个
  prompt，主卡在 PCIe 上搬专家），**不是** bench 的 pp512 那一对（默认 156.1 vs peer 493.6，
  `2026-10-25-q36-peer-prefill-today.md` §1）。两个都是真的，量的是不同的东西：门量的是**同一趟里
  两条路径的对比**，bench 量的是稳态的 prompt 吞吐。
- **C6 的深度由答案的长度决定，不是门决定的**：这些 prompt（parity prompt 拼起来的）模型答得短
  （1、26、9、10、1、64 个 token），所以「前 20 个 token 相同」这种话只在这么长的答案里有意义。
  真正的长文档 prompt 会有更长的共同前缀——**没量过**。
- **C6 与 plans/2026-10-25-q36-peer-prefill-today.md §3 的数不矛盾**：那里的 216 token prompt 是
  36 个 id 重复 6 遍（退化输入），32 行 logits 里 9 行 argmax 相同、第 0 行就不同；这里用真实的
  parity prompt，文字大多相同。**两条路径的差别有多大，取决于 prompt**。

## 4. 门怎么用

```
flock /data/gpu.lock -c 'python3 tools/qwen35/q36_peer_gate.py --exe <binary> --args "<engine args>" \
    --base-env "<两条路径共用的旗标>" --peer-env "STRATA_Q36_PREFILL_PEERS=1 ..." --out peer.json'
```

- `--golden`：默认路径的参考（一次门跑的输出里有 `default_tokens`；`q36_solo_gate.py` 的输出里是
  `tokens`）——不给出就只报一句 note。
- 输出的 JSON 里 `tokens` 是 **peer 臂**的 token ⇒ 两个构建可以用现成的 `q36_cmp_gate.py` 比。
- 门的进程是引擎的父进程，所以 GPU 锁要包住**整个 python 调用**（不能像 `parity.py` 那样把锁做成
  命令前缀）。

## 5. 还没做的

- 长文档 prompt 下的 C6（深度）没量（机器上没有那六个文档）。
- 「进默认」的判据（与默认路径逐字节相同）没有门：**它做不到**，要它成立得让 helper 用主卡的
  MMQ int8 算同样的东西，而那套 int8 权重在 8 GB 的 helper 上放不下。
