# q36：keep-resident 改默认开（decode 专家全放 helper 算，主卡 0 专家）

日期：2026-10-24。目标机：192.168.1.98（1× Tesla P100-12GB 主卡 + 3× P104-100 8GB helper，16 核）。
前置：编码修复（`53e05a83`）、keep-resident 实验（`272eff23`，`STRATA_Q36_PEER_KEEP_RESIDENT` opt-in，keep8 实测）。

## 0. 这一轮改什么

把 keep-resident 的默认从**关**改成**开**：

- **改法**：`open_helpers` 里的 `filter_resident`（`q36_engine.cu:6358`）无条件跑——
  每个 helper 的 `ranked` 过滤掉主卡 `slot_of ≥ 0` 的专家。
- **旗标 `STRATA_Q36_PEER_KEEP_RESIDENT` 退化成 opt-out**：默认开；设成 **`0`** 可退回
  keep-resident 前的形状（helper 接全部专家，主卡 0 专家但 39 个权重占 VRAM 白拷 D2D 73 MiB/层）。
  不设 = 开。只认精确的 `"0"`。
- **默认路径变字节**：这是有意的（用户要求「decode 阶段专家全放 helper 计算」）。
  对照基线 = **编码修复 + keep-resident 关**（`bin/strata-q36.own`，md5 `5bdddd29`），
  不是 keep-resident 前的旧形状。`bin/strata-q36`（`b7d53f977`，编码修复前）全程不动。

## 1. 为什么这样改是对的（基于 keep8 实测）

keep8 实测（开 vs 关）：

| | 关（编码修复基线） | 开 |
|---|---|---|
| prompt pp512 | 461.6 | 461.8（持平） |
| decode tg64 | 51.8 | **45.3（−12.5%）** |
| 主卡 VRAM 的 39 个专家权重 | 占着（2.86 GiB），`stream_kick` 每层白拷 73 MiB D2D | 主卡 0 专家，不占 |
| helper 专家数/层 | 3440/3400/3400 | 2920/2880/2880 |

**用户的意图是「decode 专家全放 helper」**。开 keep-resident 后：

- 主卡 VRAM 的 39 个专家**不再占显存**（`slot_of` 仍按 39 个放，但那 39 个的 `peer_owner = 0`，
  主卡 MMQ 在 prefill 算它们；decode 时主卡 local/CPU pool 不补它们——helper 没接、主卡没 slot 才算，
  所以 decode 这 39 个也归 helper 的行）。
- 主卡真正「空出来」的 2.86 GiB VRAM（39 专家 × 40 层）= 给 KV cache / 更大 context 用。
- **prompt 持平**（461.6 → 461.8）：主卡 sm_60 算这 39 个的时间 ≈ 省下的 helper_wait，净额不变。
- **decode 掉 12.5%**（51.8 → 45.3）：这是**代价**——把 57.2% 热点行从最快 helper 挪到最慢主卡。

**所以：「decode 专家全放 helper」的收益是主卡 VRAM 空出来 + 不白拷 73 MiB/层，代价是 decode 掉 12.5%。**
用户要的是前者，接受后者。默认开。

## 2. 改法（最小 diff）

`open_helpers`（`q36_engine.cu:6312`）现在的形状：

```cpp
const bool keep_resident_on = Engine::keep_resident() && E.main_hybrid;   // keep_resident() = 旗标开
...
for (Tier& t : tiers) filter_resident(t.ranked);   // 过滤主卡 slot_of≥0 的专家
```

改成：

```cpp
// keep-resident 是默认形状：helper 不接主卡常驻的专家（主卡 0 专家，VRAM 空出来）。
// 只认精确的 "0" 退回 keep-resident 前的形状（helper 接全部专家）。
static const bool keep_resident_off = [] {
    const char* v = std::getenv("STRATA_Q36_PEER_KEEP_RESIDENT");
    return v != nullptr && v[0] == '0';
}();
const bool keep_resident_on = !keep_resident_off && E.main_hybrid;
```

旗标语义翻转：
- 不设 / 设 `1` → **开**（默认：decode 专家全放 helper，主卡 0 专家）
- 设 `0` → 关（helper 接全部专家，主卡 0 专家但 39 个权重占 VRAM 白拷）

`filter_resident` 本身不动，还是 `if (!keep_resident_on) return;`。

## 3. 要核对的（AGENTS.md：改默认必须实测）

- [ ] 编译新二进制 `bin/strata-q36.krd`（对照 `bin/strata-q36.own` = `5bdddd29`，`bin/strata-q36` 不动）
- [ ] **默认开**（不设旗标）：prompt pp512 / tg64，对照 keep8 的 461.8 / 45.3
- [ ] **opt-out 关**（`STRATA_Q36_PEER_KEEP_RESIDENT=0`）：应回到 keep-resident 前的形状
      （helper 3440/3400/3400，prompt ≈ 461.6，decode ≈ 51.8）
- [ ] logits 逐字节对照 `bin/strata-q36.own`（编码修复基线）—— **预期不同**（数值路径变了，
      keep8 已证 max|diff|=0.76），记录差异
- [ ] IMA=0、nan=0、rc=0、Xid=0
- [ ] 主卡 VRAM 占用：开 vs 关的 `nvidia-smi` 对照（开应少 2.86 GiB）
- [ ] 若「默认开」在某配置下崩（如 `main_hybrid=false`、无 helper、非 hybrid 模型），
      确认 `keep_resident_on` 的 `E.main_hybrid` 条件守住了（关 = 走老路，不崩）

## 4. 不变量

- **旗标只改默认，不加新行为**：开 = keep8 已实测的形状；关 = 编码修复的形状（keep8 的 K1）。
- **数值路径**：每个 expert 还是同一个算子（helper 的 q8_1 / 主卡 MMQ int8），只是归属变了。
  行契约、`eparts` 行序、三个 witness 不动。
- **旧二进制**：`bin/strata-q36`（`b7d53f977`）对照不动；`bin/strata-q36.own`（`5bdddd29`）
  是编码修复基线，保留。
