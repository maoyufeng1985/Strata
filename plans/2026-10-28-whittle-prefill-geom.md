# Whittle-45B：把 prompt 路径的几何参数化（2026-10-28）

## 0. 现状（实测）

引擎已能**把模型完整加载起来**：`session is up (engine 0.1.47)`，18.16 GiB 权重、
PLE 表（5.6 GB Q4_K，39,040,000 行）也过了。第一个真正跑起来的会话被这句话挡住：

```
strata generate: prefill: geometry differs from the artifact's
```

`src/prefill/prefill.cpp:1388`：

```cpp
constexpr int64_t N = 2560, HC = 4, D = N * HC, LR = 320, K = 10, NE = 512;   // :134
...
if (g.n_embd != N || g.hc != HC || g.hc_lr != LR || g.n_expert < 1 || ss.k != K) {
    err = "prefill: geometry differs from the artifact's"; return false;
}
```

Whittle 的实际值：**N=2048, HC=4, LR=320, K=8（expert_used_count）, NE=256（expert_count）**。

## 1. 要改的常数与它们的含义

| 常数 | 现在 | Whittle | 是什么 |
|---|---:|---:|---|
| `N` | 2560 | **2048** | `embedding_length` |
| `HC` | 4 | 4 | `hyper_connection.count` |
| `LR` | 320 | 320 | `hyper_connection.low_rank` |
| `K` | 10 | **8** | `expert_used_count`（每 token 选几个专家）|
| `NE` | 512 | **256** | `expert_count`（每层几个专家）|
| `D` | N*HC | 8192 | HC 的宽度 |
| `640` | 640 | **512** | `expert_feed_forward_length` = n_ff |
| `1280` | 1280 | **1024** | 2×n_ff（gate+up 的行数）|

`HC`、`LR` 不变；`N`、`K`、`NE`、`640`、`1280` 要变。

## 2. 改法：constexpr → 从 kCompiledGeometry 取

`layout.hpp` 已经包含 `geometry.hpp`，`kCompiledGeometry` 可直接用。改法是把文件头的
constexpr 换成从编译期几何派生的 constexpr：

```cpp
// The compiled shape (include/strata/core/geometry.hpp): Flash-Next's by default, Whittle-45B's in this build.
constexpr int64_t N = strata::core::kCompiledGeometry.n_embd;          // 2048
constexpr int64_t HC = strata::core::kCompiledGeometry.hc;              // 4
constexpr int64_t D = N * HC;
constexpr int64_t LR = strata::core::kCompiledGeometry.hc_lr;           // 320
constexpr int64_t NE = strata::core::kCompiledGeometry.n_expert;        // 256
constexpr int64_t FF = strata::core::kCompiledGeometry.n_ff;            // 512
constexpr int64_t GU = 2 * FF;                                          // 1024: gate + up
constexpr int64_t K = strata::core::kCompiledGeometry.n_expert_used;    // 8
```

`K` 和 `NE` 特别说明：
- `K`（每 token 激活几个专家）是**运行期**的（`ss.k != K` 是检查），所以
  `compiled_geometry_mismatch` 不比较 `n_expert_used`（它只比 `n_expert`）。
  ⇒ 用 `kCompiledGeometry.n_expert_used` 会把运行期的值写死成 8，
  而 `ss.k` 可能被 `--spec` 改。**要小心**：这里应该放宽成 `ss.k < 0 || ss.k > NE` 的检查，
  而不是 `ss.k != K`。
- `NE`（每层几个专家）同理是运行期值（`g.n_expert`）。

## 3. `640` / `1280` 的替换

`prefill.cpp` 里的 `640` 一律是 n_ff，`1280` 一律是 2×n_ff。逐个换：

| 行 | 现在 | 换成 |
|---|---|---|
| 1207,1218 | `mmq::matrix_bytes(dt, N, 640)` | `mmq::matrix_bytes(dt, N, FF)` |
| 1232 | `mmq::overread_bytes(dt, 640)` | `mmq::overread_bytes(dt, FF)` |
| 1234 | `mmq::matrix_bytes(dt, N, 640)` | `mmq::matrix_bytes(dt, N, FF)` |
| 1236 | `mmq::overread_bytes(dt, 640)` | `mmq::overread_bytes(dt, FF)` |
| 1274 | `T*K*1280, T*K*640` | `T*K*GU, T*K*FF` |
| 1277,1279 | `640` | `FF` |
| 1289 | `T*K*640` | `T*K*FF` |
| 1290,1291 | `T*640` | `T*FF` |
| 1395 | `lay.fmt[l].n_ff != 640` | `!= FF`（并改错误文案） |
| 1624,1626 | `T*K*640`, `T*640` | `T*K*FF`, `T*FF` |
| 2102,2104 | `Gr, 640` | `Gr, FF` |
| 2128,2130 | `R, 640` | `R, FF` |
| 4063 | `mmq::matrix_bytes(mmq_dt, N, 640)` | `..., FF)` |
| 4279,4281,4282 | `nr, 640` | `nr, FF` |
| 4286,4364,4633 | `dn.w_cols = 640` | `= FF` |
| 4275,4354,4617 | `gu.w_rows = 1280` | `= GU` |
| 1640,2373 | `1280*2560`, `2560*640` | `GU*N`, `N*FF` |

**不动的**：
- `src/prefill/kernels.cu` / `moe_fused.cu` / `moe_f16_kq*.cu` 里的 640/1280/2560
  是 **Q2_0 canonical blob 的布局**，只有**非 native** pack 走；Whittle 是 native pack
  （`lay.native == true`），这条路走不到。
- `prefill.cpp:1344` 的 `constexpr int64_t T = 64, Nn = 64, K = 320;` 是另一个函数里的局部量，与几何无关。

## 4. 顺序

1. 改 constexpr 头部 + `ss.k != K` 的检查。
2. 换 640/1280。
3. 编译（编译器会指出漏掉的）。
4. 重跑加载测试。

## 5. 风险

- `K` 从 10 变 8：MMQ 的分组表、group bounds 的大小都跟着 K 变（`T*K`），是纯运行期尺寸，OK。
- `NE` 从 512 变 256：`kGrpEv = NE/16+2`、`kHostBounds` 变小，OK。
- `N` 从 2560 变 2048：**很多 kernel 的 tile 常量**（如 `moe_f16_kq.cu` 的 `GU_NS = 2560/64`）
  是 Flash-Next 专属，native pack 不走，但要确认 fused/MMQ 路径不引用它们。
