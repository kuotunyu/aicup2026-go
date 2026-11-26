# 11/04 開頭欄位（HD／CA／RE）稽核與演練

結論：**需要正規化**（PREREG_02 rev 11，檔案 `docs/PREREG_22_ROOT_PROPS.md`，工具 `tools/root_props.py`，操作頁 `docs/TESTDAY_22_ROOTPROPS.md`）。

## 官方格式

- v1.1 release notes：`(;SZ[19]HD[2];B[pd];…`；`sample_train_A.csv`：`(;SZ[19]HD[0];B[pd];W[pq];…`（20 盤：HD0 18、HD1 2，沒有 CA／RE）。
- 主 session 11/04 13:15 看真實測試檔（本 session 沒讀）：6 檔每盤 `(;SZ[19]HD[n];`，n 只有 0／1，沒有其他欄位。

## 靜態稽核（file:line）

| 路徑 | 判定 |
|---|---|
| `tools/p02_moves.py:44` MOVE_RE（開局、p02_player_open、p02_rank_opening、p04_opp、p06_style、t06／t10 經它） | 忽略 |
| `third_party/aicup2026_tutorial/utils.py:75,160`（sgfmill；R20／Q04／04 notebook） | 忽略（根節點沒有落子，HD 不擺子） |
| `tools/t02_checks.py:506-525` sgf_check（T1.4） | 走詞元解析，HD 只進 `other_props`，不算違規 |
| `t02_checks.py:450-464` T2 鍵、`t14_copy.py:204`、`t21_copy.py:204` | 忽略（只用落子） |
| `tools/trailing_pass.py:60-65` | 忽略（只看結尾） |
| `tools/t02_test_bundle.py` ingest／build | 逐位元組複製 |
| **`results/prereg07/explore/pilot_humansl/hs_lib.py:139,233-235` FAST_SGF／parse_moves**（07 HS、08／09／19 P1 經 `results/prereg08/p1/p1_extract.py:188-192`、競賽一 B′ 經 `t10_b.py`、Colab 備援 `colab_gpu/gpu_fallback_core.py`） | **壞掉**：每盤 HSError → 失敗率超過門檻，07／08／09／19／B′ 照登記「不採用」、悄悄退回保底 |

## 演練（`rehearsal_summary.json`、`rehearsal_h01.json`）

- 真實格式 H01：KataGo 解析 0／100,000（沒 HD 的 A：100,000／100,000）；正規化後 6 檔和 A 逐位元組相同，build 相同。
- 其他解析函式在沒正規化的 H 上和 A 差異 0；T1 除 `other_properties_record_only` 外相同。
- 和 rev 9 同時出現時順序和冪等都驗過（見 `docs/PREREG_22_ROOT_PROPS.md` 第 7 節）。
