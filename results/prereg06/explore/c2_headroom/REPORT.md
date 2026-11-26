# prereg06 探索：c2_headroom（2026-10-02）

> 這份由 workflow `wf_85ecc07b-a64` 的探索代理寫成。子代理不能直接寫 .md，主 session 照它回傳的結構化結果（同資料夾 `result.json`）原文排版存檔，沒有改內容。數字都是設計階段、只用訓練玩家（fit／inner），不是驗證集結果。

## 摘要

結論：(a) 不做。(b) 失分大多卡在「玩家層級」，(c) 便宜的 CPU 新資訊全部沒用。要再往上只剩兩條路：更強的逐盤證據，或測試日才看得到的「同一位玩家」連結。（只用 inner 單色偽題、沒打分、沒用 GPU。）
(a) 整份 400 題一起解碼（每級 40 題）：就算官方真的每級剛好 40 題，H40 也只多 +0.0012［−0.0047, +0.0083］，SK 多 +0.0024［−0.0029, +0.0097］。46% 的檔案反而變差。
(a) 等級不均衡時會倒扣：接近均衡 −0.002、照玩家人數比例 −0.035、隨機比例 −0.024～−0.052。EM 在均衡時也有 −0.026。
(a) 加上不看答案的閘門後，最壞情況收斂到平均約 −0.002，但好處也只剩 +0.001。判定：不值得做。
(b) 用一位玩家同一執色的全部棋（中位數 144 盤）來猜，也只到約 0.71。官方題型（5–20 盤）約 0.63。
(b) 官方題型的失分有 70–84% 是玩家層級（全部棋也猜錯，而且多半是穩定地差一級），只有 16–30% 是盤數不夠。
(b) 分數隨盤數成長：5→10→15→20 盤是 0.563→0.618→0.643→0.658，越後面加得越少。
(b) 每盤資訊變 2 倍 ≈ 官方題型 +0.040［+0.033, +0.047］，4 倍 ≈ +0.067［+0.052, +0.080］。這是更強逐盤模型的上限參考。
(b) 新模型的及格線：合進 Q02 後，5 盤單色偽題要從 0.563 升到約 0.618（現在 10 盤的水準），才算「2 倍」。
(b) 只看自己落子和只看對手落子的兩個來源，判斷玩家偏強或偏弱時約一半一致（去雜訊相關 0.50）。這一半很可能是標籤和實際棋力不一致，任何讀棋的模型都救不回。
(c) 結果都是 0 或負：T4 加進 Q02 權重在 5 折都是 0；棋局長度／最後一手／逐盤分散度 −0.0001～−0.0026；去頭尾平均 −0.0048。都不做。
附帶：均衡的 400 題檔案之間，Q02 分數本身就差 ±0.03（0.576–0.640），public 排行榜的雜訊和任何改進一樣大。
揭露：有一次遞迴 grep（只比對 *.py）可能走過 data/local 的資料夾名稱，約 2 分鐘後停掉，沒有輸出也沒有開任何 CSV。另外 harness 禁止子代理寫 REPORT.md，報告內容放在 verdicts，請主 session 存檔。

## 判定與證據

### (a) Transductive balanced decoding of a whole 400-question file (H40 Hungarian on the expected-score matrix, 40 per level)

- 判定：Do not use. The expected gain is about +0.001 even when files are perfectly balanced, and the losses on unbalanced files are 10 to 50 times larger.
- 證據：a_summary.json (1,000 simulated files per scenario plus a 200x10 player bootstrap; Q02 mono OOF). Gain over rule B on the official question mean: bal_cell +0.0012 [-0.0047,+0.0083], per-file p5 -0.0136 / p95 +0.0168, 46% of files worse; bal_level +0.0011; near_bal (80/level split across public+private) -0.0021 [-0.0100,+0.0051]; prop_players -0.0346 [-0.0485,-0.0215]; dir10 -0.0236; dir3 -0.0521 (per-file p5 -0.110). Q00/R20 bal_cell -0.0014. Reason: on balanced files Q02's rule-B histogram is already near uniform (33.7-43.6 per level, mean chi2 7.8 at df 9).

### (a) Other transductive variants: SK (Sinkhorn prior matching), H10k (10 per level per k), EM prior re-estimation (Saerens), ORC (true counts)

- 判定：None is usable. SK is the best on balanced files at +0.0024 but loses up to -0.066. H10k loses when only the level totals are balanced. EM loses -0.026 even on balanced files. ORC needs the true counts, which we do not have on test day.
- 證據：a_summary.json: SK bal_cell +0.0024 [-0.0029,+0.0097], prop -0.0402, dir3 -0.0659. H10k bal_cell +0.0018, bal_level -0.0140 [-0.0228,-0.0052]. EM bal_cell -0.0255 [-0.0479,-0.0139], dir3 +0.0090. ORC prop +0.0156 [+0.0081,+0.0246], dir3 +0.0316.

### (a) Label-free gate (chi2 of the rule-B histogram vs 40 each; EM likelihood ratio vs uniform; the model's own expected-score sacrifice of H40), thresholds at the 95th percentile of separately simulated balanced files

- 判定：The gate catches strong imbalance but cannot make the method worth it. Gated H40 averages +0.0009 on balanced files and about -0.0004 to -0.0019 otherwise; single files still go down to -0.03 to -0.05.
- 證據：a_summary.json. Thresholds (bal_cell p95): chi2_B 14.26, lr_em 14.67, sacrifice_H40 0.0079. sacrifice gate acceptance and mean gain: bal_cell 0.95 / +0.0011 (boot +0.0009 [-0.0047,+0.0071]); near_bal 0.71 / -0.0007; prop 0.05 / -0.0007; dir10 0.22 / -0.0019; dir3 0.06 / -0.0010; worst single file -0.031 to -0.048.

### (b) Player-level vs question-level share of Q02's loss on the official shape (mean over k=5/10/15/20)

- 判定：70-84% of the loss is player level: the answer is still wrong when every same-colour game of the player is used, and mostly it is adjacent-level and consistent. Only 16-30% comes from too few games.
- 證據：b_decompose.json. Virtual questions are rebuilt from per-game terms with each player's fold parameters; they reproduce mono_P_Q02 to a max difference of 3.2e-15. Panel A (735 player-colour pools with >=20 games): test-shape 0.620, whole pool (median 62 games) 0.682, so 84% is player level. Panel B (277 pools with >=80 games): 0.628 vs whole pool (median 144 games) 0.706, so 79% is player level; the curve-fit s_inf 0.741 gives 70%. At k=20, 80% of the loss comes from pools whose whole-pool answer is not exact. 47% of pools (rank-balanced) stay wrong with all their games; only 6% are 2 or more levels off. The between-pool variance share is 24% at k=5 and 55% at k=20.

### (b) Score growth with k (fixed panel)

- 判定：Gains shrink fast with more games. The current model plateaus at about 0.71 from 40 games on.
- 證據：b_decompose.json. Panel A: k1 0.405, 2 0.478, 3 0.520, 5 0.563, 7 0.597, 10 0.618, 15 0.643, 20 0.658. Panel B: 5 0.566, 10 0.622, 20 0.670, 40 0.697, 60 0.704, 80 0.709, whole pool 0.706. All-player mono by k (u1_analysis.json): 0.554 / 0.611 / 0.647 / 0.655.

### (b) Value of better per-game evidence (for example KataGo human SL), using the approximation that 2x information per game equals 2x games

- 判定：2x information would give the test shape about +0.040 and 4x about +0.067. To count as 2x, a new per-game source fused into Q02 must lift the 5-game inner single-colour score from 0.563 to about 0.618 (the single-game score from 0.405 to about 0.478). Temperature or sharpening is not new information.
- 證據：b_decompose.json panel B test_shape: f1 0.628 [0.598,0.656]; f2 (k=10/20/30/40) 0.668, gain +0.0404 [+0.0328,+0.0474]; f4 (k=20/40/60/80) 0.695, gain +0.0668 [+0.0523,+0.0800]. The curve fit gives f2 0.668 and f4 0.694. This assumes the player-level bias stays the same; k>20 uses Q02's temperature (n/12)^gamma extrapolated. KataGo's human SL net exists (https://github.com/lightvector/KataGo/releases/tag/v1.15.0); its information gain on this task was not measured.

### (b) Shared player-level bias between the target's own moves and the opponent's moves

- 判定：About half of a player's systematic stronger/weaker deviation shows in both the target's own moves and the opponent's moves. That shared half is most likely real strength differing from the label, so no game-reading model can remove it. The other half could partly be fixed by a better model.
- 證據：b_decompose.json shared_bias (508 players with >=20 inner games; within-level residual of the mean per-game expected level). Split-half reliabilities: C5 0.89, T4 0.92, O4 0.69, SWA 0.88, SWB 0.89. Disattenuated cross-half correlations: T4~O4 0.50 (the only fully independent pair: own vs opponent moves, same count tables); C5~O4 0.74; C5~SWA/SWB 0.98-0.99 (same board, not independent). Correlation with Q02's whole-pool signed error: C5 0.84, SWA 0.84, SWB 0.84, T4 0.67, O4 0.62.

### (c) T4 (own moves scored with the O4 count tables) as a 5th fusion term

- 判定：No information beyond Q02. The fitted weight is 0 (the lower bound) in all 5 folds.
- 證據：c_ideas.json: delta rb 0.0000 [0.0,0.0], w_T4 by fold [0,0,0,0,0]. A rerun of Q02 through u1_lib.run_gen reproduces mono_P_Q02 with max abs diff 0.0.

### (c) Game-length features (mean/sd of log moves, share where the target played the last move, share of games >=250 moves; the CNN only sees the last 250 board states) and per-game dispersion features (sd of the per-game expected level, mean entropy), stacked on log P_Q02 with player-fold OOF

- 判定：No gain. All variants are at or below the bias-only and random-noise controls.
- 證據：c_ideas.json, delta rank-balanced vs Q02 (0.6107): bias-only -0.0007 [-0.0033,+0.0019]; noise control -0.0004; length -0.0022 [-0.0065,+0.0017]; dispersion -0.0001 [-0.0038,+0.0036]; all -0.0026 [-0.0076,+0.0020]. Inner games: median 226 moves; 32.7% are >=250 moves; the target played the last move in 49.3%.

### (c) Robust pooling (drop the highest and lowest game per level column, same parameters)

- 判定：Worse. Do not use.
- 證據：c_ideas.json trim_minmax -0.0048 [-0.0081,-0.0011]; -0.0102 at k=5.

### Public leaderboard noise

- 判定：A single 400-question file moves Q02's official score by about ±0.03 from the draw alone. Public LB differences of that size say nothing about which method is better.
- 證據：a_summary.json bal_cell: rule-B question mean 0.6083, 95% range across simulated files 0.5755-0.6397. Estimate based on 623 inner players.

### Process note: data/local guard incident

- 判定：Disclosed. No validation content was read or printed.
- 證據：A recursive `grep -rn --include=*.py` started at the repo root (to find the feature extractor). It was stopped after under 2 minutes and printed nothing. It may have walked directory names under data/, including data/local; --include=*.py only opens .py files, and data/local holds CSVs. Every analysis script runs behind the u1_lib guard. Later searches were limited to tools/, colab/ and third_party/.

### REPORT.md

- 判定：Not written: the harness refuses .md report files from subagents. These verdicts and the JSON files hold the full content; the parent session can write REPORT.md from them.
- 證據：The Write tool error said: 'Subagents should return findings as text, not write report files.'

## 機會

### Stronger per-game evidence (for example KataGo human SL per-move likelihoods across ranks) fused into Q02（comp2，信心 low）

- 預期增益：Estimate only. About +0.040 on the official shape if per-game information doubles, about +0.067 if it quadruples. Ceiling with the current player-level bias is about 0.71 vs 0.63 now. Whether KataGo reaches 2x is unknown.
- 依據：b_decompose.json panel B virtual-k curve: f2 gain +0.0404 [+0.0328,+0.0474], f4 +0.0668 [+0.0523,+0.0800]. Pass mark for a pilot: the fused 5-game inner single-colour score must rise from 0.563 to about 0.618.
- 工作量：約 5 天；Colab 4 次
- 使用者要做：Decide whether an external pretrained model is allowed (read the rules). Allow the weight download (>50 MB). Run the Colab notebooks. Approve one of the 2 remaining validation scorings. Upload on test day.
- 主要風險：Rules on external models. Must finish with test-day tooling by 11/03, which adds a new GPU notebook to the test-day path. Up to about half of the player-level bias looks irreducible (label vs real strength), and a better model may reach well under 2x.

### Test-day player linking: pool evidence when the same player appears in several competition-2 questions, or in the competition-1 candidate pool (100 black games per candidate)（both，信心 low）

- 預期增益：Conditional and speculative. Up to about +0.07 on each correctly linked question (official shape 0.628 toward the whole-pool plateau 0.706). That is at most about +0.035 overall if all black questions link. Zero if there is no overlap.
- 依據：b_decompose.json: panel B test-shape 0.628 vs whole pool (median 144 games) 0.706. Linking would rely on the competition-1 opening-token retrieval (about 0.92 on black games per docs/DATA.md). Overlap is unknown until 11/04.
- 工作量：約 3 天；Colab 0 次
- 使用者要做：Decide whether competition-1 test files may be used in competition 2 (two teams, one author; rule against private sharing between teams). One extra upload if it triggers.
- 主要風險：Most likely no overlap, so the work earns nothing. Wrong links can hurt. Rules interpretation. Needs a gate and a dry run before 11/03, without touching the R20/04 paths.

### Transductive balanced decoding (H40/SK) with a label-free gate（comp2，信心 high）

- 預期增益：About +0.001 if the official files are exactly 40 per level. About -0.002 average with a gate when they are not; single files down to -0.03 to -0.05. Not recommended.
- 依據：a_summary.json, six scenarios with 1,000 files each plus a player bootstrap.
- 工作量：約 1 天；Colab 0 次
- 使用者要做：None. Not recommended.
- 主要風險：The level balance of the official files is unknown, and a mis-specified count costs far more than balance gains.

## 未解問題

- REPORT.md could not be written: the harness refuses .md report files from subagents. Should the parent session write results/prereg06/explore/c2_headroom/REPORT.md (zh-TW) from these verdicts and the JSON files?
- Process note: a recursive grep for *.py started at the repo root may have listed directory names under data/local. It printed nothing, opened no CSV, and was stopped within 2 minutes. Record it in the prereg06 log?
- Does KataGo human SL reach the 2x pass mark (5-game inner single-colour score fused into Q02 from 0.563 to about 0.618)? Only a Colab pilot on inner games can tell, and the external-model rule must be checked first.
- On 11/04: do competition-2 test players repeat across questions or files, or appear in the competition-1 candidate pools? Is using competition-1 test files in competition 2 acceptable under the rules?
- Are the official files level-balanced? (a) is dropped, so this matters only as a diagnostic: the chi2 of the Q02 rule-B histogram on public and private can be logged without changing any submission.

## 寫出的檔案

- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\c2_headroom\hc_lib.py`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\c2_headroom\a_transductive.py`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\c2_headroom\a_transductive_bal_cell.json`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\c2_headroom\a_transductive_bal_level.json`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\c2_headroom\a_transductive_near_bal.json`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\c2_headroom\a_transductive_prop_players.json`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\c2_headroom\a_transductive_dir10.json`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\c2_headroom\a_transductive_dir3.json`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\c2_headroom\a_transductive_*.log (6 files)`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\c2_headroom\a_summarize.py`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\c2_headroom\a_summary.json`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\c2_headroom\b_decompose.py`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\c2_headroom\b_decompose.json`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\c2_headroom\b_decompose.log`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\c2_headroom\c_ideas.py`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\c2_headroom\c_ideas.json`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\c2_headroom\c_ideas.log`
