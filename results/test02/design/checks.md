**T1–T6 test-day check spec: all six checks can run on 11/04 11:00 without answer files. The T3 null has been computed from training players only.**

Rules followed: only training players were read, through `c10data.load_rows(part="train")`, which is fit plus inner. No answer files were opened, `score.py` was not run, no GPU was used, and the repo is unchanged (`git status` clean). All scripts and outputs are in `C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go\943dc84e-3a07-4c6f-b22a-052eec94e7d1\scratchpad\testkit\C\`.

## Three points to know before registering

1. **T3 statistic.** I tested six statistics on training data and chose **S2**: the top-1 minus top-2 opening TF-IDF cosine between a candidate's 100-game profile and 5,767 fit reference players. S2 and S9 were best (true-positive rate 0.382 / 0.383 at the null's 99th percentile, raw orientation). S2 is simpler. This choice must be disclosed, and so must the fact that the inner holdout rates were seen for all six statistics before S2 was chosen.
2. **Gap in the registration.** T1's registered consequence is "only change the reader/writer code". That cannot fix games that do not start at move 1, setup stones, passes or handicap. In those cases the opening terms are wrong: the rank opening term and O_raw/O_canon for player ID. A rev-5 addendum should say what happens before 11/04. A possible rule: the rank side falls back to R10, refit with the frozen tune procedure; the player side keeps the per-pool self-retrieval and guard; T3 is marked "not evaluable".
3. **T3 power limit.** Complete overlap is always detected. If only about 20% or less of the candidates are training players, the rule will not detect it (see T3).

## Common setup

- **Game key** (T2): all moves from `;([BW])\[([a-s]{2})\]`, with no 60-move limit. The key is sha1[:12] of the smallest points string among the 8 D4 symmetries (`p02_moves.SYM_TABLE`). The symmetries tied on the first 12 moves are found first and only those are fully compared; the result equals the full 8-way minimum. The move-colour letters are left out, so the same game seen from the other side also matches.
- Three keys per game: full game, first 100 moves (K100), first 60 moves (K60).
- **Speed and order:** building the index from 793,863 train games took 151 s (14 s to load, 135 s to parse and hash). Test files hash at about 0.17 ms per game. Run order: T1, T2, T3, T4, T5, then T6 once predictions exist.

## T1: schema

Exact PASS criteria:

- **T1.1 Question files** (rank and player): the column set equals `{question_id, num_games, sgf_1..sgf_20, color_1..color_20}` (order ignored). Candidate files: the column set equals `{game_id, player_id, sgf_content, color}`.
- **T1.2 Counts and IDs:**
  - Each question file has exactly 400 rows.
  - `question_id` is unique across public and private. It starts with `pub_` in the public file and `priv_` in the private file.
  - Each candidate file has 40,000 rows and exactly 400 distinct `player_id`s, each with exactly 100 rows.
  - `player_id` starts with `pub_p_` / `priv_p_`; `game_id` is unique and starts with `pub_` / `priv_`.
  - The public and private pools share no `player_id` and no `game_id`.
- **T1.3 Slots:** `num_games` is an integer in 5..20. For i ≤ n, `sgf_i` is non-empty and `color_i` is B or W. For i > n, both are empty.
  - rev 8: for i > n, a colour letter B/W/b/w with an empty or whitespace-only `sgf_i` is normalizable (`clear_colour_after_num_games_if_sgf_empty`); `sgf_i` with content after n stays `FAIL_STRUCT`.
- **T1.4 Each SGF:**
  - Property names are a subset of {SZ, B, W}, SZ is 19, and there is at least one move.
  - The first move is B and moves strictly alternate.
  - No pass (`B[]` / `B[tt]`), no AB/AW/AE/PL/HA.
  - Training passes all of these (0 passes, 0 setup properties, all alternating).
- **T1.5 Starts at move 1.** The statistic is the share of a file's games whose first move falls in S99. S99 is the set of D4 classes of first moves covering 99% of training: `{cc,cd,ce,cf,dd,de,df,jj}`. PASS iff share ≥ 0.95.
  - Under the null, the minimum over 100 simulated files was 0.983 for candidate pools and 0.985 for question files.
  - If games start at move 5, 11 or 21, the share drops to 0.766, 0.470 or 0.278.
  - A start at move 2 is caught by T1.4 (first colour W). A start at move 3 cannot be detected (0.983).
- **Record only:** move-count min, p1, median and max. Training values are 100 / 107 / 228 / 445.
- **Consequence:** registered as "only change the reader/writer code". T1.4 and T1.5 failures are not covered (point 2 above).

## T2: exact duplicates

**Index.** Built from `load_rows(part="train")` only: 6,407 players, 793,863 games, 79.4% of all games, with no validation players.

- This is enough, because copies would be spread over all players, so 79.4% of them would still be found.
- A pass over the validation players that keeps counts only (12-byte hashes, no `player_id`, rank or `game_id`) is possible. It adds nothing to the decision and would need your explicit approval, so I don't recommend it.

**Null from training players only:**

| Comparison | Full game | First 100 moves | First 60 moves |
|---|---|---|---|
| 73,556 inner games vs 720,307 fit games (different people) | 0 | 0 | 0 |
| Within fit, full duplicates | 0 | – | – |
| Within fit, games sharing a key only with the same player | – | 15 | 156 |
| Within fit, games sharing a key with another player | – | – | 54 |

**"Copy" definition.** A test game is a copy if its K100 key matches, or its full key matches when it has fewer than 100 moves. For short games, an index at that length is built when needed. K60 matches are record-only.

**Threshold.** FAIL iff copies ≥ max(10, ceil(0.001 × games in that file)), applied to each unit below:

- **(a) Test vs training:** per file. This fires once about 0.16% of a file's games are copies. Also record how many candidates have a copied game and whether their copies all map to one training player; that directly shows the person is in training.
- **(b) Questions vs own-pool candidates:** full-key only, per pool. Same-player repeats of the first 100 moves are real style and are record-only.
- **(c) Across questions in one file:** full-key, count games. Record same colour (the same player appears in several questions) separately from opposite colour (the two target players played each other).
- **(d) Public vs private:** per competition, full-key.
- Comparing rank test files with player-ID test files is not registered. I recommend skipping it to keep the two teams separate.

**Consequence** (registered): open split v2 and write a new registration. Uploads of the frozen model are still allowed. No prediction may use the copies until the new registration exists.

## T3: are the candidates training players?

**Reference and profiles:**

- Reference R is all 5,767 fit players, each profiled over all their fit games. This matches the registered "computed with fit".
- Tokens are `p02_player_open.game_tokens` (both token sets, 10 moves). Weight is 1 + ln(count). IDF is ln(5768 / (1 + df)) + 1, with df counted over R. Rows are L2-normalised.
- The vocabulary comes from fit games. A test token not in it enters only the row norm, with idf = ln(5768) + 1.
- A candidate's profile is its 100 pool games.
- **S2(c)** is the top-1 minus top-2 cosine over the 5,767 references.

**Null:**

- 1,985 fit players with ≥100 games. Each uses the first 100 games in the order `sha256('c10-t3|cand|{pid}|{gid}')`, with its own column dropped (leave-one-out).
- Threshold is `np.quantile(null, 0.99, method="higher")`.

**Rule:** per pool and per orientation, count candidates with S2 > q99. "Overlap" iff that count is ≥ 40 in raw OR canon.

**Results:**

| | raw | canon |
|---|---|---|
| q99 | 0.1004267242871777 | 0.08085695648061325 |
| Vocabulary size | 146,421 | 144,097 |
| Null values above q99 | 19/1,985 | 19/1,985 |
| Inner holdout above q99 (226 players, never in R) | 1 (0.44%) | 3 (1.33%) |
| Highest null exceedance for a single level | 2.04% (1D) | 3.16% (2D) |
| Worst P(≥40 of 400) for a single-level pool | 3.5e-16 | 2.5e-10 |

- Null exceedance per candidate for raw OR canon: 1.31%.

**Positive control** (same player: 100 games vs his other games):

| Same player's games in the reference | raw TPR | canon TPR |
|---|---|---|
| 20 | 0.045 | 0.046 |
| 50 | 0.16 | 0.18 |
| 100 | 0.34 | 0.40 |
| all remaining | 0.380 | 0.430 |

- By number of other games (bins 1–20 / 21–50 / 51–100 / 101–200 / ≥201): raw 0.038 / 0.100 / 0.231 / 0.438 / 0.646; canon 0.033 / 0.086 / 0.263 / 0.503 / 0.736.
- A player's own reference is the top-1 match for 0.912 of positives at m = 100. This figure is from the first round, with train as the reference.

**Power of the pool rule** (5,000 simulated pools; an overlapping person is referenced with probability 5,767/8,007):

| Share of candidates who are training players | P(declare overlap), raw OR canon |
|---|---|
| 0 | 0.0 |
| 0.2 | 0.020 |
| 0.3 | 0.539 |
| 0.4 | 0.974 |
| 0.5 | 0.9998 |
| 1.0 | 1.0 |

- Under complete overlap, the mean share above q99 is 0.277 (raw) and 0.313 (canon).

**Statistic comparison** (reference = train, raw, TPR at m = all): S1 0.342, S2 0.382, S3 0.154, S6 0.316, S8 0.277, S9 0.383.

**Runtime:** per orientation, 41 s (raw) / 62 s (canon) to build tokens, 3 s for the reference, 21–23 s for the null. The full dry run took 594 s, with a peak of 2.7 GB. On test day, both orientations and both pools should take about 3 minutes.

**Rules for reading T3:**
- If T1.5 fails, T3 is "not evaluable", not "no overlap".
- If T2(a) fires, that directly proves the overlap.
- Consequence (registered): consider 乙 (retrain on all players) and write a new registration.

**Files to commit before 11/04:**
- `results/prereg02/testday/t3_null.json`:
  - Definitions, the fit-ID sha256, and the vocabulary sizes.
  - Per orientation: q99, `idf_sha256`, per-level exceedance, holdout result, TPR tables and power table.
  - The rule, and the T3q thresholds.
- `t3_null_{raw,canon}.npz`, about 270 KB each: `player_id`, `level`, `null_S2`, `holdout_S2`, `pos_all_S2`, `pos_all_m`, `idf`.
- On test day, rebuild R and assert that q99 and `idf_sha256` are identical before applying the rule.

**T3q** (record-only; rank questions and player questions with k-game profiles):
- q99(k) for k = 5..20: raw 0.070–0.077, canon 0.065–0.076.
- TPR rises from 0.061 / 0.048 at k = 5 to 0.214 / 0.189 at k = 20 (raw / canon).
- Under complete overlap the expected share above is only about 0.10, too weak for a decision.

## T4: colour and first move

- **Colour share:** training black share is 0.5006. Simulated candidate pools ranged 0.490–0.521; question files 0.472–0.517.
- **Consequence** (registered): |b_c − b_q| > 0.5 sends player ID to guard level 3 (CNN only), already in `p02_player_fuse` apply. Everything else in T4 is record-only.
- **First-move distribution:** TVD against training. Null maximum is 0.048 for candidate pools and 0.073 for question files.
- **Rotation flag:** share of first moves in the upper-right quarter < 0.60.
  - Training 0.8166; null 0.791–0.839 (candidate pools) and 0.749–0.806 (question files); randomly rotated games give 0.245 (TVD 0.576).
  - If rotated: the rank model R20 is unaffected, because C5 averages 8 symmetries and the opening n-gram is canonical. For player ID, the per-pool self-retrieval chooses the orientation, and T3 already runs both.

## T5: num_games

Record only. Per file: histogram over 5..20, mean, share below 10, and a chi-square test against a uniform 5..20 (df 15).
- Uniform reference: mean 12.5, share below 10 is 0.3125.
- Local rank validation: mean 12.06, 36.4% below 10 (because 311 questions were cut short).

## T6: one question per candidate?

- **Registered:** without an explicit official statement, this structure is not used. Using it would need a new registration (VALIDATION says so).
- **Descriptive only:**
  - Whether 400 questions = 400 candidates per pool.
  - T2(c) same-colour duplicates.
  - The number of distinct top-1 candidates in the submission vs the same count in each self-retrieval round of that pool. Self-retrieval is one-to-one by construction, so this needs a small added log.
- **Simulation** (fit pools, opening-only): one-to-one gives 348 / 354 / 369 distinct top-1; self-retrieval rounds give 347–367; drawing question owners with replacement (252–261 distinct owners) gives 244–254.

## Files

- Scripts: `cache.py`, `t2_null.py`, `t3.py`, `t3b.py`, `t3final.py`, `t4.py`, `t6.py`
- Results: `t2_null.json`, `t3_raw_canon.json`, `t3b_raw.json`, `t3final.json`, `t3_null_proposed_raw.npz`, `t3_null_proposed_canon.npz`, `t4.json`, `t6.json`
- `cache_train.pkl` (171 MB, training games only) can be deleted.