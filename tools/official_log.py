"""Official upload log: results/official_scores.json (spec 3.5 and 6; VALIDATION discipline 4). Stdlib only.

    PY tools/official_log.py prepare  --comp rank|player --file PATH --reason TEXT [--method TEXT]
                                      [--checks-missing-reason TEXT]
                                      [--not-04-reason TEXT]    (rank only; PREREG_04 5.3, below)
    PY tools/official_log.py uploaded --id N --ok-at ISO --uploaded-at ISO
    PY tools/official_log.py public   --id N --score FLOAT [--lb-rank INT] [--note TEXT]
    PY tools/official_log.py final    --comp rank|player --id N --basis TEXT
                                      [--not-04-reason TEXT]    (rank only; PREREG_04 5.3, below)
    PY tools/official_log.py status   [--now ISO]
    (every command also takes --log PATH, default results/official_scores.json; prepare takes
     --tests-dir, default data/test)

prepare   refuses unless the file is committed and clean, its name matches --comp
          (submission_rank.csv / submission_player.csv), tools/check_submission.py and this file are committed
          and clean, data/test matches data/test/MANIFEST.json, check_submission --official passes against
          data/test, and the file is bound to the committed tool reports next to it (PREREG_02 rev 5 9.12:
          "checks and seams passed"): rank apply_test_report.json (mode real, local seam passed,
          check_submission official without errors, outputs_sha256 of this file), player submit_report.json
          (mode real, check_submission official passed, output.sha256 of this file);
          <dir>/seam_val/seam_local.json passed in mode real; results/test02/checks/<comp>/t1.json is the full
          T1 (complete: T1.1-T1.5) on the raw files (stage 'raw', mode real) with verdict PASS or NORMALIZED
          (FAIL_OPENING: rank only with fallback_used, player as usual); T2-T5 are recorded in the committed
          results/test02/checks/<comp>/summary.json with their committed details files (docs/TESTDAY.md U1;
          any verdict, they never block the first upload; --checks-missing-reason TEXT uploads without them and
          records why). The report SHA-256s, the T1 verdict and the T2-T5 verdicts go into the entry. Rank's
          default method names R10 when the file was made with the fallback; a --method naming the other
          combination is refused. Appends an entry with status 'prepared' and
          prints the OK request for the user (competition, team, file, rows, sha256[:12]). The user's OK is
          per file and per upload; the upload instruction goes in its own message (U2).
uploaded  records the user's OK time and the upload time (ISO; no offset = Taiwan time +08:00) for a
          prepared entry whose file is still byte-identical; warns past 5 uploads per local date.
public    records the public score; investigate = true when it is below the official baseline
          (rank 0.4631, player 0.2845) or at least 0.05 below the previous upload of the same competition
          (U3: look for bugs only; never change the method because of the score).
final     records the final choice (U5) by its entry; the last upload must carry the same SHA-256.
status    per competition: last upload and whether it equals the final choice, uploads per local date
          (max 5), entries prepared but not uploaded, time left to 11/23 23:59 (final upload target) and
          11/24 23:59 (deadline).
Scores are quoted only from this file (U6).

PREREG_04 rev 3 (competition 2, test04). Added lines only (tests/test_p04_hygiene.py): every line of the tagged
r20-testday-pre04 version is still here, in order. A rank file in results/test04/rank/ is a 04 file (Q02 = R20 + O4
+ SWA + SWB); every other file is handled exactly as above.
prepare   (04) binds the file to the committed tools/t04_rank.py report next to it (t04_apply_report.json: real
          mode, combo Q02, the notebook fields all true, the local Q02 seam passed, check_submission official, this
          file's SHA-256), recomputes PREREG_04 5.2 from the report's numbers (dkappa > 0 for O4 / SWA / SWB on
          public and private; O4 replay failures <= 0.5 % of the slots of seam_val, public and private; seam_val
          byte-identical to the committed validation O4), re-hashes every committed input the report lists under
          inputs_bound, and needs the committed results/test04/opp/opp_report.json, sw/seam_check.json and
          rank/seam_val/seam_local.json, test02's real R20 run (not the R10 fallback), the full raw T1 (PASS or
          NORMALIZED), at most one counted rerun archived in results/test04/failed/, R20's first upload logged as
          uploaded, and the report committed by K6 (2026-11-16 23:59:59 +08:00). The method names Q02.
          An R20 / R10 file while a logged 04 entry meets PREREG_04 5.3: only with --not-04-reason "PREREG_04 rev N:
          ..." naming a revision (N >= 4) whose row in the committed docs/PREREG_04_OPPONENT.md was added after those
          entries were prepared (free text alone is refused).
final     PREREG_04 5.3: a 04 entry only while it still meets every condition; an R20 / R10 entry while some 04
          entry meets them only with the same --not-04-reason check; recorded in final_choice.rank.prereg04_5_3.
public    warns when a 04 score equals an R20 score exactly (both files are named submission_rank.csv).
status    adds the K6 countdown and, with 04 entries, their state and what PREREG_04 5.3 asks for.
A log without 04 entries gives the same output and JSON as before; status adds the K6 line only.

PREREG_06 rev 3 (competition 1, test06). Added lines only (tests/test_p06_hygiene.py: every line of tag
p04-testday-pre06 is still here, in order). A player file in results/test06/player/off/ or on/ is a 06 file (06-OFF =
Q04 + v2 style + CSLS; 06-ON = 06-OFF + the one-to-one switch); every other file is handled exactly as above.
prepare   (06) binds the file to the committed tools/t06_player.py report next to it (t06_submit_report.json: real
          mode, its variant and method, check_submission official, this file's SHA-256, the 06 seam and H6P recorded;
          06-ON: every PREREG_06 5.3 condition true), the committed 06 seam and both pool reports, test02's real Q04
          run (submit report, local seam, the full raw T1 with PASS or NORMALIZED), every committed input listed under
          inputs_bound, and the report committed by K6' (2026-11-21 23:59:59 +08:00). Q04's first upload must be
          logged as uploaded; 06-ON also needs this run's 06-OFF uploaded (it pairs with the latest one), the public
          scores of Q04 and that 06-OFF logged without a same-file warning between them, K7' (2026-11-22 23:59:59
          +08:00) not passed and no 06-ON uploaded before (PREREG_06 5.5 E1: one per season). The OK request says 06-OFF or 06-ON,
          the top-1 differences against Q04 and, for 06-ON, that its public pool is the forced probe. A Q04 file while
          a 06-OFF upload meets PREREG_06 5.4: only with --not-06-reason "PREREG_06 rev N: ..." (N >= 4, the row added
          after those entries were prepared).
public    the same-file warning (Q04, the paired 06-OFF, 06-ON: two equal public scores while the 06 report counts
          public top-1 differences > 0; a 06 upload whose bytes were uploaded before with another public score) or,
          without one and only for 06-ON or its paired 06-OFF, d = public(06-ON) - public(06-OFF) and the veto
          (printed only; the decision is tools/t06_player.py veto -> results/test06/player/veto.json).
final     PREREG_06 5.4: competition 1's final choice must be the designated upload (06-ON when its veto says d >= 0
          and it is still the last upload, else 06-OFF; Q04 when no 06-OFF upload holds; undecided, refused, while
          06-ON waits for its veto or the veto is paused before K7'); another choice only with the same
          --not-06-reason check; recorded in final_choice.player.prereg06_5_4.
status    with 06 entries only: their state, the veto, the designated upload, K6' / K7' and whether 06-OFF must be
          uploaded once more.
A log without 06 entries gives the same output, log bytes and returned values as tag p04-testday-pre06
(tests/test_t06_official_log.py TagVersusNew).

PREREG_07 rev 4 (competition 2, test07). Added lines only (tests/test_t07_hygiene.py: every line of the branch point
2d81c80 and of tag p04-testday-pre06 is still here, in order). A rank file in results/test07/rank/ is a 07 file
(Q02H = Q02 + KataGo HumanSL); it is recognised before the 04 rule, so it is never taken for R20.
prepare   (07) binds the file to the committed tools/t07_rank.py report next to it (t07_apply_report.json: real mode,
          exit 0, combo Q02H, check_submission official, this file's SHA-256) and recomputes PREREG_07 5.4 from its
          numbers: H7R scored with Q02H >= 0.57 and the reference re-read equal (rev 3; score_H7R.json re-read), the
          GPU seam (counts equal, per move <= 0.05, per game and profile <= 0.002, lp_HS <= 0.02), Q02 equal to 04's
          rank_test_probs.csv, the move cross-check 0, the local seam, HS failures <= 0.5 % per set, the direction
          check, T1 PASS / NORMALIZED, every bound input unchanged and the report committed by K6-07 (2026-11-19
          23:59:59 +08:00). R20's and 04's first uploads must be logged as uploaded, the 04 entry must be the one the
          07 report binds and still meet PREREG_04 5.3, and every HS chunk must be later than R20's first upload.
          A 04 / R20 / R10 file while a logged 07 entry meets 5.4: only with --not-07-reason "PREREG_07 rev N: ..."
          (N >= 5, the row added after those entries were prepared).
final     PREREG_07 5.4: a 07 entry only while it still meets every condition; another choice while some 07 entry
          meets them only with the same --not-07-reason check; recorded in final_choice.rank.prereg07_5_4. The 04
          rule skips a 07 entry (it is judged here, not as R20).
public    the same-file warning between 07, 04 and R20 (equal public scores while the 07 report counts public-question
          rule-B differences > 0).
status    with 07 entries only: their state, what PREREG_07 5.4 asks for and the K6-07 countdown.
A log without 07 entries gives the same output, log bytes and returned values as the branch point 2d81c80 and, with no
06 entry either, as tag p04-testday-pre06 (tests/test_t07_official_log.py TagVersusNew).

PREREG_08 rev 3 (competition 2, test08). Added lines only (tests/test_t08_hygiene.py: every line of tag
p07-testday-pre08 is still here, in order). A rank file in results/test08/rank/ is an 08 file (Q08 = Q02H + 4 x P1TO);
it is recognised before the 07 and 04 rules, so it is never taken for 07, 04 or R20.
prepare   (08) binds the file to the committed tools/t08_rank.py report next to it (t08_apply_report.json: real mode,
          exit 0, combo Q08, check_submission official, this file's SHA-256) and recomputes PREREG_08 5.2 from its
          numbers: H8R adopted (score_H8R.json re-read: lower bound > 0, reference re-read equal, Q08 >= 0.57), the
          GPU seam (counts equal, standardised features <= 0.05, lp_P1TO <= 0.04), Q02H equal to 07's
          rank_test_probs.csv, the move cross-check 0, the local seam, P1 failures <= 0.5 % per set, the direction
          check, T1 PASS / NORMALIZED, every bound input unchanged and the report committed by K6-08 (2026-11-19
          23:59:59 +08:00). R20's, 04's and 07's first uploads must be logged as uploaded, the 07 entry must be the
          one the 08 report binds and still meet PREREG_07 5.4, and every P1 chunk must be later than R20's first
          upload. A 07 / 04 / R20 / R10 file while a logged 08 entry meets 5.4: only with
          --not-08-reason "PREREG_08 rev N: ..." (N >= 4, the row added after those entries were prepared).
final     PREREG_08 5.4: an 08 entry only while it still meets every condition; another choice while some 08 entry
          meets them only with the same --not-08-reason check; recorded in final_choice.rank.prereg08_5_4. The 07
          and 04 rules skip an 08 entry (it is judged here).
public    the same-file warning between 08 and 07 / 04 / R20 (equal public scores while the 08 report counts
          public-question rule-B differences > 0).
status    with 08 entries only: their state, what PREREG_08 5.4 asks for and the K6-08 countdown.
A log without 08 entries gives the same output, log bytes and returned values as tag p07-testday-pre08
(tests/test_t08_official_log.py TagVersusNew).

PREREG_09 rev 3 (competition 2, test09). Added lines only (tests/test_t09_hygiene.py: every line of tag
p08-testday-pre09 is still here, in order). A rank file in results/test09/rank/ is a 09 file (Q09 = Q02H + 4 x P1TO_E,
ens3fam); it is recognised before the 08, 07 and 04 rules, so it is never taken for 08, 07, 04 or R20.
prepare   (09) binds the file to the committed tools/t09_rank.py report next to it (t09_apply_report.json: real mode,
          exit 0, combo Q09, check_submission official, this file's SHA-256) and recomputes PREREG_09 5.2 from its
          numbers: H8R and the 09 adoption (results/prereg09/rank/apply_report.json re-read: usable, finalized), the
          09 GPU seam (counts equal, standardised EX <= 0.05 over the 42 features rev 4 judges, lp_P1TO_E <= tol_E),
          Q08 recomputed equal to 08's
          rank_test_probs.csv, the local seam, P1 failures <= 0.5 % per set, the direction check, T1 PASS /
          NORMALIZED, every bound input unchanged and the report committed by K6-09 (2026-11-19 23:59:59 +08:00).
          R20's, 04's, 07's and 08's first uploads must be logged as uploaded, the 08 entry must be the one the 09
          report binds and still meet PREREG_08 5.4, and every P1 chunk must be later than R20's first upload. An
          08 / 07 / 04 / R20 / R10 file while a logged 09 entry meets 5.3: only with
          --not-09-reason "PREREG_09 rev N: ..." (N >= 5, the row added after those entries were prepared).
final     PREREG_09 5.3: a 09 entry only while it still meets every condition; another choice while some 09 entry
          meets them only with the same --not-09-reason check; recorded in final_choice.rank.prereg09_5_3. The 08,
          07 and 04 rules skip a 09 entry (it is judged here).
public    the same-file warning between 09 and 08 / 07 / 04 / R20 (equal public scores while the 09 report counts
          public-question rule-B differences > 0).
status    with 09 entries only: their state, what PREREG_09 5.3 asks for and the K6-09 countdown.
A log without 09 entries gives the same output, log bytes and returned values as tag p08-testday-pre09
(tests/test_t09_official_log.py TagVersusNew).

PREREG_10 rev 3 (competition 1, test10; part A). Added lines only (tests/test_t10_hygiene.py: every line of tag
p09-testday-pre10 is still here, in order). A player file in results/test10/player/ is a 10 file (06-OFF with the
all-black level-3 pools lifted to 06's level-1 computation, A; B, the P1 LDA term, is not adopted in this kit); it is
recognised before the 06 rule, so it is never taken for Q04.
prepare   (10) binds the file to the committed tools/t10_player.py report next to it (t10_submit_report.json: real
          mode, method, check_submission official, this file's SHA-256, at least one pool lifted (10A), B not used),
          the committed 10 seam (passed) and both 10 pool reports, the 06 seam (passed), test02's real Q04 run (submit
          report, local seam, the full raw T1 with PASS or NORMALIZED), every committed input listed under
          inputs_bound, the report committed by K6-10 (2026-11-21 23:59:59 +08:00) and K5-10 (the PREREG_10 test-day
          revision row committed by 2026-10-31 23:59:59 +08:00). Q04's first upload must be logged as uploaded; when
          this season has a 06-OFF, the 06-OFF the 10 report was compared with must be uploaded; when 06-ON was
          produced, 10 waits until the 06-ON is not the designated final upload (vetoed, wrong upload, check
          inconclusive, K7' passed, or 06-ON not eligible): a kept 06-ON means no 10 (PREREG_10 4.4). A Q04 / 06-OFF /
          06-ON file while a logged 10 upload is the designated final upload: only with --not-10-reason
          "PREREG_10 rev N: ..." (N >= 2, the row added after those entries were prepared).
final     PREREG_10 4.4 for competition 1 when the log holds 10 entries (it replaces the PREREG_06 5.4 rule then):
          06-ON when PREREG_06 5.3 / 5.4 designates it; else a 10 upload that meets every condition above; else as
          PREREG_06 5.4 (06-OFF -> Q04). Another choice only with the same --not-10-reason check; recorded in
          final_choice.player.prereg10_4_4.
public    the same-file warning between 10 and Q04 / 06-OFF (equal public scores while the 10 report counts public
          top-1 differences > 0), and the reminder that a drop of 0.05 or more is checked for bugs only.
status    with 10 entries only: their state, the designated upload and the K6-10 countdown.
A log without 10 entries gives the same output, log bytes and returned values as tag p09-testday-pre10
(tests/test_t10_official_log.py TagVersusNew).

PREREG_14 (competition 2, test14: the exact-copy layer). Added lines only (tests/test_t14_hygiene.py: every line of
main be7dc4d is still here, in order). A rank file in results/test14/rank/ is a 14 file: the designated competition-2
file (final_choice.rank, uploaded) with only the questions whose test games are exact same-side copies of one training
player's games set to that player's level (tools/t14_copy.py apply).
prepare   (14) binds the file to the committed t14_copy_report.json next to it (real mode, verdict used, check_submission
          official without errors, this file's SHA-256, the index pins equal on part 'all', at least one changed row,
          and the file differs from its base exactly in the rows the report lists), the full raw T1 (PASS or
          NORMALIZED), the committed index pins and docs/PREREG_14_RANK_EXACT_COPY.md, the report committed by K6-14
          (2026-11-23 18:00:00 +08:00). The report's base must be final_choice.rank, an uploaded non-14 entry, and the
          competition-2 final rules (PREREG_04 5.3, 07 5.4, 08 5.4, 09 5.3, with the reasons recorded in that final
          choice) must still designate it. The 04-09 prepare rules skip a 14 file. A non-14 rank file while a logged 14
          upload meets these conditions: only with --not-14-reason "PREREG_14 rev N: ..." (N >= 3).
final     a 14 entry: the 04-09 final rules are applied to its base (with the given --not-0x-reason), then the 14
          entry itself is checked; recorded in final_choice.rank.prereg14_7 (the 04-09 records describe the base).
          Another rank choice while a logged 14 upload meets the conditions: only with --not-14-reason.
public    the same-file warnings between 14 and its base; the 04-09 warnings skip 14 entries.
status    with 14 entries only: the 04-09 blocks judge a 14 final through its base; the 14 block and the K6-14 countdown.
A log without 14 entries gives the same output, log bytes and returned values as main be7dc4d
(tests/test_t14_official_log.py TagVersusNew).

PREREG_19 rev 3 (competition 2, test19). Added lines only (tests/test_t19_hygiene.py: every line of tag
p14-testday-pre19 is still here, in order). A rank file in results/test19/rank/ is a 19 file (Q19 = Q09 + 4 x SET, the
question-level set model S19); it is recognised before the 09, 08, 07 and 04 rules, so it is never taken for them.
prepare   (19) binds the file to the committed tools/t19_rank.py report next to it (t19_apply_report.json: real mode,
          exit 0, combo Q19, check_submission official, this file's SHA-256) and recomputes PREREG_19 4.2 from its
          numbers: H8R, the 09 and 19 adoptions (results/prereg19/rank/apply_report.json re-read: usable, finalized),
          Q09 recomputed equal to 09's rank_test_probs.csv, the local seam, the SET direction check, T1 PASS /
          NORMALIZED, every bound input unchanged and the report committed by K6-19 (2026-11-19 23:59:59 +08:00).
          R20's first upload and the 09 entry the 19 report binds must be logged as uploaded and that 09 must still
          meet PREREG_09 5.3. A 09 / 08 / 07 / 04 / R20 / R10 file while a logged 19 entry meets 4.3: only with
          --not-19-reason "PREREG_19 rev N: ..." (N >= 4, the row added after those entries were prepared).
final     PREREG_19 4.3: a 19 entry only while it still meets every condition; another choice while some 19 entry
          meets them only with the same --not-19-reason check; recorded in final_choice.rank.prereg19_4_3. The 09,
          08, 07 and 04 rules skip a 19 entry. 14 takes a 19 file as its base and applies 4.3 to it.
public    the same-file warning between 19 and 09 / 08 / 07 / 04 / R20.
status    with 19 entries only: their state, what PREREG_19 4.3 asks for and the K6-19 countdown.
A log without 19 entries gives the same output, log bytes and returned values as tag p14-testday-pre19
(tests/test_t19_official_log.py TagVersusNew).
PREREG_25 rev 3 (competition 2, test25). Added lines only (tests/test_t25_hygiene.py: every line of tag
p19-testday-pre25 is still here, in order). A rank file in results/test25/rank/ is a 25 file (Q25 = Q19 with the SET
term's set model S19 replaced by S25, retrained with the validation players); it is recognised before the 19, 09, 08,
07 and 04 rules, so it is never taken for them.
prepare   (25) binds the file to the committed tools/t25_rank.py report next to it (t25_apply_report.json: real mode,
          exit 0, combo Q25, check_submission official, this file's SHA-256) and recomputes PREREG_25 4.2 from its
          numbers: H8R, the 09, 19 and 25 adoptions (results/prereg25/rank/adoption.json re-read: usable, finalized),
          Q09 and Q19 recomputed equal to 09's and 19's rank_test_probs.csv, the SET25 direction check, T1 PASS /
          NORMALIZED, every bound input unchanged and the report committed by K6-25 (2026-11-18 23:59:59 +08:00).
          R20's first upload and the 19 entry the 25 report binds must be logged as uploaded and that 19 must still
          meet PREREG_19 4.3. A 19 / 09 / 08 / 07 / 04 / R20 / R10 file while a logged 25 entry meets 4.3: only with
          --not-25-reason "PREREG_25 rev N: ..." (N >= 4, the row added after those entries were prepared).
final     PREREG_25 4.3: a 25 entry only while it still meets every condition; another choice while some 25 entry
          meets them only with the same --not-25-reason check; recorded in final_choice.rank.prereg25_4_3. The 19,
          09, 08, 07 and 04 rules skip a 25 entry. 14 never takes a 25 file as its base (14 is not used).
public    the same-file warning between 25 and 19 / 09 / 08 / 07 / 04 / R20.
status    with 25 entries only: their state, what PREREG_25 4.3 asks for and the K6-25 countdown.
A log without 25 entries gives the same output, log bytes and returned values as tag p19-testday-pre25
(tests/test_t25_official_log.py TagVersusNew).
PREREG_21 (competition 1, test21: the exact-copy layer). Added lines only (tests/test_t21_hygiene.py: every line of
main c398d23 is still here, in order). A player file in results/test21/player/ is a 21 file: the designated
competition-1 file (final_choice.player, uploaded) with only the questions whose games are exact same-pool, same-colour
copies of one candidate's games rewritten to that candidate first, then the base row (tools/t21_copy.py apply).
prepare   (21) binds the file to the committed t21_copy_report.json next to it (real mode, verdict used, check_submission
          official without errors, this file's SHA-256, the four competition-1 test files it read equal to data/test,
          at least one changed row, every override 'the candidate first, then the base row', and the file differs from
          its base exactly in the rows the report lists), the full raw T1 (PASS or NORMALIZED), the committed passed
          simulation, K5-21 (the '| rev 2 |' row of docs/PREREG_21_PLAYER_EXACT_COPY.md on main's first-parent history
          by 2026-11-03 23:59:59 +08:00) and K6-21 (the report committed by 2026-11-23 18:00:00 +08:00). The report's
          base must be final_choice.player, an uploaded non-21 entry, and the competition-1 final rules (PREREG_06 5.4,
          PREREG_10 4.4, with the reasons recorded in that final choice) must still designate it on the log without
          21 entries. The 06 / 10 prepare rules skip a 21 file. A non-21 player file while a logged 21 upload meets
          these conditions: only with --not-21-reason "PREREG_21 rev N: ..." (N >= 3).
final     a 21 entry: the 06 / 10 final rules are applied to its base on the log without 21 entries (with the
          recorded --not-0x-reason), then the 21 entry itself is checked; recorded in final_choice.player.prereg21_7.
          Another player choice while a logged 21 upload meets the conditions: only with --not-21-reason.
public    the same-file warnings between 21 and its base; the 06 / 10 warnings never see a 21 entry.
status    with the final choice a 21 entry, the 06 / 10 blocks judge its base on the log without 21 entries; with 21
          entries, the 21 block and the K6-21 countdown.
A log without 21 entries gives the same output, log bytes and returned values as main c398d23
(tests/test_t21_official_log.py TagVersusNew).
"""

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import check_submission  # noqa: E402

FORMAT = "c10 official_scores v1"
TEAMS = {"rank": "TEAM_11248", "player": "TEAM_11249"}
COMP_LABEL = {"rank": "競賽二（棋力預測）", "player": "競賽一（玩家辨識）"}
BASELINES = {"rank": 0.4631, "player": 0.2845}
FILE_NAMES = {"rank": "submission_rank.csv", "player": "submission_player.csv"}
DEFAULT_METHOD = {"rank": "prereg02 R20 rule B (甲)",
                  "player": "prereg02 per-pool self-retrieval + guard (甲)"}
FALLBACK_METHOD = "prereg02 R10 fallback rule B (甲, T1 FAIL_OPENING)"   # rank file made with --fallback R10
RANK_COMBOS = ("R20", "R10")
OWN_TOOLS = ("tools/check_submission.py", "tools/official_log.py")      # prepare runs them: committed and clean
CHECKS_T2_T5 = ("T2", "T3", "T4", "T5")
QUESTION_FILES = {"rank": ("rank_prediction_test_public.csv", "rank_prediction_test_private.csv"),
                  "player": ("player_identification_test_public.csv", "player_identification_test_private.csv")}
CANDIDATE_FILES = ("player_identification_candidates_public.csv", "player_identification_candidates_private.csv")
TW = timezone(timedelta(hours=8))
FINAL_TARGET = datetime(2026, 11, 23, 23, 59, tzinfo=TW)
DEADLINE = datetime(2026, 11, 24, 23, 59, tzinfo=TW)
MAX_PER_DAY = 5
DROP_ALERT = 0.05
_NEEDLE = "ans" + "wers"
# ---- PREREG_04 rev 3 (test04, competition 2 only); added lines, R20's constants above are unchanged
T04_DIR = ("results", "test04", "rank")                   # only tools/t04_rank.py apply writes a 04 file here
T04_REPORT = "t04_apply_report.json"                      # never apply_test_report.json (that one is R20's)
T04_FORMAT = "c10-test04 rank apply v1"
T04_COMBO_ID = "Q02"
T04_COMBO_SOURCES = ("C5", "OP", "O4", "SWA", "SWB")
T04_METHOD = "prereg04 Q02 = R20 + O4 + SWA + SWB, rule B (甲)"
T04_OPPONENT_SOURCES = ("O4", "SWA", "SWB")               # PREREG_04 5.2: dkappa > 0 on each test set
T04_TEST_SETS = ("public", "private")
T04_REPLAY_SETS = ("seam_val", "public", "private")
T04_REPLAY_MAX = 0.005                                    # PREREG_04 5.2: more than 0.5 % of a set's slots -> no 04
T04_R20_FILE = "results/test02/rank/submission_rank.csv"
T04_OPP_REPORT = "results/test04/opp/opp_report.json"
T04_SEAM_CHECK = "results/test04/sw/seam_check.json"
T04_SEAM_LOCAL = "results/test04/rank/seam_val/seam_local.json"
T04_FAILED = "results/test04/failed"
T04_NOTEBOOK_TRUE = ("seam_check_passed", "same_as_repo_notebook", "test_bundle_equal_to_test02",
                     "local_recompute_passed", "env_same")
T04_RERUN_GPU = "NVIDIA RTX PRO 6000 Blackwell Server Edition"   # rev 3 items 4, 9: the registered rerun is on G4 only
T04_REDO_HINT = ("輸入已變更：這筆 04 不再符合 PREREG_04 5.3，最終 R20（docs/TESTDAY_04.md 第 7 步）；要重做 04 須先寫並 "
                 "commit PREREG_04 修訂")
T04_INPUTS_REQUIRED = (T04_OPP_REPORT, "results/test04/unpack_sw.json", "results/test04/sw/config.json",
                       T04_SEAM_CHECK, T04_SEAM_LOCAL, "results/test02/rank/apply_test_report.json",
                       "results/test02/rank/opening_test.npz", "results/test02/checks/rank/t1.json",
                       "results/prereg04/rank/frozen_config.json") + tuple(
    f"results/test02/rank/{s}/dumps/{n}.npz" for s in T04_REPLAY_SETS for n in ("val_keys", "val_p02e9_d4"))
P04_DOC = "docs/PREREG_04_OPPONENT.md"
P04_MIN_REV = 4                                           # rev 3 registers the test day; an override is rev 4 or later
P04_H4R = "adopted (results/prereg04/RESULTS.md, commit e8b24f2)"
K6 = datetime(2026, 11, 16, 23, 59, 59, tzinfo=TW)        # PREREG_04 4: a successful official 04 run committed by K6
# ---- PREREG_06 rev 3 (test06, competition 1 only); added lines, the constants above are unchanged
T06_DIR = ("results", "test06", "player")                 # only tools/t06_player.py submit writes a 06 file here
T06_VARIANTS = ("off", "on")
T06_REPORT = "t06_submit_report.json"                     # never submit_report.json (that one is test02's Q04)
T06_FORMAT = "c10-test06 submit v1"
T06_SEAM_FORMAT = "c10-test06 seam v1"
T06_POOL_FORMAT = "c10-test06 pool v1"
T06_VETO_FORMAT = "c10-test06 veto v1"
T06_METHOD = {"off": "prereg06 06-OFF: Q04 + v2 style + CSLS (甲)",
              "on": "prereg06 06-ON: 06-OFF + one-to-one switch (甲)"}
T06_LABEL = {"off": "06-OFF", "on": "06-ON"}
T06_Q04_FILE = "results/test02/player/submission_player.csv"
T06_SEAM = "results/test06/player/seam_val/t06_seam.json"
T06_POOL_REPORTS = ("results/test06/player/public/t06_pool.json", "results/test06/player/private/t06_pool.json")
T06_VETO = "results/test06/player/veto.json"
T06_T02_REPORTS = ("results/test02/player/submit_report.json", "results/test02/player/seam_val/seam_local.json",
                   "results/test02/checks/player/t1.json")
T06_INPUTS_REQUIRED = (T06_Q04_FILE, T06_SEAM, "results/prereg06/player/frozen_config.json",
                       "results/prereg06/player/score_H6P_switch.json") + T06_T02_REPORTS + T06_POOL_REPORTS + tuple(
    f"results/test02/player/{p}/dumps/{n}.npz" for p in ("seam_val", "public", "private")
    for n in ("cand_games", "query_games"))
T06_ON_CONDITIONS = ("private_switch_usable_and_detector_open", "h6p_secondary_not_negative", "pools_400_by_400",
                     "t2_c_zero_and_abd_not_failed", "both_pools_06_at_guard_level_1",
                     "aidea_page_not_against_one_to_one", "before_k7")
T06_SAME_SCORE = 1e-9                                     # PREREG_06 5.3 same-file warning: "完全相同"
T06_UPLOAD_CHECK_OFF = ("wrong_upload", "upload_check_inconclusive")   # rev 3 veto outcomes: no switch, final 06-OFF
T06_REDO_HINT = ("輸入已變更：這筆 06 不再符合 PREREG_06 5.4，最終照 U5 交 Q04；要重做 06 照 docs/TESTDAY_06.md 從 feats "
                 "重跑（PREREG_06 rev 3）")
P06_DOC = "docs/PREREG_06_PLAYER_STYLE.md"
P06_MIN_REV = 4                                           # rev 3 registers the test day; an override is rev 4 or later
P06_H6P = "adopted (results/prereg06/RESULTS.md, commit 5c1eea8)"
K6P = datetime(2026, 11, 21, 23, 59, 59, tzinfo=TW)       # PREREG_06 4: a successful official 06 run committed by K6'
K7P = datetime(2026, 11, 22, 23, 59, 59, tzinfo=TW)       # PREREG_06 4: 06-ON uploaded, both scores, veto decided

T07_DIR = ("results", "test07", "rank")                   # only tools/t07_rank.py apply writes a 07 file here
T07_REPORT = "t07_apply_report.json"
T07_FORMAT = "c10-test07 rank apply v1"
T07_COMBO_ID = "Q02H"
T07_COMBO_SOURCES = ("C5", "OP", "O4", "SWA", "SWB", "HS")
T07_METHOD = "prereg07 Q02H = Q02 + HumanSL, rule B (甲)"
T07_SETS = ("public", "private")
T07_HS_SETS = ("seam", "public", "private")
T07_GPU_SEAM = "results/test07/hs/gpu_seam.json"
T07_SEAM_LOCAL = "results/test07/rank/seam_val/seam_local.json"
T07_04_REPORT = "results/test04/rank/t04_apply_report.json"
T07_04_PROBS = "results/test04/rank/rank_test_probs.csv"
T07_H7R = "results/prereg07/rank/score_H7R.json"
T07_H7R_SHA = "32686f6c91fb0a14301c85ecc1c299844d9346fb2efb8f0fd0b5891f5609cbf5"     # LF; commit 31b26ca
T07_H7R_A = {"A_rank_balanced": 0.6236017652948923, "A_mean": 0.6245054776753989}  # prereg04 H4R, B row
T07_H7R_FLOOR = 0.57
T07_H7R_N = 1589
T07_TOL = {"move": 0.05, "profile_mean": 0.002, "lp_hs": 0.02}                    # PREREG_07 5.2 GPU seam
T07_RHO_MONO = 0.9502568925299116
T07_RHO_FLOOR = 0.155
T07_SEAM_B_MAX = 4
T07_INPUTS_REQUIRED = (T07_H7R, "results/prereg07/rank/frozen_config.json", "results/prereg07/rank/hs_map.npz",
                       "results/prereg07/hs/val_hs.npz", "results/prereg07/hs/val_seam_moves.npz",
                       "results/test07/hs/seam_hs.npz", "results/test07/hs/public_hs.npz",
                       "results/test07/hs/private_hs.npz", T07_GPU_SEAM, T07_SEAM_LOCAL, T07_04_REPORT, T07_04_PROBS,
                       "results/test02/rank/apply_test_report.json", "results/test02/checks/rank/t1.json")
T07_REDO_HINT = ("輸入已變更：這筆 07 不再符合 PREREG_07 5.4，最終照 PREREG_04 5.3（docs/TESTDAY_07.md 第 6 步）；要重做 07 "
                 "須先寫並 commit PREREG_07 的修訂（rev 5 起）")
P07_DOC = "docs/PREREG_07_HUMANSL.md"
P07_MIN_REV = 5                                           # rev 4 registers the test day; an override is rev 5 or later
P07_H7R = "rev 3: H7R scored, Q02H >= 0.57, reference re-read equal (results/prereg07/RESULTS.md, commit a7fb0ca)"
K6_07 = datetime(2026, 11, 19, 23, 59, 59, tzinfo=TW)     # PREREG_07 4: a successful official 07 run committed by K6-07

T08_DIR = ("results", "test08", "rank")                   # only tools/t08_rank.py apply writes an 08 file here
T08_REPORT = "t08_apply_report.json"
T08_FORMAT = "c10-test08 rank apply v1"
T08_COMBO_ID = "Q08"
T08_COMBO_SOURCES = ("C5", "OP", "O4", "SWA", "SWB", "HS", "P1TO")
T08_METHOD = "prereg08 Q08 = Q02H + 4 x P1TO, rule B (甲)"
T08_SETS = ("public", "private")
T08_P1_SETS = ("seam", "public", "private")
T08_GPU_SEAM = "results/test08/p1/gpu_seam.json"
T08_SEAM_LOCAL = "results/test08/rank/seam_val/seam_local.json"
T08_07_REPORT = "results/test07/rank/t07_apply_report.json"
T08_07_PROBS = "results/test07/rank/rank_test_probs.csv"
T08_H8R = "results/prereg08/rank/score_H8R.json"
T08_H8R_SHA = "8ecd2425fe3eef784e28d7587184aec67c0b41b49503e95b42f8532757209047"     # LF; commit 08b5e4a
T08_H8R_A = {"A_rank_balanced": 0.632702198843505, "A_mean": 0.6345453305055525}   # prereg07 H7R, B row
T08_H8R_FLOOR = 0.57
T08_H8R_N = 1589
T08_TOL = {"std_feature": 0.05, "lp_p1": 0.04}                                     # PREREG_08 5.2 GPU seam
T08_RHO_MONO = 0.9673017067776464
T08_RHO_FLOOR = 0.155
T08_SEAM_B_MAX = 4
T08_INPUTS_REQUIRED = (T08_H8R, "results/prereg08/rank/frozen_config.json", "results/prereg08/rank/p1_map.npz",
                       "results/prereg08/feat/val_seam_feats.npz", "results/prereg08/rank/val_lp_p1.npz",
                       "results/test08/p1/seam_p1.npz", "results/test08/p1/public_p1.npz",
                       "results/test08/p1/private_p1.npz", T08_GPU_SEAM, T08_SEAM_LOCAL, T08_07_REPORT, T08_07_PROBS,
                       "results/test07/rank/submission_rank.csv", "results/test04/rank/t04_apply_report.json",
                       "results/test02/rank/apply_test_report.json", "results/test02/checks/rank/t1.json")
T08_REDO_HINT = ("輸入已變更：這筆 08 不再符合 PREREG_08 5.4，最終照 PREREG_07 5.4（docs/TESTDAY_08.md 第 6 步）；要重做 08 "
                 "須先寫並 commit PREREG_08 的修訂（rev 4 起）")
P08_DOC = "docs/PREREG_08_STRENGTH_PROBE.md"
P08_MIN_REV = 4                                           # rev 3 registers the test day; an override is rev 4 or later
P08_H8R = "H8R adopted: Q08 - Q02H lower bound > 0 (results/prereg08/RESULTS.md, commit 08b5e4a)"
K6_08 = datetime(2026, 11, 19, 23, 59, 59, tzinfo=TW)     # PREREG_08 4: a successful official 08 run committed by K6-08

T09_DIR = ("results", "test09", "rank")                   # only tools/t09_rank.py apply writes a 09 file here
T09_REPORT = "t09_apply_report.json"
T09_FORMAT = "c10-test09 rank apply v1"
T09_COMBO_ID = "Q09"
T09_COMBO_SOURCES = ("C5", "OP", "O4", "SWA", "SWB", "HS", "P1TO_E")
T09_METHOD = "prereg09 Q09 = Q02H + 4 x P1TO_E (ens3fam), rule B (甲)"
T09_SETS = ("public", "private")
T09_P1_SETS = ("seam", "public", "private")
T09_GPU_SEAM = "results/test09/gpu_seam.json"
T09_SEAM_LOCAL = "results/test09/rank/seam_val/seam_local.json"
T09_08_REPORT = "results/test08/rank/t08_apply_report.json"
T09_08_PROBS = "results/test08/rank/rank_test_probs.csv"
T09_ADOPTION = "results/prereg09/rank/apply_report.json"
T09_ADOPTION_SHA = "691cd3a7c06e86947d96af1d2faf152c52006e336ebe15e81124ebeb3fd773b3"   # LF; finalized 9b4d1c4
T09_TOL = {"std_ex": 0.05, "lp_p1e": 0.04}                                          # PREREG_09 5.2 / rev 3 / rev 4 GPU seam
# std_ex: the report's max_abs_std_ex, the 42 EX features other than t_frac_lowp / o_frac_lowp (PREREG_09 rev 4)
T09_TOL_Q08 = 1e-12
T09_RHO_MONO = 0.9685160520465519
T09_RHO_FLOOR = 0.155
T09_SEAM_B_MAX = 4
T09_INPUTS_REQUIRED = (T09_ADOPTION, "results/prereg09/rank/frozen_config.json", "results/prereg09/rank/p1e_map.npz",
                       "results/prereg09/rank/val_lp_p1e.npz", "results/prereg09/rank/rank_val_probs.csv",
                       T09_GPU_SEAM, T09_SEAM_LOCAL, T09_08_REPORT, T09_08_PROBS,
                       "results/test08/rank/submission_rank.csv", "results/test08/rank/test_lp_p1.npz",
                       "results/test08/p1/p1_seam_report.json", "results/test08/p1/p1_public_report.json",
                       "results/test08/p1/p1_private_report.json", "results/test07/rank/t07_apply_report.json",
                       "results/test04/rank/t04_apply_report.json", "results/test02/rank/apply_test_report.json",
                       "results/test02/checks/rank/t1.json")
T09_REDO_HINT = ("輸入已變更：這筆 09 不再符合 PREREG_09 5.3，最終照 PREREG_08 5.4（docs/TESTDAY_09.md 第 5 步）；要重做 09 "
                 "須先寫並 commit PREREG_09 的修訂（rev 5 起）")
P09_DOC = "docs/PREREG_09_MAPPING_ENSEMBLE.md"
P09_MIN_REV = 5                                           # rev 3 / rev 4 register the test day; an override is rev 5 or later
P09_ADOPTED = "09 adopted on the inner gate, validation not scored (results/prereg09/RESULTS.md, finalize 9b4d1c4)"
K6_09 = datetime(2026, 11, 19, 23, 59, 59, tzinfo=TW)     # PREREG_09 4: a successful official 09 run committed by K6-09
# ---- PREREG_10 rev 3 (test10, competition 1 only, part A); added lines, the constants above are unchanged
T10_DIR = ("results", "test10", "player")                 # only tools/t10_player.py submit writes a 10 file here
T10_REPORT = "t10_submit_report.json"
T10_FORMAT = "c10-test10 submit v1"
T10_SEAM_FORMAT = "c10-test10 seam v1"
T10_POOL_FORMAT = "c10-test10 pool v1"
T10_METHOD = "prereg10 10: 06-OFF + all-black level-3 lift (A) [+ P1 LDA (B)] (甲)"
T10_SEAM = "results/test10/player/seam_val/t10_seam.json"
T10_POOL_REPORTS = ("results/test10/player/public/t10_pool.json", "results/test10/player/private/t10_pool.json")
T10_SOURCES = ("06-OFF", "10A", "test02")                 # 10B only once PREREG_10 B is adopted (not in this kit)
T10_T06_ON_REPORT = "results/test06/player/on/t06_submit_report.json"
T10_INPUTS_REQUIRED = (T06_Q04_FILE, T06_SEAM, T10_SEAM, "results/prereg02/player/frozen_config.json",
                       "results/prereg06/player/frozen_config.json")
T10_REDO_HINT = ("輸入已變更：這筆 10 不再符合 PREREG_10 4.4，最終照 PREREG_06 5.4；要重做 10 照 docs/TESTDAY_10.md 從 seam 重跑"
                 "（06 重做時 10 也從頭重做）")
P10_DOC = "docs/PREREG_10_PLAYER_GUARD_P1.md"
P10_MIN_REV = 2                                           # PREREG_10 4.4: an override cites rev 2 or later (in practice rev 4 or later: rev 2 / rev 3 predate the test day)
P10_TESTDAY_ROW = "| rev 3 |"                             # the test-day revision of PREREG_10 (K5-10)
K5_10 = datetime(2026, 10, 31, 23, 59, 59, tzinfo=TW)     # PREREG_10 3: the test-day extension (A) committed and merged
K6_10 = datetime(2026, 11, 21, 23, 59, 59, tzinfo=TW)     # PREREG_10 3: a successful official 10 run committed by K6-10
# ---- PREREG_14 (test14, competition 2): the exact-copy layer; added lines, the constants above are unchanged
T14_DIR = ("results", "test14", "rank")                   # only tools/t14_copy.py apply writes a 14 file here
T14_REPORT = "t14_copy_report.json"
T14_FORMAT = "c10-test14 copy apply v1"
T14_METHOD = "prereg14 exact-copy layer (PREREG_14 4) on the designated competition-2 file"
T14_PINS = "results/test14/design/index_pins.json"
T14_SIM = "results/test14/design/sim.json"
T14_PIN_FIELDS = ("rows", "players", "distinct_full_keys", "keys_with_two_or_more_rows",
                  "sha256_sorted_key_moves_colour_level_player")
K5_14 = datetime(2026, 11, 4, 11, 0, 0, tzinfo=TW)        # PREREG_14 8: pins, sim and the rev 2 row committed before
T14_BASE_DIRS = ("results/test02/rank", "results/test04/rank", "results/test07/rank", "results/test08/rank",
                 "results/test09/rank")
P14_DOC = "docs/PREREG_14_RANK_EXACT_COPY.md"
P14_MIN_REV = 3                                           # rev 1 registers, rev 2 is the implementation; overrides cite 3+
K6_14 = datetime(2026, 11, 23, 18, 0, 0, tzinfo=TW)       # PREREG_14 8: the 14 report's last commit
# ---- PREREG_19 rev 3 (test19, competition 2): Q19 = Q09 + 4 x SET; added lines, the constants above are unchanged
T19_DIR = ("results", "test19", "rank")                   # only tools/t19_rank.py apply writes a 19 file here
T19_REPORT = "t19_apply_report.json"
T19_FORMAT = "c10-test19 rank apply v1"
T19_COMBO_ID = "Q19"
T19_COMBO_SOURCES = ("C5", "OP", "O4", "SWA", "SWB", "HS", "P1TO_E", "SET")
T19_METHOD = "prereg19 Q19 = Q09 + 4 x SET (S19 set model), rule B (甲)"
T19_SETS = ("public", "private")
T19_P1_SETS = ("seam", "public", "private")
T19_SET_SEAM = "results/test19/set_seam.json"
T19_SEAM_LOCAL = "results/test19/rank/seam_val/seam_local.json"
T19_09_REPORT = "results/test09/rank/t09_apply_report.json"
T19_09_PROBS = "results/test09/rank/rank_test_probs.csv"
T19_ADOPTION = "results/prereg19/rank/apply_report.json"
T19_ADOPTION_SHA = "5f4eafa990dd88ae51aefd6355dc204c2bd4f17336e92b92a189c0ca97b25510"   # LF; finalize 1e19a0e
T19_TOL_Q09 = 1e-12
T19_RHO_MONO = 0.9585505165633248                         # results/prereg19/rank/frozen_config.json direction_ref
T19_RHO_FLOOR = 0.155
T19_SEAM_B_MAX = 4
T19_INPUTS_REQUIRED = (T19_ADOPTION, "results/prereg19/rank/frozen_config.json", "results/prereg19/rank/s19_seed0.npz",
                       "results/prereg19/rank/s19_seed1.npz", "results/prereg19/rank/s19_seed2.npz",
                       "results/prereg19/rank/val_set.npz", "results/prereg19/rank/rank_val_probs.csv",
                       "results/prereg19/rank/apply_audit.json", T19_SET_SEAM, T19_SEAM_LOCAL, T19_09_REPORT,
                       T19_09_PROBS, "results/test09/rank/submission_rank.csv", "results/test09/rank/test_lp_p1e.npz",
                       "results/test09/gpu_seam.json", "results/test09/rank/seam_val/seam_local.json",
                       "results/test08/rank/t08_apply_report.json", "results/test08/p1/p1_seam_report.json",
                       "results/test08/p1/p1_public_report.json", "results/test08/p1/p1_private_report.json",
                       "results/test07/rank/t07_apply_report.json", "results/test04/rank/t04_apply_report.json",
                       "results/test02/rank/apply_test_report.json", "results/test02/checks/rank/t1.json")
T19_REDO_HINT = ("輸入已變更：這筆 19 不再符合 PREREG_19 4.3，最終照 PREREG_09 5.3（docs/TESTDAY_19.md 第 5 步）；要重做 19 "
                 "須先寫並 commit PREREG_19 的修訂（rev 4 起）")
P19_DOC = "docs/PREREG_19_SET_MODEL.md"
P19_MIN_REV = 4                                           # rev 3 registers the test day; an override is rev 4 or later
P19_ADOPTED = "19 adopted on the inner gate, validation not scored (results/prereg19/rank/apply_report.json, finalized)"
K6_19 = datetime(2026, 11, 19, 23, 59, 59, tzinfo=TW)     # PREREG_19 5: a successful official 19 run committed by K6-19
T14_BASE_DIRS = T14_BASE_DIRS + ("results/test19/rank",)  # PREREG_19 4.1 / 4.3: a 19 file can be the base of 14
# ---- PREREG_25 rev 3 (test25, competition 2): Q25 = Q19 with S25; added lines, the constants above are unchanged
T25_DIR = ("results", "test25", "rank")                   # only tools/t25_rank.py apply writes a 25 file here
T25_REPORT = "t25_apply_report.json"
T25_FORMAT = "c10-test25 rank apply v1"
T25_COMBO_ID = "Q25"
T25_COMBO_SOURCES = ("C5", "OP", "O4", "SWA", "SWB", "HS", "P1TO_E", "SET")
T25_METHOD = "prereg25 Q25 = Q19 with S25 (set model + validation players), rule B (甲)"
T25_SETS = ("public", "private")
T25_P1_SETS = ("seam", "public", "private")
T25_SET_SEAM = "results/test25/set_seam.json"
T25_19_REPORT = "results/test19/rank/t19_apply_report.json"
T25_19_PROBS = "results/test19/rank/rank_test_probs.csv"
T25_09_PROBS = "results/test09/rank/rank_test_probs.csv"
T25_ADOPTION = "results/prereg25/rank/adoption.json"
T25_ADOPTION_SHA = "40b100357cc3ebb679b512a05c39fc0d8a067bb427e76128706cd72c8db31605"   # LF; finalize 639d9ec
T25_TOL = 1e-12
T25_RHO_MONO = 0.9576161336208181                         # results/prereg25/rank/frozen_config.json direction_ref
T25_RHO_FLOOR = 0.155
T25_INPUTS_REQUIRED = (T25_ADOPTION, "results/prereg25/rank/frozen_config.json", "results/prereg25/rank/freeze_report.json",
                       "results/prereg25/rank/freeze_audit.json", "results/prereg25/rank/s25_seed0.npz",
                       "results/prereg25/rank/s25_seed1.npz", "results/prereg25/rank/s25_seed2.npz", T25_SET_SEAM,
                       T25_19_REPORT, T25_19_PROBS, "results/test19/rank/submission_rank.csv",
                       "results/test19/rank/test_lp_set.npz", "results/test19/set_seam.json",
                       "results/test19/rank/seam_val/seam_local.json", "results/prereg19/rank/frozen_config.json",
                       "results/prereg19/rank/apply_report.json", "results/test09/rank/t09_apply_report.json",
                       T25_09_PROBS, "results/test09/rank/test_lp_p1e.npz", "results/test08/rank/t08_apply_report.json",
                       "results/test08/p1/p1_seam_report.json", "results/test08/p1/p1_public_report.json",
                       "results/test08/p1/p1_private_report.json", "results/test02/rank/apply_test_report.json",
                       "results/test02/checks/rank/t1.json")
T25_REDO_HINT = ("輸入已變更：這筆 25 不再符合 PREREG_25 4.3，最終照 PREREG_19 4.3（docs/TESTDAY_25.md 第 5 步）；要重做 25 "
                 "須先寫並 commit PREREG_25 的修訂（rev 4 起）")
P25_DOC = "docs/PREREG_25_SET_MORE_PLAYERS.md"
P25_MIN_REV = 4                                           # rev 3 registers the test day; an override is rev 4 or later
P25_ADOPTED = "25 adopted on the inner gate, validation not scored (results/prereg25/rank/adoption.json, finalized)"
K6_25 = datetime(2026, 11, 18, 23, 59, 59, tzinfo=TW)     # PREREG_25 5: a successful official 25 run committed by K6-25
# ---- PREREG_21 (test21, competition 1): the exact-copy layer; added lines, the constants above are unchanged
T21_DIR = ("results", "test21", "player")                 # only tools/t21_copy.py apply writes a 21 file here
T21_REPORT = "t21_copy_report.json"
T21_FORMAT = "c10-test21 copy apply v1"
T21_METHOD = "prereg21 exact-copy layer (PREREG_21 4) on the designated competition-1 file"
T21_SIM = "results/test21/design/sim.json"
T21_BASE_DIRS = ("results/test02/player", "results/test06/player/off", "results/test06/player/on",
                 "results/test10/player")
P21_DOC = "docs/PREREG_21_PLAYER_EXACT_COPY.md"
P21_MIN_REV = 3                                           # rev 1 registers, rev 2 is the implementation; overrides cite 3+
K5_21 = datetime(2026, 11, 3, 23, 59, 59, tzinfo=TW)      # PREREG_21 8: rev 2, tool, tests and sim merged into main by this
K6_21 = datetime(2026, 11, 23, 18, 0, 0, tzinfo=TW)       # PREREG_21 8: the 21 report's last commit


class LogError(SystemExit):
    def __init__(self, message):
        super().__init__(f"official_log: {message}")


# ---------------------------------------------------------------- helpers
def default_log():
    return os.path.join(REPO, "results", "official_scores.json")


def default_tests():
    return os.path.join(REPO, "data", "test")


def refuse_forbidden(path):
    if _NEEDLE in os.path.basename(str(path)).lower():
        raise LogError(f"refusing to open {path!r}")
    return path


def sha256_file(path):
    h = hashlib.sha256()
    with open(refuse_forbidden(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_time(text, what):
    """ISO time; no offset means Taiwan time (+08:00)."""
    try:
        t = datetime.fromisoformat(str(text).strip())
    except ValueError:
        raise LogError(f"{what} {text!r} is not an ISO time such as 2026-11-04T15:30 or 2026-11-04T15:30+08:00")
    return t.replace(tzinfo=TW) if t.tzinfo is None else t


def local_date(iso):
    return parse_time(iso, "time").astimezone(TW).date().isoformat()


def repo_rel(path):
    real, root = os.path.realpath(os.path.abspath(path)), os.path.realpath(REPO)
    if os.path.normcase(os.path.commonpath([real, root])) != os.path.normcase(root):
        raise LogError(f"{path} is outside the repository; upload only committed files from the repo")
    return os.path.relpath(real, root).replace(os.sep, "/")


def git(*args):
    try:
        r = subprocess.run(["git", "-C", REPO] + list(args), capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        return None, str(e)
    return r.returncode, r.stdout.strip()


def git_file_state(rel):
    """{'tracked', 'clean', 'head', 'file_commit'} for a repo-relative path."""
    rc_t, _ = git("ls-files", "--error-unmatch", "--", rel)
    rc_s, st = git("status", "--porcelain", "--", rel)
    rc_h, head = git("rev-parse", "HEAD")
    rc_f, fc = git("log", "-1", "--format=%H", "--", rel)
    return {"tracked": rc_t == 0, "clean": rc_s == 0 and not st, "head": head if rc_h == 0 else None,
            "file_commit": fc if rc_f == 0 and fc else None}


def count_rows(path):
    csv.field_size_limit(10**8)
    with open(refuse_forbidden(path), newline="", encoding="utf-8-sig") as f:
        return sum(1 for _ in csv.reader(f)) - 1


def empty_log():
    return {"format": FORMAT, "teams": dict(TEAMS), "baselines_public": dict(BASELINES),
            "final_choice": {"rank": None, "player": None}, "uploads": []}


def load_log(path):
    if not os.path.exists(path):
        return empty_log()
    with open(path, encoding="utf-8") as f:
        log = json.load(f)
    if log.get("format") != FORMAT:
        raise LogError(f"{path}: format {log.get('format')!r}, expected {FORMAT!r}")
    return log


def save_log(path, log):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(log, f, indent=1, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def entry(log, eid):
    for e in log["uploads"]:
        if e["id"] == eid:
            return e
    raise LogError(f"no entry with id {eid}")


def uploads_of(log, comp):
    return sorted((e for e in log["uploads"] if e["comp"] == comp and e["status"] == "uploaded"),
                  key=lambda e: (parse_time(e["uploaded_at"], "uploaded_at"), e["id"]))


def check_tests_dir(tests_dir, comp):
    """data/test files of this competition equal data/test/MANIFEST.json (the bytes Colab used)."""
    mpath = os.path.join(tests_dir, "MANIFEST.json")
    if not os.path.exists(mpath):
        raise LogError(f"{mpath} is missing")
    with open(mpath, encoding="utf-8") as f:
        man = json.load(f)
    if man.get("kind") != "official":
        raise LogError(f"{mpath} is kind {man.get('kind')!r}; uploads need the official test files (kind 'official')")
    files = man.get("files") or {}
    names = list(QUESTION_FILES[comp]) + (list(CANDIDATE_FILES) if comp == "player" else [])
    for n in names:
        p = os.path.join(tests_dir, n)
        if not os.path.exists(p) or sha256_file(p) != (files.get(n) or {}).get("sha256"):
            raise LogError(f"{p} is missing or differs from {mpath}")
    return names


REPORT_OF = {"rank": "apply_test_report.json", "player": "submit_report.json"}
T1_UPLOAD_OK = ("PASS", "NORMALIZED")


def _committed_json(path, what):
    if not os.path.exists(path):
        raise LogError(f"{what} {path} is missing; run the test-day tool first (docs/TESTDAY.md)")
    rel = repo_rel(path)
    st = git_file_state(rel)
    if not st["tracked"] or not st["clean"]:
        raise LogError(f"{rel} must be committed and unchanged before an upload (tracked={st['tracked']}, "
                       f"clean={st['clean']})")
    with open(refuse_forbidden(path), encoding="utf-8") as f:
        return rel, json.load(f)


def lf_hint(path, recorded, rel):
    """A hint when the file differs from the recorded SHA-256 only by CRLF line ends: with core.autocrlf=true (this
    machine; the repo has no .gitattributes) git writes CRLF whenever it checks the file out again, and git status
    still says clean (review finding 2026-09-26)."""
    try:
        with open(refuse_forbidden(path), "rb") as f:
            data = f.read()
    except OSError:
        return ""
    if b"\r\n" in data and hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest() == recorded:
        return (f"; the file now has CRLF line ends (a git checkout with core.autocrlf=true) and the recorded bytes "
                f"are LF: restore them with  git -c core.autocrlf=false checkout -- {rel}")
    return ""


def t1_problems(t1, trel):
    """The T1 behind an upload is the full T1 (T1.1-T1.5) on the raw files in real mode (review finding 2026-09-26:
    a structure-only t1.json, as the rehearsal commands write with --only, used to pass)."""
    probs = []
    if t1.get("complete") is not True:
        probs.append(f"{trel}: T1 ran only {t1.get('subchecks_run')} (complete={t1.get('complete')!r}); an upload "
                     f"needs the full T1.1-T1.5 on data/test/raw (PY tools/t02_checks.py t1 --comp "
                     f"{t1.get('comp') or '<comp>'} --rerun --reason TEXT, without --only)")
    if t1.get("stage") != "raw":
        probs.append(f"{trel}: stage {t1.get('stage')!r}; the upload is backed by T1 on the raw files (stage 'raw')")
    mode = (t1.get("provenance") or {}).get("mode")
    if mode != "real":
        probs.append(f"{trel}: written in mode {mode!r}, not 'real'")
    return probs


def check_t2_t5(comp, missing_reason):
    """docs/TESTDAY.md U1: T2-T5 ran and are recorded (committed summary.json and details files). Their verdicts
    never block the first upload (PREREG_02 rev 5 9.7-9.9); only their absence does, unless a reason is given
    (review finding 2026-09-26)."""
    cdir = os.path.join(REPO, "results", "test02", "checks", comp)
    spath = os.path.join(cdir, "summary.json")
    rec = {"recorded": {}, "missing": [], "missing_reason": None, "reports": {}}
    checks = {}
    if os.path.exists(spath):
        srel, s = _committed_json(spath, "the checks summary")
        rec["reports"][srel] = sha256_file(spath)
        checks = s.get("checks") or {}
    for c in CHECKS_T2_T5:
        ent = checks.get(c) or {}
        det = ent.get("details_file")
        dpath = os.path.join(cdir, det) if det else None
        if not dpath or not os.path.exists(dpath):
            rec["missing"].append(c)
            continue
        drel = repo_rel(dpath)
        st = git_file_state(drel)
        if not st["tracked"] or not st["clean"]:
            rec["missing"].append(f"{c} ({drel} not committed)")
            continue
        dsha = sha256_file(dpath)
        if ent.get("sha256") != dsha:
            rec["missing"].append(f"{c} ({drel} is not the file summary.json records)")
            continue
        rec["recorded"][c] = ent.get("verdict")
        rec["reports"][drel] = dsha
    if rec["missing"]:
        if not (missing_reason or "").strip():
            raise LogError(f"T2-T5 are not all recorded for {comp}: missing {rec['missing']}. Run PY tools/t02_checks.py "
                           f"run --comp {comp} and commit results/test02/checks/{comp}/ (docs/TESTDAY.md step 3). Their "
                           f"verdicts never block the first upload; to upload without them anyway (e.g. a bug in "
                           f"T2-T5), pass --checks-missing-reason TEXT, which is recorded")
        rec["missing_reason"] = missing_reason.strip()
    return rec


def method_for(comp, fallback_used, given):
    """The method recorded for the upload. Rank: R10 when the file was made with the fallback (review finding
    2026-09-26: the default used to say R20 for an R10 file); a --method naming the other combination is refused."""
    if comp != "rank":
        return given or DEFAULT_METHOD[comp]
    used = "R10" if fallback_used else "R20"
    other = [c for c in RANK_COMBOS if c != used]
    if given and any(re.search(r"(?<![A-Za-z0-9])%s(?![0-9])" % c, given) for c in other):
        raise LogError(f"--method {given!r} names {other[0]}, but the file was made with {used} "
                       f"(apply_test_report.json fallback_used={bool(fallback_used)})")
    return given or (FALLBACK_METHOD if fallback_used else DEFAULT_METHOD["rank"])


def check_reports(comp, path, sha):
    """PREREG_02 rev 5 9.12 ('checks and seams passed'): bind the upload file to the committed reports of the tool
    that wrote it (review finding 2026-09-26). Returns {'reports': {rel: sha256}, 't1_verdict': ...}."""
    if is_t21_file(comp, path):          # PREREG_21: a results/test21/player file, before the 10, 06 and Q04 rules
        return check_reports_t21(comp, path, sha)
    if is_t14_file(comp, path):          # PREREG_14: a results/test14/rank file, before every other rank rule
        return check_reports_t14(comp, path, sha)
    if is_t25_file(comp, path):          # PREREG_25 rev 3: a results/test25/rank file, before the 19-04 rules
        return check_reports_t25(comp, path, sha)
    if is_t19_file(comp, path):          # PREREG_19 rev 3: a results/test19/rank file, before the 09, 08, 07, 04 rules
        return check_reports_t19(comp, path, sha)
    if is_t09_file(comp, path):          # PREREG_09 rev 3: a results/test09/rank file, before the 08, 07 and 04 rules
        return check_reports_t09(comp, path, sha)
    if is_t08_file(comp, path):          # PREREG_08 rev 3: a results/test08/rank file, before the 07 and 04 rules
        return check_reports_t08(comp, path, sha)
    if is_t07_file(comp, path):          # PREREG_07 rev 4: a results/test07/rank file, before the 04 rule (not R20)
        return check_reports_t07(comp, path, sha)
    if is_t04_file(comp, path):          # PREREG_04 rev 3: a results/test04/rank file is bound to its own reports
        return check_reports_t04(comp, path, sha)
    if is_t10_file(comp, path):          # PREREG_10 rev 3: a results/test10/player file, before the 06 rule (not Q04)
        return check_reports_t10(comp, path, sha)
    if is_t06_file(comp, path):          # PREREG_06 rev 3: a results/test06/player/{off,on} file, its own reports
        return check_reports_t06(comp, path, sha)
    d = os.path.dirname(os.path.abspath(path))
    frel = repo_rel(path)
    out = {}
    rel, rep = _committed_json(os.path.join(d, REPORT_OF[comp]), "the tool report")
    out[rel] = sha256_file(os.path.join(d, REPORT_OF[comp]))
    problems = []
    if rep.get("mode") != "real":
        problems.append(f"{rel}: mode {rep.get('mode')!r} (uploads come from real-mode runs only)")
    if comp == "rank":
        if (rep.get("seam_local") or {}).get("passed") is not True:
            problems.append(f"{rel}: the local seam did not pass")
        cs = rep.get("check_submission") or {}
        if cs.get("mode") != "official" or cs.get("errors"):
            problems.append(f"{rel}: check_submission {cs.get('mode')!r} with errors {cs.get('errors')}")
        want = (rep.get("outputs_sha256") or {}).get(FILE_NAMES[comp])
        if want != sha:
            problems.append(f"{rel}: records another {FILE_NAMES[comp]} than this file ({sha[:12]})"
                            + lf_hint(path, want, frel))
    else:
        cs = rep.get("check_submission") or {}
        if cs.get("official") is not True or cs.get("passed") is not True:
            problems.append(f"{rel}: check_submission official={cs.get('official')!r} passed={cs.get('passed')!r}")
        want = (rep.get("output") or {}).get("sha256")
        if want != sha:
            problems.append(f"{rel}: records another {FILE_NAMES[comp]} than this file ({sha[:12]})"
                            + lf_hint(path, want, frel))
    seam_path = os.path.join(d, "seam_val", "seam_local.json")
    srel, seam = _committed_json(seam_path, "the local seam record")
    out[srel] = sha256_file(seam_path)
    if seam.get("passed") is not True or seam.get("mode") != "real":
        problems.append(f"{srel}: passed={seam.get('passed')!r}, mode={seam.get('mode')!r}")
    t1_path = os.path.join(REPO, "results", "test02", "checks", comp, "t1.json")
    trel, t1 = _committed_json(t1_path, "the T1 result")
    out[trel] = sha256_file(t1_path)
    problems += t1_problems(t1, trel)
    v = t1.get("verdict")
    if v == "FAIL_OPENING":
        if comp == "rank" and rep.get("fallback_used") is not True:
            problems.append(f"{trel}: FAIL_OPENING, but the rank file was not made with --fallback R10")
    elif v not in T1_UPLOAD_OK:
        problems.append(f"{trel}: verdict {v!r} (FAIL_STRUCT: no upload before a revision)")
    if problems:
        raise LogError("the file is not backed by passed checks and seams (PREREG_02 rev 5 9.12):\n - "
                       + "\n - ".join(problems))
    return {"reports": out, "t1_verdict": v, "fallback_used": rep.get("fallback_used") if comp == "rank" else None}


# ---------------------------------------------------------------- PREREG_04 rev 3: test04 (competition 2)
def t04_dir():
    return os.path.join(REPO, *T04_DIR)


def repo_path(rel):
    """Absolute path of a repo-relative 'a/b/c' path (computed at call time: tests mock REPO)."""
    return os.path.join(REPO, *str(rel).split("/"))


def is_t04_file(comp, path):
    """A competition-2 file in results/test04/rank/ is a 04 file (only tools/t04_rank.py apply writes there)."""
    if comp != "rank":
        return False
    here = os.path.realpath(os.path.dirname(os.path.abspath(path)))
    return os.path.normcase(here) == os.path.normcase(os.path.realpath(t04_dir()))


def git_commit_time(rel):
    """(commit, committer time ISO) of the last commit that touched rel; (None, None) when there is none."""
    rc, out = git("log", "-1", "--format=%H %cI", "--", rel)
    if rc != 0 or not out:
        return None, None
    h, t = out.split()
    return h, t


def git_head_text(rel):
    """rel as committed at HEAD, decoded as UTF-8 (git() decodes with the locale, which breaks on Chinese); None when
    it is not there."""
    try:
        r = subprocess.run(["git", "-C", REPO, "show", f"HEAD:{rel}"], capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.decode("utf-8", "replace") if r.returncode == 0 else None


def git_row_added(rel, row):
    """(commit, committer time ISO) of the first commit that added the text row to rel (git log --reverse -S);
    (None, None) when there is none."""
    rc, out = git("log", "--reverse", "-S" + row, "--format=%H %cI", "--", rel)
    if rc != 0 or not out:
        return None, None
    h, t = out.splitlines()[0].split()
    return h, t


def head_has_revision_row(n, head=None):
    head = git_head_text(P04_DOC) if head is None else head
    return bool(head) and any(line.startswith(f"| rev {n} |") for line in head.splitlines())


def p04_revision(reason, overridden, what):
    """PREREG_04 rev 3 item 20 (spec 3.0 'registration revision check', 9.3): overriding a registered 04 decision needs
    a reason naming 'PREREG_04 rev N' (N >= 4) while docs/PREREG_04_OPPONENT.md is committed and clean, HEAD has a
    row starting '| rev N |', and the commit that added that row is later than every overridden record (ISO times).
    Free text alone is always refused. -> the record kept with the decision."""
    text = (reason or "").strip()
    m = re.search(r"PREREG_04\s+rev\s+(\d+)", text)
    if not m:
        raise LogError(f"{what}: needs --not-04-reason \"PREREG_04 rev N: <the program error and the condition of "
                       f"PREREG_04 5.3 it breaks>\" citing a committed revision (N >= {P04_MIN_REV}); a reason "
                       f"without a revision is refused (PREREG_04 rev 3 item 20)")
    n = int(m.group(1))
    if n < P04_MIN_REV:
        raise LogError(f"{what}: --not-04-reason cites PREREG_04 rev {n}; an override needs rev {P04_MIN_REV} or "
                       f"later (PREREG_04 rev 3 item 20)")
    st = git_file_state(P04_DOC)
    if not st["tracked"] or not st["clean"]:
        raise LogError(f"{what}: {P04_DOC} must be committed and unchanged (tracked={st['tracked']}, "
                       f"clean={st['clean']})")
    row = f"| rev {n} |"
    if not head_has_revision_row(n):
        raise LogError(f"{what}: {P04_DOC} at HEAD has no row starting {row!r}; commit the revision first")
    h, t = git_row_added(P04_DOC, row)
    if h is None:
        raise LogError(f"{what}: no commit found that added {row!r} to {P04_DOC}")
    at = parse_time(t, "commit time")
    early = [w for w in overridden if at <= parse_time(w, "time")]
    if early:
        raise LogError(f"{what}: PREREG_04 rev {n} was committed at {t} ({h[:12]}), not after the records it "
                       f"overrides ({', '.join(early)}); a revision must come after the decision it overturns")
    return {"rev": n, "row": row, "doc": P04_DOC, "commit": h, "committed_at": t, "reason": text,
            "overridden_at": list(overridden)}


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def _is_count(v):
    return isinstance(v, int) and not isinstance(v, bool)


def sha256_bound(path):
    """The inputs_bound rule (spec 3.6): .csv and .json with CRLF folded to LF, anything else (.npz) raw bytes."""
    with open(refuse_forbidden(path), "rb") as f:
        data = f.read()
    if path.lower().endswith((".csv", ".json")):
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def t04_inputs_problems(bound):
    """Every input the apply report lists under inputs_bound is committed, clean and still has the recorded SHA-256
    (spec 9.2: R20 fixed after its first upload, e.g. test02's opening or apply re-run, invalidates the 04 run)."""
    if not isinstance(bound, dict) or not bound:
        return [f"the apply report lists no inputs_bound ({T04_REDO_HINT})"]
    probs = []
    missing = [r for r in T04_INPUTS_REQUIRED if r not in bound]
    if missing:
        probs.append(f"inputs_bound lacks {missing}")
    for rel_, want in sorted(bound.items()):
        s = str(rel_)
        if os.path.isabs(s) or "\\" in s or ".." in s.split("/") or s.startswith("/"):
            probs.append(f"inputs_bound key {s!r} is not a repo-relative path")
            continue
        p = repo_path(s)
        if not os.path.isfile(p):
            probs.append(f"{s} is missing ({T04_REDO_HINT})")
            continue
        st = git_file_state(s)
        if not st["tracked"] or not st["clean"]:
            probs.append(f"{s} is not committed and clean (先 commit；內容若已改變：{T04_REDO_HINT})")
            continue
        got = sha256_bound(p)
        if got != want:
            probs.append(f"{s} is {got[:12]}, the apply report bound {str(want)[:12]} ({T04_REDO_HINT})")
    return probs


def _bug_revision_number(v):
    """N of an archive's recorded bug revision (tools/t04_rank.py archive --bug-revision N); None when absent."""
    if isinstance(v, dict):
        for k in ("rev", "n", "number", "N", "revision"):
            if k in v:
                return _bug_revision_number(v[k])
        return None
    if _is_count(v):
        return v
    m = re.search(r"(\d+)", v) if isinstance(v, str) else None
    return int(m.group(1)) if m else None


def t04_attempt_times(att):
    """ISO times of an archived attempt (review round 2): its runs' starts, its _DONE_ markers' times and the times its
    bug revision record says it overturned (tools/t04_rank.py archive writes them)."""
    out = [r.get("start") for r in (att.get("runs") or []) if isinstance(r, dict) and r.get("start")]
    out += [t for t in (att.get("done_times") or {}).values() if t]
    br = att.get("bug_revision")
    if isinstance(br, dict):
        out += [o.get("time") for o in (br.get("overturns") or []) if isinstance(o, dict) and o.get("time")]
    return out


def t04_bug_revision_problems(where, n, att, head):
    """An archive's bug revision (rev 3 item 20, review round 2): N >= 4, HEAD's PREREG_04 has the row, and the commit
    that added the row is later than every recorded time of the archived attempt."""
    if n is None or n < P04_MIN_REV or not head_has_revision_row(n, head):
        return [f"{where}: bug revision {att.get('bug_revision')!r} is not a committed PREREG_04 revision "
                f"(rev {P04_MIN_REV} or later with its row at HEAD)"]
    row = f"| rev {n} |"
    h, t = git_row_added(P04_DOC, row)
    if h is None:
        return [f"{where}: no commit found that added {row!r} to {P04_DOC}"]
    times = t04_attempt_times(att)
    if not times:
        return [f"{where}: attempt.json records no time of the archived run; PREREG_04 rev {n} cannot be shown to come "
                f"after that failure"]
    try:
        at = parse_time(t, "commit time")
        early = [x for x in times if at <= parse_time(x, "time")]
    except LogError as e:
        return [f"{where}: {e}"]
    if early:
        return [f"{where}: PREREG_04 rev {n} was committed at {t} ({h[:12]}), not after the archived run "
                f"({', '.join(sorted(early)[-3:])}); a revision must come after the failure it exempts"]
    return []


def t04_archive_problems():
    """PREREG_04 rev 3 item 9 / 20: results/test04/failed/ holds at most one counted rerun, and an archive that holds a
    t04_apply_report.json (a registered decision) only together with a bug revision that HEAD's PREREG_04 has.
    Review round 2: an uncounted archive is before_r20_u1 or bug_revision; a bug revision's row was committed after
    the archived run; no two archives cite the same revision (one revision, one uncounted rerun)."""
    base = repo_path(T04_FAILED)
    if not os.path.isdir(base):
        return []
    probs, counted, head, cited = [], [], None, {}
    for name in sorted(os.listdir(base)):
        d = os.path.join(base, name)
        if not (name.startswith("attempt_") and os.path.isdir(d)):
            continue
        try:
            with open(os.path.join(d, "attempt.json"), encoding="utf-8") as f:
                att = json.load(f)
        except (OSError, ValueError):
            probs.append(f"{T04_FAILED}/{name}/attempt.json is missing or unreadable")
            continue
        if att.get("counts_toward_rerun") is not False:       # the safe side, as tools/t04_rank.py is_counted
            counted.append(name)
        elif att.get("failure_class") not in ("before_r20_u1", "bug_revision"):
            probs.append(f"{T04_FAILED}/{name}: counts_toward_rerun false with failure_class "
                         f"{att.get('failure_class')!r} (only before_r20_u1 and bug_revision are not counted)")
        n = _bug_revision_number(att.get("bug_revision"))
        if n is not None or att.get("failure_class") == "bug_revision":
            head = (git_head_text(P04_DOC) or "") if head is None else head
            probs += t04_bug_revision_problems(f"{T04_FAILED}/{name}", n, att, head)
            cited.setdefault(n, []).append(name)
        if os.path.exists(os.path.join(d, "rank", T04_REPORT)):
            head = (git_head_text(P04_DOC) or "") if head is None else head
            if n is None or n < P04_MIN_REV or not head_has_revision_row(n, head):
                probs.append(f"{T04_FAILED}/{name} archived an apply report (a registered decision) without a "
                             f"committed PREREG_04 revision (bug_revision {att.get('bug_revision')!r})")
    for n, names in sorted(cited.items(), key=lambda kv: str(kv[0])):
        if n is not None and len(names) > 1:
            probs.append(f"PREREG_04 rev {n} is cited by {len(names)} archives ({names}); one revision exempts one "
                         f"rerun only")
    if len(counted) >= 2:
        probs.append(f"{len(counted)} counted reruns archived ({counted}); PREREG_04 allows one rerun")
    return probs


def t04_conditions(rep):
    """PREREG_04 5.2 recomputed from the numbers of the apply report (never from its own verdicts).
    -> ({'dkappa': {'public.O4': ..}, 'replay_rate': {'seam_val': ..}}, problems)."""
    probs, dk, rates = [], {}, {}
    col = rep.get("colour_check") or {}
    for s in T04_TEST_SETS:
        for src in T04_OPPONENT_SOURCES:
            v = ((col.get(s) or {}).get(src) or {}).get("dkappa")
            dk[f"{s}.{src}"] = v
            if not _is_num(v) or not v > 0:
                probs.append(f"dkappa of {src} on {s} is {v!r}; PREREG_04 5.2 needs > 0 (04 is not used)")
    rp = rep.get("replay") or {}
    for s in T04_REPLAY_SETS:
        r = rp.get(s) or {}
        n, f = r.get("slots"), r.get("failures")
        if not (_is_count(n) and n > 0 and _is_count(f) and 0 <= f <= n):
            rates[s] = None
            probs.append(f"O4 replay of {s}: slots {n!r}, failures {f!r}")
            continue
        rates[s] = f / n
        if f * 200 > n:                                   # f / n > 0.005, in integers
            probs.append(f"O4 replay of {s}: {f} of {n} slots failed ({f / n:.2%} > {T04_REPLAY_MAX:.1%}; no 04)")
        elif s == "seam_val" and f:                       # spec 7: the committed validation O4 has 0 failures
            probs.append(f"O4 replay of seam_val: {f} of {n} slots failed; seam_val must equal the committed "
                         f"validation O4 (0 failures): local drift, not a 04 decision (spec 3.2 exit 4)")
    if (rp.get("seam_val") or {}).get("equal_to_committed_val_o4") is not True:
        probs.append("O4 of seam_val is not byte-identical to results/prereg04/opp/val_o4.npz (PREREG_04 5.2)")
    return {"dkappa": dk, "replay_rate": rates}, probs


def t04_report_problems(rep, rel, path, sha):
    """The apply report's own conditions (format, mode, combo, notebook fields, seams, check_submission, this file,
    PREREG_04 5.2 recomputed). -> (conditions, problems)."""
    probs = []
    if rep.get("format") != T04_FORMAT or rep.get("mode") != "real":
        probs.append(f"{rel}: format {rep.get('format')!r}, mode {rep.get('mode')!r} (a real-mode {T04_FORMAT})")
    combo = rep.get("combo") or {}
    if combo.get("id") != T04_COMBO_ID or tuple(combo.get("sources") or ()) != T04_COMBO_SOURCES:
        probs.append(f"{rel}: combination {combo.get('id')!r} {combo.get('sources')!r} is not the adopted Q02")
    nb = rep.get("notebook") or {}
    bad = [k for k in T04_NOTEBOOK_TRUE if nb.get(k) is not True]
    keq = nb.get("keys_equal_to_test02")
    if not isinstance(keq, dict) or any(keq.get(s) is not True for s in T04_REPLAY_SETS):
        bad.append(f"keys_equal_to_test02={keq!r}")
    if nb.get("earlier_failed_runs") != []:
        bad.append(f"earlier_failed_runs={nb.get('earlier_failed_runs')!r}")
    n_att = (rep.get("attempt") or {}).get("number")
    if _is_count(n_att) and n_att >= 2:                   # review round 1: the registered rerun ran on G4 only
        # review round 2: tools/t04_rank.py writes notebook.env flat ({gpu_name, torch, cudnn}: the one environment of
        # all 15 markers, env_same); a nested {marker: {gpu_name}} record is read too; no env is not shown to be G4
        env = nb.get("env") if isinstance(nb.get("env"), dict) else {}
        gpus = {nb.get("gpu")} | ({env.get("gpu_name")} if "gpu_name" in env else
                                  {(v or {}).get("gpu_name") for v in env.values() if isinstance(v, dict)})
        if not env:
            gpus.add(None)
        if gpus != {T04_RERUN_GPU}:
            bad.append(f"attempt {n_att} (the registered rerun) ran on {sorted(map(str, gpus))}, not only on G4 "
                       f"({T04_RERUN_GPU})")
    if bad:
        probs.append(f"{rel}: the test04 notebook record does not hold: {bad} (PREREG_04 rev 3 items 4, 5, 9)")
    if (rep.get("seam_local") or {}).get("passed") is not True:
        probs.append(f"{rel}: the local Q02 seam did not pass")
    cs = rep.get("check_submission") or {}
    if cs.get("mode") != "official" or cs.get("errors"):
        probs.append(f"{rel}: check_submission {cs.get('mode')!r} with errors {cs.get('errors')}")
    want = (rep.get("outputs_sha256") or {}).get(FILE_NAMES["rank"])
    if want != sha:
        try:
            hint = lf_hint(path, want, repo_rel(path)) if path and os.path.exists(path) else ""
        except LogError:                                  # a file outside the repository (a dry-run tree)
            hint = ""
        probs.append(f"{rel}: records another {FILE_NAMES['rank']} than this file ({str(sha)[:12]})" + hint)
    cond, cprobs = t04_conditions(rep)
    return cond, probs + cprobs


def rule_b_differences(path, other):
    """(questions of path whose level differs in other, questions of path); (None, None) when a file is missing."""
    if not (os.path.exists(path) and os.path.exists(other)):
        return None, None
    levels = []
    for p in (path, other):
        csv.field_size_limit(10**8)
        with open(refuse_forbidden(p), newline="", encoding="utf-8-sig") as f:
            r = csv.reader(f)
            next(r, None)
            levels.append({row[0]: row[1] for row in r if len(row) >= 2})
    a, b = levels
    return sum(1 for q, v in a.items() if b.get(q) != v), len(a)


def check_reports_t04(comp, path, sha):
    """PREREG_04 5.2 / 5.3 and rev 3 (spec 9.2): bind a 04 file to the committed t04_rank.py apply report next to
    it, its O4 replay, notebook seam and local seam records, the committed real R20 run of test02 (not the R10
    fallback), the full raw T1 (PASS or NORMALIZED), the committed inputs it lists, the archive history and K6.
    Returns check_reports' shape plus 'test04'."""
    out, problems = {}, []
    rpath = os.path.join(t04_dir(), T04_REPORT)
    rel, rep = _committed_json(rpath, "the test04 apply report (tools/t04_rank.py apply, docs/TESTDAY_04.md)")
    out[rel] = sha256_file(rpath)
    cond, probs = t04_report_problems(rep, rel, path, sha)
    problems += probs
    for srel, what, need in ((T04_SEAM_LOCAL, "the test04 local seam record", {"passed": True, "mode": "real"}),
                             (T04_OPP_REPORT, "the test04 O4 replay report",
                              {"passed": True, "usable_so_far": True, "mode": "real"}),
                             (T04_SEAM_CHECK, "the test04 notebook seam record", {"passed": True})):
        p = repo_path(srel)
        srel_, obj = _committed_json(p, what)
        out[srel_] = sha256_file(p)
        wrong = {k: obj.get(k) for k, w in need.items() if not (obj.get(k) is True if w is True else obj.get(k) == w)}
        if wrong:
            problems.append(f"{srel_}: {wrong}; needs {need}")
    t2 = os.path.join(REPO, "results", "test02", "rank", REPORT_OF["rank"])
    t2rel, t2rep = _committed_json(t2, "the test02 rank apply report (R20 goes first, PREREG_04 5.3)")
    out[t2rel] = sha256_file(t2)
    if t2rep.get("mode") != "real" or t2rep.get("fallback_used") is not False \
            or (t2rep.get("seam_local") or {}).get("passed") is not True:
        problems.append(f"{t2rel}: mode {t2rep.get('mode')!r}, fallback_used {t2rep.get('fallback_used')!r}, local "
                        f"seam {(t2rep.get('seam_local') or {}).get('passed')!r}: 04 needs test02's passed real-mode "
                        f"R20 run (FAIL_OPENING: 04 never plays, R10 is final)")
    t1_path = os.path.join(REPO, "results", "test02", "checks", comp, "t1.json")
    trel, t1 = _committed_json(t1_path, "the T1 result")
    out[trel] = sha256_file(t1_path)
    problems += t1_problems(t1, trel)
    v = t1.get("verdict")
    if v not in T1_UPLOAD_OK:
        problems.append(f"{trel}: verdict {v!r}; 04 plays only with T1 PASS or NORMALIZED (PREREG_04 5.3: "
                        f"FAIL_OPENING -> R10, FAIL_STRUCT -> stop)")
    problems += t04_inputs_problems(rep.get("inputs_bound"))
    problems += t04_archive_problems()
    h, t = git_commit_time(rel)
    k6_ok = t is not None and parse_time(t, "commit time") <= K6
    if not k6_ok:
        problems.append(f"{rel} was last committed at {t} ({h}), not by K6 {K6.isoformat()}: no 04 this season "
                        f"(PREREG_04 4); the final choice is R20")
    if problems:
        raise LogError("the file is not backed by a usable 04 run (PREREG_04 5.2, 5.3, rev 3):\n - "
                       + "\n - ".join(problems))
    n_diff, n_q = rule_b_differences(path, repo_path(T04_R20_FILE))
    rec = dict(cond, combo=T04_COMBO_ID, h4r=P04_H4R, report=rel, report_sha256=out[rel],
               k6={"commit": h, "committed_at": t, "deadline": K6.isoformat(), "ok": k6_ok},
               seam_local_tol=(rep.get("seam_local") or {}).get("tol"),
               notebook_gpu=(rep.get("notebook") or {}).get("gpu"),
               inputs_bound_checked=len(rep.get("inputs_bound") or {}),
               rule_b_differences_vs_r20=n_diff, questions_compared_with_r20=n_q,
               rule_b_differences_vs_r20_report=rep.get("rule_b_differences_vs_r20"))
    return {"reports": out, "t1_verdict": v, "fallback_used": False, "test04": rec}


def method_for_t04(given):
    """A 04 file is recorded as Q02, never with R20's default label (method_for has already refused a name of R10)."""
    if given and not re.search(r"(?<![A-Za-z0-9])%s(?![0-9])" % T04_COMBO_ID, given):
        raise LogError(f"--method {given!r} does not name {T04_COMBO_ID}; a results/test04/rank file is the 04 "
                       f"combination (default {T04_METHOD!r})")
    return given or T04_METHOD


def r20_first_upload(log):
    """PREREG_04 5.3: R20's first upload = the first uploaded competition-2 entry of test02's file (not a 04 one)."""
    ups = [x for x in uploads_of(log, "rank") if not x.get("test04") and x.get("file") == T04_R20_FILE]
    return ups[0] if ups else None


def print_t04_ok_lines(t4):
    n, q = t4.get("rule_b_differences_vs_r20"), t4.get("questions_compared_with_r20")
    print(f"- 這一份是 04，不是 11/04 那份 R20（{T04_R20_FILE}）；兩份在 {q} 題裡有 {n} 題的規則 B 等級不同")
    print(f"- 04 的方法：{T04_METHOD}；R20 的第一次上傳是 #{t4.get('r20_first_upload_id')}")
    print("- 04 的檢查：test04 notebook 和本機（Q02）的接縫都通過；重播失敗 "
          + "、".join(f"{s} {r:.2%}" for s, r in t4["replay_rate"].items())
          + f"（每組上限 {T04_REPLAY_MAX:.1%}）；Δκ 都 > 0："
          + "、".join(f"{k} {v:+.4f}" for k, v in t4["dkappa"].items()))
    print(f"- K6：04 的報告在 {t4['k6']['committed_at']} commit（期限 11/16 23:59）")


def t04_entries(log):
    return [x for x in log["uploads"] if x.get("comp") == "rank" and x.get("test04")]


def t04_entry_eligible(x):
    """PREREG_04 5.3 for one logged 04 entry, re-checked now (spec 9.3): H4R adopted (P04_H4R), the report's last
    commit is the one recorded at prepare and is by K6, T1 PASS / NORMALIZED, every dkappa > 0, replay <= 0.5 %,
    the report and the uploaded file unchanged, every bound input unchanged, the notebook record holds, and the
    archive history is within PREREG_04. -> (ok, reasons)."""
    t4, why = x.get("test04") or {}, []
    if x.get("t1_verdict") not in T1_UPLOAD_OK:
        why.append(f"T1 {x.get('t1_verdict')!r}")
    rel_ = t4.get("report") or ""
    rep = None
    rp = repo_path(rel_) if rel_ else ""
    if not rel_ or not os.path.isfile(rp) or sha256_file(rp) != t4.get("report_sha256"):
        why.append(f"{rel_ or 'the apply report'} changed since prepare ({T04_REDO_HINT})")
    else:
        with open(refuse_forbidden(rp), encoding="utf-8") as f:
            rep = json.load(f)
    fp = repo_path(x.get("file") or "") if x.get("file") else ""
    if not fp or not os.path.isfile(fp) or sha256_file(fp) != x.get("sha256"):
        why.append(f"{x.get('file')} changed since prepare")
    k6 = t4.get("k6") or {}
    h, t = git_commit_time(rel_) if rel_ else (None, None)
    if not k6.get("ok") or h is None or h != k6.get("commit") or parse_time(t, "commit time") > K6:
        why.append(f"K6: the report's last commit is {h} at {t}; prepare recorded {k6.get('commit')} at "
                   f"{k6.get('committed_at')} (deadline {K6.isoformat()})")
    dk = t4.get("dkappa") or {}
    if set(dk) != {f"{s}.{c}" for s in T04_TEST_SETS for c in T04_OPPONENT_SOURCES} \
            or not all(_is_num(v) and v > 0 for v in dk.values()):
        why.append(f"dkappa {dk} (each must be > 0)")
    rr = t4.get("replay_rate") or {}
    if set(rr) != set(T04_REPLAY_SETS) or not all(_is_num(v) and v <= T04_REPLAY_MAX for v in rr.values()):
        why.append(f"O4 replay failure rates {rr} (each <= {T04_REPLAY_MAX})")
    if rep is not None:
        why += t04_report_problems(rep, rel_, fp if fp and os.path.isfile(fp) else None, x.get("sha256"))[1]
        why += t04_inputs_problems(rep.get("inputs_bound"))
    why += t04_archive_problems()
    return not why, why


def t04_eligibility(log):
    """{entry id: (ok, reasons)} for every logged 04 entry."""
    return {x["id"]: t04_entry_eligible(x) for x in t04_entries(log)}


def prepare_rule_t04(log, comp, bound, not_04_reason):
    """PREREG_04 5.3 at prepare (spec 9.2): an R20 / R10 file while a logged 04 entry meets 5.3 needs the revision
    check of --not-04-reason. None when the log has no 04 entry: R20's prepare is exactly as before."""
    if bound.get("test14"):              # PREREG_14 7: a 14 file is judged on its base in prepare_order_t14
        if (not_04_reason or "").strip():
            raise LogError("--not-04-reason does not apply to a 14 file")
        return None
    reason = (not_04_reason or "").strip()
    if bound.get("test25"):              # PREREG_25 rev 3: a 25 file is judged by prepare_rule_t25
        if reason:
            raise LogError("--not-04-reason does not apply to a 25 file")
        return None
    if bound.get("test19"):              # PREREG_19 rev 3: a 19 file is not R20 / R10; judged by prepare_rule_t19
        if reason:
            raise LogError("--not-04-reason applies to an R20 / R10 file, not to a 19 file")
        return None
    if bound.get("test09"):              # PREREG_09 rev 3: a 09 file is not R20 / R10; judged by prepare_rule_t09
        if reason:
            raise LogError("--not-04-reason applies to an R20 / R10 file, not to a 09 file")
        return None
    if bound.get("test08"):              # PREREG_08 rev 3: an 08 file is not R20 / R10; judged by prepare_rule_t08
        if reason:
            raise LogError("--not-04-reason applies to an R20 / R10 file, not to an 08 file")
        return None
    if bound.get("test07"):              # PREREG_07 rev 4: a 07 file is not R20 / R10; judged by prepare_rule_t07
        if reason:
            raise LogError("--not-04-reason applies to an R20 / R10 file, not to a 07 file")
        return None
    if comp != "rank" or bound.get("test04"):
        if reason:
            raise LogError("--not-04-reason applies to a competition-2 R20 / R10 file only")
        return None
    elig = t04_eligibility(log)
    if not elig:
        return {"rule": "PREREG_04 5.3", "eligible_04_entries": [], "not_04_reason": reason} if reason else None
    ok_ids = [i for i, (ok, _w) in elig.items() if ok]
    rec = {"rule": "PREREG_04 5.3", "h4r": P04_H4R, "eligible_04_entries": ok_ids,
           "not_eligible_04_entries": {str(i): w for i, (ok, w) in elig.items() if not ok}, "checked_at": now_iso(),
           "not_04_reason": reason or None}
    if ok_ids:
        when = [x["prepared_at"] for x in t04_entries(log) if x["id"] in ok_ids]
        rec["revision"] = p04_revision(reason, when, f"04 entries {ok_ids} meet PREREG_04 5.3, so an R20 / R10 file "
                                                     f"is not the final choice")
    return rec


def final_rule_t04(log, e, not_04_reason):
    """PREREG_04 5.3 for competition 2 (spec 9.3). None when no 04 entry is logged (the R20-only log is handled
    exactly as before). A 04 entry is final only while t04_entry_eligible holds; while some 04 entry holds, an R20 /
    R10 final needs --not-04-reason citing a committed later revision (p04_revision), which is recorded."""
    reason = (not_04_reason or "").strip()
    if e.get("test25"):                  # PREREG_25 rev 3: a 25 final is judged by final_rule_t25
        if reason:
            raise LogError("--not-04-reason does not apply to a 25 entry")
        return None
    if e.get("test19"):                  # PREREG_19 rev 3: a 19 final is judged by final_rule_t19, never as R20
        if reason:
            raise LogError("--not-04-reason applies to an R20 / R10 final choice, not to a 19 entry")
        return None
    if e.get("test09"):                  # PREREG_09 rev 3: a 09 final is judged by final_rule_t09, never as R20
        if reason:
            raise LogError("--not-04-reason applies to an R20 / R10 final choice, not to a 09 entry")
        return None
    if e.get("test08"):                  # PREREG_08 rev 3: an 08 final is judged by final_rule_t08, never as R20
        if reason:
            raise LogError("--not-04-reason applies to an R20 / R10 final choice, not to an 08 entry")
        return None
    if e.get("test07"):                  # PREREG_07 rev 4: a 07 final is judged by final_rule_t07, never as R20
        if reason:
            raise LogError("--not-04-reason applies to an R20 / R10 final choice, not to a 07 entry")
        return None
    if e["comp"] != "rank":
        if reason:
            raise LogError("--not-04-reason applies to competition 2 (rank) only")
        return None
    elig = t04_eligibility(log)
    if not elig:
        return {"rule": "PREREG_04 5.3", "eligible_04_entries": [], "not_04_reason": reason} if reason else None
    ok_ids = [i for i, (ok, _w) in elig.items() if ok]
    rec = {"rule": "PREREG_04 5.3", "h4r": P04_H4R, "eligible_04_entries": ok_ids,
           "not_eligible_04_entries": {str(i): w for i, (ok, w) in elig.items() if not ok}, "checked_at": now_iso()}
    if e.get("test04"):
        ok, why = elig[e["id"]]
        if not ok:
            raise LogError(f"entry #{e['id']} (04) does not meet PREREG_04 5.3 now ({'; '.join(why)}): the final "
                           f"choice is R20 (R10 when T1 is FAIL_OPENING); upload it again so that it is the last "
                           f"upload (docs/TESTDAY_04.md step 7)")
        return dict(rec, chosen="04", not_04_reason=None)
    rec.update(chosen="R10" if e.get("fallback_used") else "R20", not_04_reason=reason or None)
    if ok_ids:
        when = [x["prepared_at"] for x in t04_entries(log) if x["id"] in ok_ids]
        rec["revision"] = p04_revision(reason, when, f"04 entries {ok_ids} meet every condition of PREREG_04 5.3, "
                                                     f"so the final choice is 04")
    return rec


def t04_status(log, fin, last):
    """The 04 state for status (spec 9.2): entries, which meet 5.3 now, and what 5.3 asks for."""
    elig = t04_eligibility(log)
    ok_ids = [i for i, (ok, _w) in elig.items() if ok]
    by_id = {x["id"]: x for x in log["uploads"]}
    fin_e = by_id.get((fin or {}).get("id"))
    rule = (fin or {}).get("prereg04_5_3") or {}
    final_is_04 = bool(fin_e and fin_e.get("test04"))
    rev = rule.get("revision") or {}

    def overridden_by_rev(i):
        """Review round 1: an R20 / R10 final overrides a 04 entry only when the revision recorded with it covered
        that entry and was committed after its prepare (the time rule of final, re-checked now)."""
        prep = (by_id.get(i) or {}).get("prepared_at")
        if i not in (rule.get("eligible_04_entries") or []) or not rev.get("committed_at") or not prep:
            return False
        try:
            return parse_time(rev["committed_at"], "revision commit time") > parse_time(prep, "prepared_at")
        except LogError:
            return False
    final_ok = (final_is_04 and fin_e["id"] in ok_ids and rule.get("chosen") == "04") \
        or (bool(fin) and not final_is_04 and bool(rev) and all(overridden_by_rev(i) for i in ok_ids))
    final_ok = final_ok or t07_final_holds(log, fin)   # PREREG_07 5.4: a holding 07 final is not a 5.3 violation
    final_ok = final_ok or t08_final_holds(log, fin)   # PREREG_08 5.4: a holding 08 final is not a 5.3 violation
    final_ok = final_ok or t09_final_holds(log, fin)   # PREREG_09 5.3: a holding 09 final is not a 5.3 violation
    final_ok = final_ok or t19_final_holds(log, fin)   # PREREG_19 4.3: a holding 19 final is not a 5.3 violation
    final_ok = final_ok or t25_final_holds(log, fin)   # PREREG_25 4.3: a holding 25 final is not a violation
    last_bad = bool(last and last.get("test04") and not elig.get(last["id"], (True, []))[0])
    final_bad = bool(final_is_04 and not elig[fin_e["id"]][0])
    return {"entries": sorted(elig), "eligible": ok_ids,
            "not_eligible": {str(i): w for i, (ok, w) in elig.items() if not ok},
            "final_set": bool(fin), "final_is_04": final_is_04,
            "violation": bool(ok_ids) and bool(fin) and not final_ok,
            **t07_status_key(log),         # PREREG_07 rev 4: only with 07 entries in the log
            **t08_status_key(log),         # PREREG_08 rev 3: only with 08 entries in the log
            **t09_status_key(log),         # PREREG_09 rev 3: only with 09 entries in the log
            **t19_status_key(log),         # PREREG_19 rev 3: only with 19 entries in the log
            **t25_status_key(log),         # PREREG_25 rev 3: only with 25 entries in the log
            "redo_r20": not ok_ids and (last_bad or final_bad)}


def t04_status_lines(st):
    lines = [f"  04（PREREG_04 5.3）：紀錄 {st['entries']}；現在符合條件的 {st['eligible'] or '沒有'}；最終選擇"
             f"{'是 04' if st['final_is_04'] else '不是 04'}"]
    for i, why in st["not_eligible"].items():
        text = "；".join(why)
        lines.append(f"  #{i} 不符合：{text[:600]}{'…' if len(text) > 600 else ''}")
    if st.get("t25_eligible") and not st["final_set"]:   # PREREG_25 4.3: an eligible 25 replaces the 04-19 hints
        lines.append("  有符合 PREREG_25 4.3 的 25：最終選擇要設成 25（docs/TESTDAY_25.md 第 5 步）")
        st = dict(st, eligible=[], t07_eligible=[], t08_eligible=[], t09_eligible=[], t19_eligible=[])
    if st.get("t19_eligible") and not st["final_set"]:   # PREREG_19 4.3: an eligible 19 replaces the 04-09 hints
        lines.append("  有符合 PREREG_19 4.3 的 19：最終選擇要設成 19（docs/TESTDAY_19.md 第 5 步）")
        st = dict(st, eligible=[], t07_eligible=[], t08_eligible=[], t09_eligible=[])
    if st.get("t09_eligible") and not st["final_set"]:   # PREREG_09 5.3: an eligible 09 replaces the 04 / 07 / 08 hints
        lines.append("  有符合 PREREG_09 5.3 的 09：最終選擇要設成 09（docs/TESTDAY_09.md 第 5 步）")
        st = dict(st, eligible=[], t07_eligible=[], t08_eligible=[])
    if st.get("t08_eligible") and not st["final_set"]:   # PREREG_08 5.4: an eligible 08 replaces the 04 / 07 hints
        lines.append("  有符合 PREREG_08 5.4 的 08：最終選擇要設成 08（docs/TESTDAY_08.md 第 6 步）")
        st = dict(st, eligible=[], t07_eligible=[])
    if st.get("t07_eligible") and not st["final_set"]:   # PREREG_07 5.4: an eligible 07 replaces the 04 hint
        lines.append("  有符合 PREREG_07 5.4 的 07：最終選擇要設成 07（docs/TESTDAY_07.md 第 6 步）")
        st = dict(st, eligible=[])
    if st["violation"]:
        lines.append("  違反 PREREG_04 5.3：有符合條件的 04，最終選擇卻不是它（沒有引用修訂，或引用的修訂沒有涵蓋它）；照 "
                     "docs/TESTDAY_04.md 第 7 步重跑 final")
    elif st["eligible"] and not st["final_set"]:
        lines.append("  有符合 PREREG_04 5.3 的 04：最終選擇要設成 04（docs/TESTDAY_04.md 第 7 步）")
    if st["redo_r20"]:
        lines.append("  照 PREREG_04 5.3：final 改 R20，再 prepare 並重傳 R20（docs/TESTDAY_04.md 第 7 步）")
    return lines


def t04_public_lines(log, e):
    """Spec 9.2: a 04 public score equal to an R20 / R10 score (or the other way round) hints at the wrong file
    uploaded (both are named submission_rank.csv). Empty when the log has no 04 entry."""
    if e.get("test07"):                  # PREREG_07 rev 4: a 07 entry is warned by t07_public_lines
        return []
    if e.get("test25"):                  # PREREG_25 rev 3: a 25 entry is warned by t25_public_lines
        return []
    if e.get("test19"):                  # PREREG_19 rev 3: a 19 entry is warned by t19_public_lines
        return []
    if e.get("test09"):                  # PREREG_09 rev 3: a 09 entry is warned by t09_public_lines
        return []
    if e.get("test08"):                  # PREREG_08 rev 3: an 08 entry is warned by t08_public_lines
        return []
    if e["comp"] != "rank" or e.get("public_score") is None:
        return []
    if e.get("test14"):                  # PREREG_14: a 14 entry is warned by t14_public_lines
        return []
    same = [x["id"] for x in uploads_of(log, "rank") if x["id"] != e["id"] and x.get("public_score") is not None
            and bool(x.get("test04")) != bool(e.get("test04")) and abs(x["public_score"] - e["public_score"]) < 1e-9]
    same = [i for i in same if not entry(log, i).get("test07")]   # PREREG_07 rev 4: a 07 entry is not R20 / R10
    same = [i for i in same if not entry(log, i).get("test08")]   # PREREG_08 rev 3: an 08 entry is not R20 / R10
    same = [i for i in same if not entry(log, i).get("test09")]   # PREREG_09 rev 3: a 09 entry is not R20 / R10
    same = [i for i in same if not entry(log, i).get("test19")]   # PREREG_19 rev 3: a 19 entry is not R20 / R10
    same = [i for i in same if not entry(log, i).get("test25")]   # PREREG_25 rev 3: a 25 entry is not R20 / R10
    same = [i for i in same if not entry(log, i).get("test14")]   # PREREG_14: a 14 entry is not R20 / R10
    if not same:
        return []
    if e.get("test04"):
        return [f"WARNING (PREREG_04 rev 3): #{e['id']}（04）的 public 分數和 R20 的 {same} 完全相同：可能傳成 R20 的檔"
                f"（兩份同名）：核對 AIdea 的上傳紀錄"]
    return [f"WARNING (PREREG_04 rev 3): #{e['id']}（R20／R10）的 public 分數和 04 的 {same} 完全相同：可能傳成 04 的檔"
            f"（兩份同名）：核對 AIdea 的上傳紀錄"]


# ---------------------------------------------------------------- PREREG_06 rev 3: test06 (competition 1)
def t06_dir():
    return os.path.join(REPO, *T06_DIR)


def t06_now():
    """The clock for K7' at prepare and final (a module function: the tests replace it)."""
    return datetime.now(TW)


def t06_variant(comp, path):
    """'off' / 'on' for a competition-1 file in results/test06/player/off/ or on/ (only tools/t06_player.py submit
    writes there); None for every other file."""
    if comp != "player":
        return None
    here = os.path.normcase(os.path.realpath(os.path.dirname(os.path.abspath(path))))
    for v in T06_VARIANTS:
        if here == os.path.normcase(os.path.realpath(os.path.join(t06_dir(), v))):
            return v
    return None


def is_t06_file(comp, path):
    return t06_variant(comp, path) is not None


def head_has_p06_revision_row(n, head=None):
    head = git_head_text(P06_DOC) if head is None else head
    return bool(head) and any(line.startswith(f"| rev {n} |") for line in head.splitlines())


def p06_revision(reason, overridden, what):
    """PREREG_06 5.4 (same definition as p04_revision, PREREG_04 rev 3 item 20): overriding the registered 06 choice
    needs --not-06-reason naming 'PREREG_06 rev N' (N >= 4) while docs/PREREG_06_PLAYER_STYLE.md is committed and
    clean, HEAD has a row starting '| rev N |', and the commit that added the row is later than every overridden record.
    Free text alone is refused. -> the record kept with the decision."""
    text = (reason or "").strip()
    m = re.search(r"PREREG_06\s+rev\s+(\d+)", text)
    if not m:
        raise LogError(f"{what}: needs --not-06-reason \"PREREG_06 rev N: <the program error and the condition of "
                       f"PREREG_06 5.3 / 5.4 it breaks>\" citing a committed revision (N >= {P06_MIN_REV}); a reason "
                       f"without a revision is refused (PREREG_06 5.4)")
    n = int(m.group(1))
    if n < P06_MIN_REV:
        raise LogError(f"{what}: --not-06-reason cites PREREG_06 rev {n}; an override needs rev {P06_MIN_REV} or "
                       f"later (PREREG_06 5.4)")
    st = git_file_state(P06_DOC)
    if not st["tracked"] or not st["clean"]:
        raise LogError(f"{what}: {P06_DOC} must be committed and unchanged (tracked={st['tracked']}, "
                       f"clean={st['clean']})")
    row = f"| rev {n} |"
    if not head_has_p06_revision_row(n):
        raise LogError(f"{what}: {P06_DOC} at HEAD has no row starting {row!r}; commit the revision first")
    h, t = git_row_added(P06_DOC, row)
    if h is None:
        raise LogError(f"{what}: no commit found that added {row!r} to {P06_DOC}")
    at = parse_time(t, "commit time")
    early = [w for w in overridden if at <= parse_time(w, "time")]
    if early:
        raise LogError(f"{what}: PREREG_06 rev {n} was committed at {t} ({h[:12]}), not after the records it "
                       f"overrides ({', '.join(early)}); a revision must come after the decision it overturns")
    return {"rev": n, "row": row, "doc": P06_DOC, "commit": h, "committed_at": t, "reason": text,
            "overridden_at": list(overridden)}


def t06_inputs_problems(bound):
    """Every input the 06 submit report lists under inputs_bound is committed, clean and still has the recorded
    SHA-256 (PREREG_06 5.4: test02 re-run after the 06 run, e.g. its dumps or reports, invalidates the 06 run)."""
    if not isinstance(bound, dict) or not bound:
        return [f"the 06 submit report lists no inputs_bound ({T06_REDO_HINT})"]
    probs = []
    missing = [r for r in T06_INPUTS_REQUIRED if r not in bound]
    if missing:
        probs.append(f"inputs_bound lacks {missing}")
    for rel_, want in sorted(bound.items()):
        s = str(rel_)
        if os.path.isabs(s) or "\\" in s or ".." in s.split("/") or s.startswith("/"):
            probs.append(f"inputs_bound key {s!r} is not a repo-relative path")
            continue
        p = repo_path(s)
        if not os.path.isfile(p):
            probs.append(f"{s} is missing ({T06_REDO_HINT})")
            continue
        st = git_file_state(s)
        if not st["tracked"] or not st["clean"]:
            probs.append(f"{s} is not committed and clean (先 commit；內容若已改變：{T06_REDO_HINT})")
            continue
        got = sha256_bound(p)
        if got != want:
            probs.append(f"{s} is {got[:12]}, the 06 submit report bound {str(want)[:12]} ({T06_REDO_HINT})")
    return probs


def t06_report_problems(rep, rel, path, sha, variant):
    """The submit report's own conditions (format, mode, variant, method, check_submission, this file, the seam and
    H6P it records; 06-ON: every PREREG_06 5.3 condition true). -> problems."""
    probs = []
    if rep.get("format") != T06_FORMAT or rep.get("mode") != "real":
        probs.append(f"{rel}: format {rep.get('format')!r}, mode {rep.get('mode')!r} (a real-mode {T06_FORMAT})")
    if rep.get("variant") != variant or rep.get("method") != T06_METHOD[variant]:
        probs.append(f"{rel}: variant {rep.get('variant')!r}, method {rep.get('method')!r}; the file is in "
                     f"results/test06/player/{variant}/ ({T06_METHOD[variant]})")
    cs = rep.get("check_submission") or {}
    if cs.get("official") is not True or cs.get("passed") is not True:
        probs.append(f"{rel}: check_submission official={cs.get('official')!r} passed={cs.get('passed')!r}")
    want = (rep.get("output") or {}).get("sha256")
    if want != sha:
        try:
            hint = lf_hint(path, want, repo_rel(path)) if path and os.path.exists(path) else ""
        except LogError:                                  # a file outside the repository (a dry-run tree)
            hint = ""
        probs.append(f"{rel}: records another {FILE_NAMES['player']} than this file ({str(sha)[:12]})" + hint)
    if (rep.get("seam") or {}).get("passed") is not True:
        probs.append(f"{rel}: the 06 seam it records did not pass")
    if (rep.get("h6p") or {}).get("adopted") is not True:
        probs.append(f"{rel}: H6P is not recorded as adopted (PREREG_06 3)")
    if not rep.get("run_id"):
        probs.append(f"{rel}: no run_id")
    if variant == "on":
        cond = rep.get("on_conditions") or {}
        bad = [k for k in T06_ON_CONDITIONS if (cond.get(k) or {}).get("ok") is not True]
        if bad or rep.get("all_on_conditions") is not True or rep.get("on_produced") is not True:
            probs.append(f"{rel}: 06-ON conditions not all true {bad} (all_on_conditions "
                         f"{rep.get('all_on_conditions')!r}; PREREG_06 5.3)")
    return probs


def check_reports_t06(comp, path, sha):
    """PREREG_06 5.2-5.4 and rev 3: bind a 06 file to the committed tools/t06_player.py submit report next to it, the
    06 seam, both pools' reports, test02's competition-1 reports (submit, local seam, the full raw T1 with PASS or
    NORMALIZED), the committed inputs it lists and K6'. Returns check_reports' shape plus 'test06'."""
    variant = t06_variant(comp, path)
    out, problems = {}, []
    rpath = os.path.join(t06_dir(), variant, T06_REPORT)
    rel, rep = _committed_json(rpath, "the 06 submit report (tools/t06_player.py submit, docs/TESTDAY_06.md)")
    out[rel] = sha256_file(rpath)
    problems += t06_report_problems(rep, rel, path, sha, variant)
    for srel, what, fmt in ((T06_SEAM, "the 06 seam record", T06_SEAM_FORMAT),) + tuple(
            (p, "a 06 pool report", T06_POOL_FORMAT) for p in T06_POOL_REPORTS):
        p = repo_path(srel)
        srel_, obj = _committed_json(p, what)
        out[srel_] = sha256_file(p)
        if obj.get("format") != fmt or obj.get("mode") != "real":
            problems.append(f"{srel_}: format {obj.get('format')!r}, mode {obj.get('mode')!r} (a real-mode {fmt})")
        if fmt == T06_SEAM_FORMAT and obj.get("passed") is not True:
            problems.append(f"{srel_}: the 06 seam did not pass (verdict {obj.get('verdict')!r})")
    srel, sub = _committed_json(repo_path(T06_T02_REPORTS[0]), "test02's competition-1 submit report (Q04 first)")
    out[srel] = sha256_file(repo_path(T06_T02_REPORTS[0]))
    cs = sub.get("check_submission") or {}
    if sub.get("mode") != "real" or cs.get("official") is not True or cs.get("passed") is not True:
        problems.append(f"{srel}: mode {sub.get('mode')!r}, check_submission official {cs.get('official')!r} passed "
                        f"{cs.get('passed')!r}: 06 needs test02's passed real-mode Q04 run")
    lrel, seam = _committed_json(repo_path(T06_T02_REPORTS[1]), "test02's competition-1 local seam")
    out[lrel] = sha256_file(repo_path(T06_T02_REPORTS[1]))
    if seam.get("passed") is not True or seam.get("mode") != "real":
        problems.append(f"{lrel}: passed={seam.get('passed')!r}, mode={seam.get('mode')!r}")
    trel, t1 = _committed_json(repo_path(T06_T02_REPORTS[2]), "the T1 result")
    out[trel] = sha256_file(repo_path(T06_T02_REPORTS[2]))
    problems += t1_problems(t1, trel)
    v = t1.get("verdict")
    if v not in T1_UPLOAD_OK:
        problems.append(f"{trel}: verdict {v!r}; 06 plays only with T1 PASS or NORMALIZED (PREREG_06 2.7: "
                        f"FAIL_OPENING -> Q04, FAIL_STRUCT -> stop)")
    problems += t06_inputs_problems(rep.get("inputs_bound"))
    h, t = git_commit_time(rel)
    k6_ok = t is not None and parse_time(t, "commit time") <= K6P
    if not k6_ok:
        problems.append(f"{rel} was last committed at {t} ({h}), not by K6' {K6P.isoformat()}: no 06 this season "
                        f"(PREREG_06 4); the final choice is Q04 (U5)")
    if problems:
        raise LogError("the file is not backed by a usable 06 run (PREREG_06 5.2-5.4, rev 3):\n - "
                       + "\n - ".join(problems))
    rec = {"variant": variant, "label": T06_LABEL[variant], "run_id": rep.get("run_id"), "h6p": P06_H6P,
           "report": rel, "report_sha256": out[rel],
           "k6p": {"commit": h, "committed_at": t, "deadline": K6P.isoformat(), "ok": k6_ok},
           "top1_differences_vs_q04": rep.get("top1_differences_vs_q04"),
           "top1_differences_off_vs_on": rep.get("top1_differences_off_vs_on"),
           "all_on_conditions": rep.get("all_on_conditions"), "on_produced": rep.get("on_produced"),
           "inputs_bound_checked": len(rep.get("inputs_bound") or {})}
    return {"reports": out, "t1_verdict": v, "fallback_used": None, "test06": rec}


def method_for_t06(variant, given):
    """A 06 file is recorded under its own method; a --method naming the other 06 variant (or no 06 variant) is
    refused."""
    want, other = T06_LABEL[variant], T06_LABEL["on" if variant == "off" else "off"]
    if given and (want not in given or other in given):
        raise LogError(f"--method {given!r} does not name {want}; a results/test06/player/{variant} file is "
                       f"{T06_METHOD[variant]!r}")
    return given or T06_METHOD[variant]


def t06_entries(log):
    return [x for x in log["uploads"] if x.get("comp") == "player" and x.get("test06")]


def q04_first_upload(log):
    """PREREG_06 5.3: Q04's first upload = the first uploaded competition-1 entry of test02's file (not a 06 one)."""
    ups = [x for x in uploads_of(log, "player") if not x.get("test06") and x.get("file") == T06_Q04_FILE]
    return ups[0] if ups else None


def prepare_order_t06(log, t6, sha):
    """PREREG_06 5.3 / 5.5 E1 at prepare: Q04's first upload is logged; 06-ON only before K7', after this run's 06-OFF
    was uploaded, and at most once a season."""
    q04 = q04_first_upload(log)
    if q04 is None:
        raise LogError(f"PREREG_06 5.3: log Q04's first upload ({T06_Q04_FILE}, status uploaded) before preparing a "
                       f"06 upload (docs/TESTDAY.md step 8, then docs/TESTDAY_06.md)")
    t6["q04_first_upload_id"] = q04["id"]
    if t6["variant"] != "on":
        return t6
    now = t06_now()
    if now > K7P:
        raise LogError(f"PREREG_06 4: K7' {K7P.isoformat()} has passed; no 06-ON this season (final 06-OFF)")
    on_up = [x["id"] for x in t06_entries(log) if x["test06"].get("variant") == "on" and x["status"] == "uploaded"]
    if on_up:
        raise LogError(f"PREREG_06 5.5 E1: 06-ON was already uploaded (entry {on_up}); one 06-ON per season")
    pend = [x["id"] for x in t06_entries(log) if x["test06"].get("variant") == "on" and x["status"] == "prepared"
            and x["sha256"] == sha]
    if pend:
        raise LogError(f"this 06-ON file is already prepared as entry {pend}; log its upload with uploaded --id "
                       f"{pend[-1]} (a second 06-ON entry is refused, PREREG_06 5.5 E1)")
    offs = [x for x in uploads_of(log, "player") if (x.get("test06") or {}).get("variant") == "off"
            and x["test06"].get("run_id") == t6.get("run_id")]
    if not offs:
        raise LogError("PREREG_06 5.3: upload this 06 run's 06-OFF first (prepare, OK, uploaded), then 06-ON")
    off = offs[-1]              # PREREG_06 rev 3: 06-ON pairs with the latest 06-OFF of its run uploaded before it
    if off.get("public_score") is None or q04.get("public_score") is None:
        raise LogError(f"PREREG_06 rev 3: log the public scores of Q04 #{q04['id']} and 06-OFF #{off['id']} first "
                       f"(official_log.py public): the same-file check runs before the one 06-ON of the season")
    n = (off["test06"].get("top1_differences_vs_q04") or {}).get("public")
    if abs(off["public_score"] - q04["public_score"]) < T06_SAME_SCORE and (n or 0) > 0:
        raise LogError(f"PREREG_06 rev 3: Q04 #{q04['id']} and 06-OFF #{off['id']} have the same public score "
                       f"({off['public_score']}) although {n} public top-1 differ: possibly the wrong file. Check AIdea's "
                       f"upload record (docs/TESTDAY_06.md step 6); a wrong file: upload 06-OFF again first. A 06-ON on "
                       f"an unchecked 06-OFF is refused (one 06-ON per season)")
    t6["off_entry_id"] = off["id"]
    return t6


def print_t06_ok_lines(t6):
    d = t6.get("top1_differences_vs_q04") or {}
    print(f"- 這一份是 {t6['label']}（{T06_METHOD[t6['variant']]}），不是 11/04 那份 Q04（{T06_Q04_FILE}）；和 Q04 第 1 "
          f"名不同的題數：public {d.get('public')}、private {d.get('private')}；Q04 的第一次上傳是 "
          f"#{t6.get('q04_first_upload_id')}")
    if t6["variant"] == "on":
        o = t6.get("top1_differences_off_vs_on") or {}
        print(f"- 06-ON：public 池是強制打開的探測（PREREG_06 5.3）；和 06-OFF（#{t6.get('off_entry_id')}）第 1 名不同的"
              f"題數：public {o.get('public')}、private {o.get('private')}；PREREG_06 5.3 的條件都成立")
    print(f"- 06 的檢查：06 的 seam 和 test02 的接縫都通過；K6'：06 的報告在 {t6['k6p']['committed_at']} commit"
          f"（期限 11/21 23:59）")


def t06_entry_eligible(x):
    """PREREG_06 5.4 for one logged 06 entry, re-checked now: T1 PASS / NORMALIZED, the report and the uploaded file
    unchanged since prepare, the report's last commit the one recorded and by K6', the report's own conditions (06-ON:
    every 5.3 condition), every bound input unchanged. -> (ok, reasons)."""
    t6, why = x.get("test06") or {}, []
    if x.get("t1_verdict") not in T1_UPLOAD_OK:
        why.append(f"T1 {x.get('t1_verdict')!r}")
    rel_ = t6.get("report") or ""
    rep = None
    rp = repo_path(rel_) if rel_ else ""
    if not rel_ or not os.path.isfile(rp) or sha256_file(rp) != t6.get("report_sha256"):
        why.append(f"{rel_ or 'the 06 submit report'} changed since prepare ({T06_REDO_HINT})")
    else:
        with open(refuse_forbidden(rp), encoding="utf-8") as f:
            rep = json.load(f)
    fp = repo_path(x.get("file") or "") if x.get("file") else ""
    if not fp or not os.path.isfile(fp) or sha256_file(fp) != x.get("sha256"):
        why.append(f"{x.get('file')} changed since prepare")
    k6 = t6.get("k6p") or {}
    h, t = git_commit_time(rel_) if rel_ else (None, None)
    if not k6.get("ok") or h is None or h != k6.get("commit") or parse_time(t, "commit time") > K6P:
        why.append(f"K6': the report's last commit is {h} at {t}; prepare recorded {k6.get('commit')} at "
                   f"{k6.get('committed_at')} (deadline {K6P.isoformat()})")
    if rep is not None:
        why += t06_report_problems(rep, rel_, fp if fp and os.path.isfile(fp) else None, x.get("sha256"),
                                   t6.get("variant"))
        why += t06_inputs_problems(rep.get("inputs_bound"))
    return not why, why


def t06_veto_record(log, off, on):
    """The committed results/test06/player/veto.json when it belongs to these two entries and its d equals the log's
    public scores (PREREG_06 5.3); else (None, reason)."""
    p = repo_path(T06_VETO)
    if not os.path.isfile(p):
        return None, "veto.json missing (PY tools/t06_player.py veto)"
    st = git_file_state(T06_VETO)
    if not st["tracked"] or not st["clean"]:
        return None, f"{T06_VETO} is not committed and clean"
    with open(refuse_forbidden(p), encoding="utf-8") as f:
        v = json.load(f)
    ent = v.get("entries") or {}
    if v.get("format") != T06_VETO_FORMAT or v.get("mode") != "real" or ent.get("off") != off["id"] \
            or ent.get("on") != on["id"]:
        return None, (f"{T06_VETO} (format {v.get('format')!r}, mode {v.get('mode')!r}, entries {ent}) is not the "
                      f"veto of 06-OFF #{off['id']} and 06-ON #{on['id']}")
    if v.get("decided") and not v.get("k7_passed") and v.get("outcome") not in T06_UPLOAD_CHECK_OFF:
        if off.get("public_score") is None or on.get("public_score") is None:
            return None, f"{T06_VETO} is decided but the log lacks a public score"
        d = on["public_score"] - off["public_score"]
        if not _is_num(v.get("d")) or abs(v["d"] - d) > 1e-12 or bool(v.get("vetoed")) != (d < 0) \
                or v.get("final_variant") != ("off" if d < 0 else "on"):
            return None, (f"{T06_VETO}: d {v.get('d')!r} / vetoed {v.get('vetoed')!r} / final "
                          f"{v.get('final_variant')!r} differ from the log (d = {d!r})")
    if v.get("decided") and v.get("outcome") in T06_UPLOAD_CHECK_OFF and v.get("final_variant") != "off":
        return None, f"{T06_VETO}: outcome {v.get('outcome')!r} must give final 06-OFF"
    h, t = git_commit_time(T06_VETO)
    v["_commit"], v["_committed_at"] = h, t
    return v, None


def t06_state(log, now):
    """PREREG_06 5.3 / 5.4 from the log: which entries hold, the veto, and which upload the final choice must be
    ('q04' = U5, 'off', 'on', or 'undecided' while 06-ON waits for its veto before K7')."""
    ents = t06_entries(log)
    elig = {x["id"]: t06_entry_eligible(x) for x in ents}
    ups = uploads_of(log, "player")
    off_ok = [x for x in ups if (x.get("test06") or {}).get("variant") == "off" and elig[x["id"]][0]]
    ons = [x for x in ups if (x.get("test06") or {}).get("variant") == "on"]
    st = {"entries": sorted(elig), "eligible": [i for i, (ok, _w) in elig.items() if ok],
          "not_eligible": {str(i): w for i, (ok, w) in elig.items() if not ok}, "veto": None, "veto_problem": None,
          "on_entry": ons[0]["id"] if ons else None, "k7_passed": now > K7P}
    if not off_ok:
        st.update(designated="q04", rule="no 06-OFF upload meets PREREG_06 5.4: final Q04 (U5, test02)",
                  reupload_off_needed=False, designated_entries=[])
        return st
    on = ons[0] if ons else None
    if on is None:
        st.update(designated="off", rule="06-OFF meets PREREG_06 5.4 and no 06-ON was uploaded: final 06-OFF")
    elif not elig[on["id"]][0]:
        st.update(designated="off", rule=f"06-ON #{on['id']} does not meet PREREG_06 5.3 / 5.4: final 06-OFF")
    else:
        off = [x for x in off_ok if x["test06"].get("run_id") == on["test06"].get("run_id")
               and (parse_time(x["uploaded_at"], "t"), x["id"]) < (parse_time(on["uploaded_at"], "t"), on["id"])]
        # PREREG_06 rev 3: the latest such 06-OFF (as tools/t06_player.py veto pairs)
        v, prob = t06_veto_record(log, off[-1], on) if off else (None, "no eligible 06-OFF of the same run before "
                                                                       "06-ON")
        st["veto"] = {k: (v or {}).get(k) for k in ("decided", "paused", "vetoed", "d", "final_variant", "outcome",
                                                    "k7_passed", "_commit", "_committed_at")} if v else None
        st["veto_problem"] = prob
        in_time = bool(v and v.get("_committed_at") and parse_time(v["_committed_at"], "t") <= K7P)
        on_last = bool(ups) and ups[-1]["id"] == on["id"]
        if v and v.get("decided") and not v.get("paused") and v.get("final_variant") == "off":
            st.update(designated="off", rule={
                "vetoed": "d < 0: vetoed, final 06-OFF (upload it once more)",
                "wrong_upload": "AIdea's upload record shows a wrong file in the 06-OFF / 06-ON pair: no switch, "
                                "final 06-OFF (PREREG_06 rev 3)",
                "upload_check_inconclusive": "the same-file check could not be settled: no switch, final 06-OFF "
                                             "(PREREG_06 rev 3)"}.get(v.get("outcome"),
                                                                       "K7' passed before the veto was decided: "
                                                                       "final 06-OFF, upload 06-OFF once more "
                                                                       "(PREREG_06 4)"))
        elif v and v.get("decided") and not v.get("paused") and not v.get("k7_passed") and in_time \
                and v.get("final_variant") == "on":
            if on_last:
                st.update(designated="on", rule="d >= 0: final 06-ON (the last upload)")
            else:
                st.update(designated="off", rule="d >= 0, but 06-ON is no longer the last upload and cannot be "
                                                 "uploaded again (PREREG_06 5.5 E1): final 06-OFF (PREREG_06 rev 3)")
        elif st["k7_passed"]:
            st.update(designated="off", rule="K7' passed before the veto was decided: final 06-OFF, upload 06-OFF "
                                             "once more (PREREG_06 4)")
        elif v and v.get("paused"):
            st.update(designated="undecided", rule="veto paused (same-file warning): check AIdea's upload record "
                                                   "first (docs/TESTDAY_06.md step 7), then veto --rerun; nothing is "
                                                   "designated and nothing is uploaded before that (PREREG_06 rev 3)")
        else:
            st.update(designated="undecided", rule="06-ON waits for its veto: PY tools/t06_player.py veto, commit "
                                                   "veto.json (before K7')")
    last = ups[-1] if ups else None
    lv = (last or {}).get("test06") or {}
    st["reupload_off_needed"] = bool(st["designated"] == "off" and on is not None and lv.get("variant") != "off")
    st["designated_entries"] = [x["id"] for x in (off_ok if st["designated"] == "off" else
                                                  [on] if st["designated"] == "on" else [])]
    return st


def prepare_rule_t06(log, comp, bound, not_06_reason):
    """PREREG_06 5.4 at prepare: a Q04 (test02) file while a logged 06-OFF upload meets 5.4 needs the revision check
    of --not-06-reason. None when the log has no 06 entry: Q04's prepare is exactly as before."""
    if bound.get("test21"):              # PREREG_21 7: a 21 file is judged on its base in prepare_order_t21
        if (not_06_reason or "").strip():
            raise LogError("--not-06-reason does not apply to a 21 file")
        return None
    if bound.get("test10"):              # PREREG_10 4.4: a 10 file is judged by prepare_rule_t10, never as a Q04 file
        if (not_06_reason or "").strip():
            raise LogError("--not-06-reason applies to a competition-1 Q04 (test02) file only")
        return None
    reason = (not_06_reason or "").strip()
    if comp != "player" or bound.get("test06"):
        if reason:
            raise LogError("--not-06-reason applies to a competition-1 Q04 (test02) file only")
        return None
    if not t06_entries(log):
        return {"rule": "PREREG_06 5.4", "eligible_06_entries": [], "not_06_reason": reason} if reason else None
    st = t06_state(log, t06_now())
    rec = {"rule": "PREREG_06 5.4", "h6p": P06_H6P, "eligible_06_entries": st["eligible"],
           "designated": st["designated"], "not_eligible_06_entries": st["not_eligible"], "checked_at": now_iso(),
           "not_06_reason": reason or None}
    if st["designated"] != "q04":
        when = [x["prepared_at"] for x in t06_entries(log) if x["id"] in st["eligible"]]
        rec["revision"] = p06_revision(reason, when, f"06 entries {st['eligible']} meet PREREG_06 5.4, so a Q04 file "
                                                     f"is not the final choice")
    return rec


def final_rule_t06(log, e, not_06_reason):
    """PREREG_06 5.4 for competition 1. None when no 06 entry is logged (the Q04-only log is handled exactly as
    before). The final choice must be the designated upload (t06_state); any other needs --not-06-reason citing a
    committed later revision (p06_revision), which is recorded."""
    if e["comp"] == "player" and t10_entries(log):     # PREREG_10 4.4 replaces this rule while 10 entries exist
        if (not_06_reason or "").strip():
            raise LogError("the log holds 10 entries: competition 1's final choice follows PREREG_10 4.4; an override "
                           "needs --not-10-reason \"PREREG_10 rev N: ...\"")
        return None
    reason = (not_06_reason or "").strip()
    if e["comp"] != "player":
        if reason:
            raise LogError("--not-06-reason applies to competition 1 (player) only")
        return None
    if not t06_entries(log):
        return {"rule": "PREREG_06 5.4", "eligible_06_entries": [], "not_06_reason": reason} if reason else None
    st = t06_state(log, t06_now())
    chosen = (e.get("test06") or {}).get("variant") or "q04"
    rec = {"rule": "PREREG_06 5.4", "h6p": P06_H6P, "eligible_06_entries": st["eligible"],
           "not_eligible_06_entries": st["not_eligible"], "designated": st["designated"], "designated_rule": st["rule"],
           "veto": st["veto"], "chosen": chosen, "checked_at": now_iso(), "not_06_reason": reason or None}
    ok = chosen == st["designated"] and (chosen == "q04" or e["id"] in st["designated_entries"])
    if ok:
        return rec
    if st["designated"] == "undecided" and not reason:
        raise LogError(f"PREREG_06 5.3: 06-ON #{st['on_entry']} is uploaded and its veto is not decided ({st['rule']}); "
                       f"run PY tools/t06_player.py veto and commit veto.json first (docs/TESTDAY_06.md step 7)")
    if not reason:
        names = {"q04": "Q04 (U5)", "off": "06-OFF", "on": "06-ON", "undecided": "undecided"}
        raise LogError(f"PREREG_06 5.4: the final choice must be {names[st['designated']]} "
                       f"{st['designated_entries'] or ''} ({st['rule']}); entry #{e['id']} is "
                       f"{names.get(chosen, chosen)}. Another choice needs --not-06-reason \"PREREG_06 rev N: ...\"")
    when = [x["prepared_at"] for x in t06_entries(log)]
    rec["revision"] = p06_revision(reason, when, f"entry #{e['id']} ({chosen}) is not the upload PREREG_06 5.4 "
                                                 f"designates ({st['designated']})")
    return rec


def t06_status(log, fin, last, now):
    st = t06_state(log, now)
    by_id = {x["id"]: x for x in log["uploads"]}
    fin_e = by_id.get((fin or {}).get("id"))
    chosen = ((fin_e or {}).get("test06") or {}).get("variant") or ("q04" if fin_e else None)
    rule = (fin or {}).get("prereg06_5_4") or {}
    final_ok = bool(fin_e) and ((chosen == st["designated"] and (chosen == "q04" or fin_e["id"] in
                                                                 st["designated_entries"]))
                                or bool(rule.get("revision")))
    st.update(final_set=bool(fin), final_choice=chosen, violation=bool(fin) and not final_ok,
              last_upload=(last or {}).get("id"),
              k7_reupload_off=bool(st["k7_passed"] and st["on_entry"] and not (st["veto"] or {}).get("decided")))
    if t10_entries(log):                 # PREREG_10 4.4: the final choice is judged by t10_status (prereg10)
        s10 = t10_state(log, now)
        st.update(violation=False, final_set=True, prereg10_designated=s10["designated"])
        if s10["designated"] == "10":
            st.update(reupload_off_needed=False, k7_reupload_off=False)
    return st


def t06_status_lines(st, now):
    names = {"q04": "Q04（U5）", "off": "06-OFF", "on": "06-ON", "undecided": "還沒定（06-ON 等否決）"}
    v = st["veto"]
    veto = ("沒有 06-ON" if st["on_entry"] is None else
            "暫停（同名檔警告）" if v and v.get("paused") else
            (f"上傳紀錄核對（{v.get('outcome')}）：不用開關" if v and v.get("decided")
             and v.get("outcome") in T06_UPLOAD_CHECK_OFF else
             "K7' 已過：不用開關" if v and v.get("decided") and v.get("k7_passed") else
             f"d {v.get('d')}，{'否決' if v.get('vetoed') else '不否決'}" if v and v.get("decided") else
             f"還沒判定（{st['veto_problem']}）"))
    lines = [f"  06（PREREG_06 5.4）：紀錄 {st['entries']}；現在符合條件的 {st['eligible'] or '沒有'}；否決：{veto}；"
             f"最終應該是 {names[st['designated']]} {st['designated_entries'] or ''}"]
    if st.get("prereg10_designated"):     # PREREG_10 4.4: with 10 entries the next block decides
        lines.append("  （記錄有 10：最終選擇照 PREREG_10 4.4，見下面 10 的那一段）")
    for i, why in st["not_eligible"].items():
        text = "；".join(why)
        lines.append(f"  #{i} 不符合：{text[:600]}{'…' if len(text) > 600 else ''}")
    if st["violation"]:
        lines.append("  違反 PREREG_06 5.4：最終選擇不是規則指定的那一份（也沒有引用修訂）；照 docs/TESTDAY_06.md 第 9 步重跑 "
                     "final")
    elif not st["final_set"] and st["designated"] != "undecided":
        lines.append(f"  最終選擇要設成 {names[st['designated']]}（docs/TESTDAY_06.md 第 9 步）")
    if v and v.get("paused") and st["designated"] == "undecided":
        lines.append("  否決暫停（同名檔警告）：先照 docs/TESTDAY_06.md 第 7 步請使用者截 AIdea 的上傳紀錄核對；核對前不重傳、"
                     "不設最終選擇")
    if st["k7_reupload_off"] or st["reupload_off_needed"]:
        lines.append("  照 PREREG_06 5.3／第 4 節：再上傳一次 06-OFF，讓它成為最後一次上傳（docs/TESTDAY_06.md 第 8 步）")
    lines.append(f"  距 K6' 11/21 23:59（06 的正式執行要在這之前 commit）：{_left(now, K6P)}；距 K7' 11/22 23:59（06-ON "
                 f"上傳、兩份分數、否決都要在這之前）：{_left(now, K7P)}")
    return lines


def t06_public_lines(log, e):
    """PREREG_06 5.3: the same-file warning (Q04, 06-OFF and 06-ON are all named submission_player.csv: two equal
    public scores while the 06 reports count public top-1 differences > 0) and, without a warning, d and the veto.
    Empty when the log has no 06 entry."""
    if e["comp"] != "player" or e.get("public_score") is None or not t06_entries(log):
        return []
    ups = uploads_of(log, "player")
    q04 = q04_first_upload(log)
    ons = [x for x in ups if (x.get("test06") or {}).get("variant") == "on"]
    on = ons[0] if ons else None
    key = lambda x: (parse_time(x["uploaded_at"], "uploaded_at"), x["id"])   # noqa: E731
    offs = [x for x in ups if (x.get("test06") or {}).get("variant") == "off"
            and (on is None or (x["test06"].get("run_id") == on["test06"].get("run_id") and key(x) < key(on)))]
    off = offs[-1] if offs else None      # PREREG_06 rev 3: the latest 06-OFF before 06-ON (as the veto pairs)
    ent = {"Q04": q04, "06-OFF": off, "06-ON": on}

    def diff(a, b):
        if a == "Q04":
            src = ent[b]["test06"].get("top1_differences_vs_q04") or {}
        else:
            src = ent["06-ON"]["test06"].get("top1_differences_off_vs_on") or {}
        return src.get("public")
    lines = []
    pairs = [(a, b) for a, b in (("Q04", "06-OFF"), ("Q04", "06-ON"), ("06-OFF", "06-ON"))
             if ent[a] and ent[b] and ent[a].get("public_score") is not None and ent[b].get("public_score") is not None]
    for a, b in pairs:
        n = diff(a, b)
        if abs(ent[a]["public_score"] - ent[b]["public_score"]) < T06_SAME_SCORE and (n or 0) > 0:
            lines.append(f"WARNING (PREREG_06 5.3): {a} #{ent[a]['id']} 和 {b} #{ent[b]['id']} 的 public 分數完全相同"
                         f"（{ent[a]['public_score']}），但 public 池第 1 名不同 {n} 題：可能傳錯檔（三份同名）：核對 AIdea "
                         f"的上傳紀錄；否決暫停")
    if e.get("test06"):                   # PREREG_06 rev 3: the same bytes always score the same
        same = [x for x in ups if x["id"] != e["id"] and x["sha256"] == e["sha256"]
                and x.get("public_score") is not None and abs(x["public_score"] - e["public_score"]) >= T06_SAME_SCORE]
        for x in same:
            lines.append(f"WARNING (PREREG_06 rev 3): #{e['id']} 和 #{x['id']} 是同一份檔（SHA-256 {e['sha256'][:12]}），"
                         f"public 分數卻不同（{e['public_score']} 和 {x['public_score']}）：其中一次傳的不是這一份（三份同名）："
                         f"核對 AIdea 的上傳紀錄（docs/TESTDAY_06.md「同名檔警告」）")
    if off and on and off.get("public_score") is not None and on.get("public_score") is not None and not lines \
            and e["id"] in (off["id"], on["id"]):
        d = on["public_score"] - off["public_score"]
        lines.append(f"PREREG_06 5.3：d = public(06-ON #{on['id']}) − public(06-OFF #{off['id']}) = {d:+.6f} → "
                     + ("否決：最終 06-OFF，再上傳一次 06-OFF" if d < 0 else "不否決：最終 06-ON（已是最後一次上傳）")
                     + "；正式判定由 tools/t06_player.py veto 寫 veto.json")
    return lines


# ---------------------------------------------------------------- PREREG_07 rev 4: test07 (competition 2)
def t07_dir():
    return os.path.join(REPO, *T07_DIR)


def is_t07_file(comp, path):
    """A competition-2 file in results/test07/rank/ is a 07 file (only tools/t07_rank.py apply writes there)."""
    if comp != "rank":
        return False
    here = os.path.realpath(os.path.dirname(os.path.abspath(path)))
    return os.path.normcase(here) == os.path.normcase(os.path.realpath(t07_dir()))


def t07_h7r_problems():
    """PREREG_07 rev 3, re-read from the committed score_H7R.json: its LF SHA-256 is the pinned one, H7R was scored
    on 1,589 questions, Q02H >= 0.57 and the reference (A = Q02) equals H4R's B row within 1e-12."""
    p = repo_path(T07_H7R)
    if not os.path.isfile(p):
        return [f"{T07_H7R} is missing: H7R was not scored (PREREG_07 rev 3)"]
    st = git_file_state(T07_H7R)
    if not st["tracked"] or not st["clean"]:
        return [f"{T07_H7R} is not committed and clean"]
    if sha256_bound(p) != T07_H7R_SHA:
        return [f"{T07_H7R} is not the committed H7R record ({sha256_bound(p)[:12]} vs {T07_H7R_SHA[:12]})"]
    with open(refuse_forbidden(p), encoding="utf-8") as f:
        s = json.load(f)
    probs = []
    n = (s.get("B_minus_A_rank_balanced") or {}).get("n")
    if n != T07_H7R_N:
        probs.append(f"H7R scored {n!r} questions, registered {T07_H7R_N}")
    for k, want in T07_H7R_A.items():
        v = s.get(k)
        if not _is_num(v) or abs(v - want) > 1e-12:
            probs.append(f"H7R reference re-read {k} {v!r} is not {want} (PREREG_07 3)")
    b = s.get("B_rank_balanced")
    if not _is_num(b) or b < T07_H7R_FLOOR:
        probs.append(f"H7R Q02H {b!r} < {T07_H7R_FLOOR} (PREREG_07 rev 3)")
    return probs


def t07_inputs_problems(bound):
    """Every input the 07 apply report lists under inputs_bound is committed, clean and still has its SHA-256."""
    if not isinstance(bound, dict) or not bound:
        return [f"the 07 apply report lists no inputs_bound ({T07_REDO_HINT})"]
    probs = []
    missing = [r for r in T07_INPUTS_REQUIRED if r not in bound]
    if missing:
        probs.append(f"inputs_bound lacks {missing}")
    for rel_, want in sorted(bound.items()):
        s = str(rel_)
        if os.path.isabs(s) or "\\" in s or ".." in s.split("/") or s.startswith("/"):
            probs.append(f"inputs_bound key {s!r} is not a repo-relative path")
            continue
        p = repo_path(s)
        if not os.path.isfile(p):
            probs.append(f"{s} is missing ({T07_REDO_HINT})")
            continue
        st = git_file_state(s)
        if not st["tracked"] or not st["clean"]:
            probs.append(f"{s} is not committed and clean (先 commit；內容若已改變：{T07_REDO_HINT})")
            continue
        got = sha256_bound(p)
        if got != want:
            probs.append(f"{s} is {got[:12]}, the 07 apply report bound {str(want)[:12]} ({T07_REDO_HINT})")
    return probs


def t07_conditions(rep):
    """PREREG_07 5.2 / 5.4 recomputed from the numbers of the 07 apply report (never from its own verdicts).
    -> (record, problems)."""
    probs = []
    gs = rep.get("gpu_seam") or {}
    vals = {"move": gs.get("max_abs_move"), "profile_mean": gs.get("max_abs_profile_mean"),
            "lp_hs": gs.get("max_abs_lp_hs")}
    if gs.get("counts_equal") is not True:
        probs.append("GPU seam: failed / counts / n_moves of the 400 seam games differ from val_hs.npz")
    for k, v in vals.items():
        if not _is_num(v) or v > T07_TOL[k]:
            probs.append(f"GPU seam: largest {k} difference {v!r} > {T07_TOL[k]} (PREREG_07 5.2)")
    q = rep.get("q02_vs_04") or {}
    bound = rep.get("inputs_bound") or {}
    if q.get("equal") is not True or q.get("computed_sha256") != bound.get(T07_04_PROBS):
        probs.append(f"Q02 recomputed ({str(q.get('computed_sha256'))[:12]}) is not 04's rank_test_probs.csv "
                     f"({str(bound.get(T07_04_PROBS))[:12]})")
    cross = rep.get("move_cross_check") or {}
    for s in T07_SETS:
        mm = (cross.get(s) or {}).get("mismatches")
        if mm != 0:
            probs.append(f"move cross-check of {s}: {mm!r} mismatches (registered 0)")
    sl = rep.get("seam_local") or {}
    if sl.get("required_byte_identical") is True:
        seam_ok = sl.get("byte_identical_probs") is True and sl.get("byte_identical_pred_B") is True
    else:
        seam_ok = (_is_num(sl.get("max_abs_diff")) and _is_num(sl.get("tol")) and sl["max_abs_diff"] <= sl["tol"]
                   and _is_count(sl.get("rule_b_differences")) and sl["rule_b_differences"] <= T07_SEAM_B_MAX)
    if not seam_ok:
        probs.append(f"local Q02H seam does not hold: {sl}")
    fails = rep.get("hs_failures") or {}
    rates = {}
    for s in T07_SETS:
        f_, n_ = (fails.get(s) or {}).get("failed_or_no_move"), (fails.get(s) or {}).get("games")
        if not (_is_count(f_) and _is_count(n_) and n_ > 0 and 0 <= f_ <= n_) or f_ * 200 > n_:
            probs.append(f"HS failures of {s}: {f_!r} of {n_!r} games (> 0.5 %: no 07)")
            rates[s] = None
        else:
            rates[s] = f_ / n_
    thr = max(0.5 * T07_RHO_MONO, T07_RHO_FLOOR)
    rho = {s: ((rep.get("hs_direction") or {}).get(s) or {}).get("rho") for s in T07_SETS}
    for s, v in rho.items():
        if not _is_num(v) or v < thr:
            probs.append(f"HS direction of {s}: rho {v!r} < {thr} (no 07)")
    return {"gpu_seam": vals, "hs_failure_rate": rates, "rho": rho, "rho_threshold": thr,
            "seam_local_max_abs_diff": sl.get("max_abs_diff")}, probs


def t07_report_problems(rep, rel, path, sha):
    """The 07 apply report's own conditions (format, mode, exit, combo, check_submission, this file) and PREREG_07
    5.2 recomputed. -> (conditions, problems)."""
    probs = []
    if rep.get("format") != T07_FORMAT or rep.get("mode") != "real":
        probs.append(f"{rel}: format {rep.get('format')!r}, mode {rep.get('mode')!r} (a real-mode {T07_FORMAT})")
    if rep.get("exit_code") != 0 or rep.get("usable") is not True:
        probs.append(f"{rel}: exit_code {rep.get('exit_code')!r}, usable {rep.get('usable')!r}")
    combo = rep.get("combo") or {}
    if combo.get("id") != T07_COMBO_ID or tuple(combo.get("sources") or ()) != T07_COMBO_SOURCES:
        probs.append(f"{rel}: combination {combo.get('id')!r} {combo.get('sources')!r} is not Q02H")
    fake = [s for s, r in (rep.get("hs_sets") or {}).items() if (r or {}).get("fake_from_val")]
    if fake or set(rep.get("hs_sets") or {}) != set(T07_HS_SETS):
        probs.append(f"{rel}: HS sets {sorted(rep.get('hs_sets') or {})}, drill records {fake}")
    cs = rep.get("check_submission") or {}
    if cs.get("mode") != "official" or cs.get("errors"):
        probs.append(f"{rel}: check_submission {cs.get('mode')!r} with errors {cs.get('errors')}")
    want = (rep.get("outputs_sha256") or {}).get(FILE_NAMES["rank"])
    if want != sha:
        try:
            hint = lf_hint(path, want, repo_rel(path)) if path and os.path.exists(path) else ""
        except LogError:
            hint = ""
        probs.append(f"{rel}: records another {FILE_NAMES['rank']} than this file ({str(sha)[:12]})" + hint)
    cond, cprobs = t07_conditions(rep)
    return cond, probs + cprobs


def check_reports_t07(comp, path, sha):
    """PREREG_07 5.2 / 5.4 (spec 8): bind a 07 file to the committed t07_rank.py apply report next to it, its GPU seam
    and local seam records, test02's real R20 run, the full raw T1 (PASS or NORMALIZED), H7R (rev 3), every committed
    input it lists and K6-07. Returns check_reports' shape plus 'test07'."""
    out, problems = {}, []
    rpath = os.path.join(t07_dir(), T07_REPORT)
    rel, rep = _committed_json(rpath, "the test07 apply report (tools/t07_rank.py apply, docs/TESTDAY_07.md)")
    out[rel] = sha256_file(rpath)
    cond, probs = t07_report_problems(rep, rel, path, sha)
    problems += probs
    for srel, what in ((T07_GPU_SEAM, "the test07 GPU seam record"), (T07_SEAM_LOCAL, "the test07 local seam record")):
        p = repo_path(srel)
        srel_, obj = _committed_json(p, what)
        out[srel_] = sha256_file(p)
        if obj.get("passed") is not True or obj.get("mode") != "real":
            problems.append(f"{srel_}: passed {obj.get('passed')!r}, mode {obj.get('mode')!r}")
    t2 = os.path.join(REPO, "results", "test02", "rank", REPORT_OF["rank"])
    t2rel, t2rep = _committed_json(t2, "the test02 rank apply report (R20 goes first, PREREG_07 5.4)")
    out[t2rel] = sha256_file(t2)
    if t2rep.get("mode") != "real" or t2rep.get("fallback_used") is not False \
            or (t2rep.get("seam_local") or {}).get("passed") is not True:
        problems.append(f"{t2rel}: mode {t2rep.get('mode')!r}, fallback_used {t2rep.get('fallback_used')!r}: 07 needs "
                        f"test02's passed real-mode R20 run (FAIL_OPENING: 07 never plays)")
    t1_path = os.path.join(REPO, "results", "test02", "checks", comp, "t1.json")
    trel, t1 = _committed_json(t1_path, "the T1 result")
    out[trel] = sha256_file(t1_path)
    problems += t1_problems(t1, trel)
    v = t1.get("verdict")
    if v not in T1_UPLOAD_OK:
        problems.append(f"{trel}: verdict {v!r}; 07 plays only with T1 PASS or NORMALIZED (PREREG_07 5.4)")
    problems += t07_h7r_problems()
    problems += t07_inputs_problems(rep.get("inputs_bound"))
    h, t = git_commit_time(rel)
    k6_ok = t is not None and parse_time(t, "commit time") <= K6_07
    if not k6_ok:
        problems.append(f"{rel} was last committed at {t} ({h}), not by K6-07 {K6_07.isoformat()}: no 07 this season "
                        f"(PREREG_07 4); the final choice follows PREREG_04 5.3")
    if problems:
        raise LogError("the file is not backed by a usable 07 run (PREREG_07 5.2, 5.4):\n - " + "\n - ".join(problems))
    diffs = rep.get("rule_b_differences") or {}
    rec = dict(cond, combo=T07_COMBO_ID, h7r=P07_H7R, report=rel, report_sha256=out[rel],
               k6_07={"commit": h, "committed_at": t, "deadline": K6_07.isoformat(), "ok": k6_ok},
               t04_report_sha256=(rep.get("inputs_bound") or {}).get(T07_04_REPORT),
               earliest_hs_chunk_created_utc=rep.get("earliest_hs_chunk_created_utc"),
               inputs_bound_checked=len(rep.get("inputs_bound") or {}),
               public_rule_b_differences={k: (diffs.get(k) or {}).get("public") for k in ("04", "R20")},
               rule_b_differences_all={k: (diffs.get(k) or {}).get("all") for k in ("04", "R20")})
    return {"reports": out, "t1_verdict": v, "fallback_used": False, "test07": rec}


def method_for_t07(given):
    """A 07 file is recorded as Q02H, never with R20's or 04's label."""
    if given and not re.search(r"(?<![A-Za-z0-9])%s(?![0-9A-Za-z])" % T07_COMBO_ID, given):
        raise LogError(f"--method {given!r} does not name {T07_COMBO_ID}; a results/test07/rank file is the 07 "
                       f"combination (default {T07_METHOD!r})")
    return given or T07_METHOD


def t07_entries(log):
    return [x for x in log["uploads"] if x.get("comp") == "rank" and x.get("test07")]


def t07_bound_04(log, t7):
    """The uploaded 04 entry whose report is the one the 07 report binds (its t04_apply_report.json SHA-256, LF)."""
    want = t7.get("t04_report_sha256")
    for x in uploads_of(log, "rank"):
        t4_ = x.get("test04") or {}
        rp = repo_path(t4_.get("report") or "") if t4_.get("report") else ""
        if t4_ and rp and os.path.isfile(rp) and sha256_bound(rp) == want:
            return x
    return None


def prepare_order_t07(log, t7):
    """PREREG_07 5.4 order: R20's first upload and the 04 entry the 07 report binds are logged as uploaded (that 04
    still meets PREREG_04 5.3), and every HS chunk was written after R20's first upload."""
    r20 = r20_first_upload(log)
    if r20 is None:
        raise LogError(f"PREREG_07 5.4: log R20's first upload ({T04_R20_FILE}, status uploaded) before preparing a 07 "
                       f"upload (docs/TESTDAY.md step 8)")
    e4 = t07_bound_04(log, t7)
    if e4 is None:
        raise LogError("PREREG_07 5.4: the 04 upload the 07 report binds (results/test04/rank/t04_apply_report.json) is "
                       "not logged as uploaded: upload 04 first (docs/TESTDAY_04.md step 6)")
    ok4, why4 = t04_entry_eligible(e4)
    if not ok4:
        raise LogError(f"PREREG_07 5.4: the bound 04 entry #{e4['id']} no longer meets PREREG_04 5.3 "
                       f"({'; '.join(why4)[:600]}): 07 is not used")
    early = t7.get("earliest_hs_chunk_created_utc")
    if not early or parse_time(early, "HS chunk time") <= parse_time(r20["uploaded_at"], "uploaded_at"):
        raise LogError(f"PREREG_07 5.2: HS computed at {early}, not after R20's first upload ({r20['uploaded_at']})")
    t7["r20_first_upload_id"], t7["t04_entry_id"] = r20["id"], e4["id"]


def print_t07_ok_lines(t7):
    d = t7.get("public_rule_b_differences") or {}
    print(f"- 這一份是 07（Q02H），不是 04（{T07_04_REPORT}）也不是 R20（{T04_R20_FILE}）；public 題規則 B 等級和 04 不同 "
          f"{d.get('04')} 題、和 R20 不同 {d.get('R20')} 題")
    print(f"- 07 的方法：{T07_METHOD}；R20 的第一次上傳是 #{t7.get('r20_first_upload_id')}，綁定的 04 是 "
          f"#{t7.get('t04_entry_id')}；H7R：{P07_H7R}")
    g = t7.get("gpu_seam") or {}
    print("- 07 的檢查：GPU 接縫（每手 {move}、每盤每 profile {profile_mean}、lp_HS {lp_hs}）、Q02 對得上 04、手數交叉核對 0、"
          "本機接縫都通過；HS 失敗率 ".format(**{k: (f"{v:.3g}" if _is_num(v) else v) for k, v in g.items()})
          + "、".join(f"{s} {r:.2%}" for s, r in (t7.get("hs_failure_rate") or {}).items() if r is not None)
          + "；方向 ρ " + "、".join(f"{s} {v:.4f}" for s, v in (t7.get("rho") or {}).items() if _is_num(v)))
    print(f"- K6-07：07 的報告在 {t7['k6_07']['committed_at']} commit（期限 11/19 23:59）")


def t07_entry_eligible(x, log):
    """PREREG_07 5.4 for one logged 07 entry, re-checked now: rev 3's H7R line, the report's last commit is the one
    recorded at prepare and is by K6-07, T1 PASS / NORMALIZED, the recomputed 5.2 conditions, the report and the file
    unchanged, every bound input unchanged, the bound 04 entry still meets PREREG_04 5.3 with the same report, and the
    HS chunks later than R20's first upload. -> (ok, reasons)."""
    t7, why = x.get("test07") or {}, []
    if x.get("t1_verdict") not in T1_UPLOAD_OK:
        why.append(f"T1 {x.get('t1_verdict')!r}")
    why += t07_h7r_problems()
    rel_ = t7.get("report") or ""
    rep = None
    rp = repo_path(rel_) if rel_ else ""
    if not rel_ or not os.path.isfile(rp) or sha256_file(rp) != t7.get("report_sha256"):
        why.append(f"{rel_ or 'the 07 apply report'} changed since prepare ({T07_REDO_HINT})")
    else:
        with open(refuse_forbidden(rp), encoding="utf-8") as f:
            rep = json.load(f)
    fp = repo_path(x.get("file") or "") if x.get("file") else ""
    if not fp or not os.path.isfile(fp) or sha256_file(fp) != x.get("sha256"):
        why.append(f"{x.get('file')} changed since prepare")
    k6 = t7.get("k6_07") or {}
    h, t = git_commit_time(rel_) if rel_ else (None, None)
    if not k6.get("ok") or h is None or h != k6.get("commit") or parse_time(t, "commit time") > K6_07:
        why.append(f"K6-07: the report's last commit is {h} at {t}; prepare recorded {k6.get('commit')} at "
                   f"{k6.get('committed_at')} (deadline {K6_07.isoformat()})")
    if rep is not None:
        why += t07_report_problems(rep, rel_, fp if fp and os.path.isfile(fp) else None, x.get("sha256"))[1]
        why += t07_inputs_problems(rep.get("inputs_bound"))
    e4 = next((y for y in log["uploads"] if y["id"] == t7.get("t04_entry_id")), None)
    if e4 is None or e4.get("status") != "uploaded":
        why.append(f"the bound 04 entry #{t7.get('t04_entry_id')} is not logged as uploaded")
    else:
        rp4 = repo_path((e4.get("test04") or {}).get("report") or "")
        if not os.path.isfile(rp4) or sha256_bound(rp4) != t7.get("t04_report_sha256"):
            why.append(f"the 04 report of #{e4['id']} is not the one the 07 report binds")
        ok4, why4 = t04_entry_eligible(e4)
        if not ok4:
            why.append(f"the bound 04 #{e4['id']} no longer meets PREREG_04 5.3: {'; '.join(why4)[:300]}")
    r20 = r20_first_upload(log)
    early = t7.get("earliest_hs_chunk_created_utc")
    if r20 is None or not early or parse_time(early, "HS chunk time") <= parse_time(r20["uploaded_at"], "uploaded_at"):
        why.append(f"HS chunks ({early}) are not later than R20's first upload ({(r20 or {}).get('uploaded_at')})")
    return not why, why


def t07_eligibility(log):
    return {x["id"]: t07_entry_eligible(x, log) for x in t07_entries(log)}


def head_has_p07_revision_row(n, head=None):
    head = git_head_text(P07_DOC) if head is None else head
    return bool(head) and any(line.startswith(f"| rev {n} |") for line in head.splitlines())


def p07_revision(reason, overridden, what):
    """Overriding a registered 07 decision (PREREG_07 5.4): a reason naming 'PREREG_07 rev N' (N >= 5) whose row is at
    HEAD of the committed, clean docs/PREREG_07_HUMANSL.md and was added after every overridden record."""
    text = (reason or "").strip()
    m = re.search(r"PREREG_07\s+rev\s+(\d+)", text)
    if not m:
        raise LogError(f"{what}: needs --not-07-reason \"PREREG_07 rev N: <the program error and the condition of "
                       f"PREREG_07 5.4 it breaks>\" citing a committed revision (N >= {P07_MIN_REV}); a reason "
                       f"without a revision is refused")
    n = int(m.group(1))
    if n < P07_MIN_REV:
        raise LogError(f"{what}: --not-07-reason cites PREREG_07 rev {n}; an override needs rev {P07_MIN_REV} or later "
                       f"(rev 4 is the test-day procedure itself)")
    st = git_file_state(P07_DOC)
    if not st["tracked"] or not st["clean"]:
        raise LogError(f"{what}: {P07_DOC} must be committed and unchanged (tracked={st['tracked']}, "
                       f"clean={st['clean']})")
    row = f"| rev {n} |"
    if not head_has_p07_revision_row(n):
        raise LogError(f"{what}: {P07_DOC} at HEAD has no row starting {row!r}; commit the revision first")
    h, t = git_row_added(P07_DOC, row)
    if h is None:
        raise LogError(f"{what}: no commit found that added {row!r} to {P07_DOC}")
    at = parse_time(t, "commit time")
    early = [w for w in overridden if at <= parse_time(w, "time")]
    if early:
        raise LogError(f"{what}: PREREG_07 rev {n} was committed at {t} ({h[:12]}), not after the records it "
                       f"overrides ({', '.join(early)}); a revision must come after the decision it overturns")
    return {"rev": n, "row": row, "doc": P07_DOC, "commit": h, "committed_at": t, "reason": text,
            "overridden_at": list(overridden)}


def prepare_rule_t07(log, comp, bound, not_07_reason):
    """PREREG_07 5.4 at prepare: a 04 / R20 / R10 file while a logged 07 entry meets 5.4 needs the revision check of
    --not-07-reason. None when the log has no 07 entry (everything else is exactly as before)."""
    if bound.get("test14"):              # PREREG_14 7: a 14 file is judged on its base in prepare_order_t14
        if (not_07_reason or "").strip():
            raise LogError("--not-07-reason does not apply to a 14 file")
        return None
    reason = (not_07_reason or "").strip()
    if bound.get("test25"):              # PREREG_25 rev 3: a 25 file is judged by prepare_rule_t25
        if reason:
            raise LogError("--not-07-reason does not apply to a 25 file")
        return None
    if bound.get("test19"):              # PREREG_19 rev 3: a 19 file is judged by prepare_rule_t19
        if reason:
            raise LogError("--not-07-reason applies to a competition-2 04 / R20 / R10 file only")
        return None
    if bound.get("test09"):              # PREREG_09 rev 3: a 09 file is judged by prepare_rule_t09
        if reason:
            raise LogError("--not-07-reason applies to a competition-2 04 / R20 / R10 file only")
        return None
    if bound.get("test08"):              # PREREG_08 rev 3: an 08 file is judged by prepare_rule_t08
        if reason:
            raise LogError("--not-07-reason applies to a competition-2 04 / R20 / R10 file only")
        return None
    if comp != "rank" or bound.get("test07"):
        if reason:
            raise LogError("--not-07-reason applies to a competition-2 04 / R20 / R10 file only")
        return None
    elig = t07_eligibility(log)
    if not elig:
        if reason:
            raise LogError("--not-07-reason: the log has no 07 entry to set aside")
        return None
    ok_ids = [i for i, (ok, _w) in elig.items() if ok]
    rec = {"rule": "PREREG_07 5.4", "h7r": P07_H7R, "eligible_07_entries": ok_ids,
           "not_eligible_07_entries": {str(i): w for i, (ok, w) in elig.items() if not ok}, "checked_at": now_iso(),
           "not_07_reason": reason or None}
    if ok_ids:
        when = [x["prepared_at"] for x in t07_entries(log) if x["id"] in ok_ids]
        rec["revision"] = p07_revision(reason, when, f"07 entries {ok_ids} meet PREREG_07 5.4, so a 04 / R20 / R10 "
                                                     f"file is not the final choice")
    return rec


def final_rule_t07(log, e, not_07_reason):
    """PREREG_07 5.4 for competition 2's final choice. None when no 07 entry is logged. A 07 entry is final only while
    it holds; while some 07 entry holds, a 04 / R20 / R10 final needs --not-07-reason (p07_revision)."""
    reason = (not_07_reason or "").strip()
    if e.get("test25"):                  # PREREG_25 rev 3: a 25 final is judged by final_rule_t25
        if reason:
            raise LogError("--not-07-reason does not apply to a 25 entry")
        return None
    if e.get("test19"):                  # PREREG_19 rev 3: a 19 final is judged by final_rule_t19, never as 07
        if reason:
            raise LogError("--not-07-reason applies to a 04 / R20 / R10 final choice, not to a 19 entry")
        return None
    if e.get("test09"):                  # PREREG_09 rev 3: a 09 final is judged by final_rule_t09, never as 07
        if reason:
            raise LogError("--not-07-reason applies to a 04 / R20 / R10 final choice, not to a 09 entry")
        return None
    if e.get("test08"):                  # PREREG_08 rev 3: an 08 final is judged by final_rule_t08, never as 07
        if reason:
            raise LogError("--not-07-reason applies to a 04 / R20 / R10 final choice, not to an 08 entry")
        return None
    if e["comp"] != "rank":
        if reason:
            raise LogError("--not-07-reason applies to competition 2 (rank) only")
        return None
    elig = t07_eligibility(log)
    if not elig:
        if reason:
            raise LogError("--not-07-reason: the log has no 07 entry to set aside")
        return None
    ok_ids = [i for i, (ok, _w) in elig.items() if ok]
    rec = {"rule": "PREREG_07 5.4", "h7r": P07_H7R, "eligible_07_entries": ok_ids,
           "not_eligible_07_entries": {str(i): w for i, (ok, w) in elig.items() if not ok}, "checked_at": now_iso()}
    if e.get("test07"):
        ok, why = elig[e["id"]]
        if not ok:
            raise LogError(f"entry #{e['id']} (07) does not meet PREREG_07 5.4 now ({'; '.join(why)[:900]}): the final "
                           f"choice follows PREREG_04 5.3 (04, else R20, R10 on FAIL_OPENING); upload it again so that "
                           f"it is the last upload (docs/TESTDAY_07.md step 6)")
        if reason:
            raise LogError("--not-07-reason does not apply when the 07 entry is chosen")
        return dict(rec, chosen="07", not_07_reason=None)
    rec.update(chosen="04" if e.get("test04") else ("R10" if e.get("fallback_used") else "R20"),
               not_07_reason=reason or None)
    if ok_ids:
        when = [x["prepared_at"] for x in t07_entries(log) if x["id"] in ok_ids]
        rec["revision"] = p07_revision(reason, when, f"07 entries {ok_ids} meet every condition of PREREG_07 5.4, "
                                                     f"so the final choice is 07")
    return rec


def t07_final_holds(log, fin):
    """The final choice is a 07 entry recorded as chosen 07 that still meets PREREG_07 5.4 (False without 07 entries)."""
    if not fin or not t07_entries(log):
        return False
    e = next((x for x in log["uploads"] if x["id"] == fin.get("id")), None)
    if not e or not e.get("test07") or (fin.get("prereg07_5_4") or {}).get("chosen") != "07":
        return False
    return t07_entry_eligible(e, log)[0]


def t07_status_key(log):
    """{'t07_eligible': [...]} for t04_status, only when the log has 07 entries (else {}: the same dict as before)."""
    if not t07_entries(log):
        return {}
    return {"t07_eligible": [i for i, (ok, _w) in t07_eligibility(log).items() if ok]}


def t07_status(log, fin, last):
    elig = t07_eligibility(log)
    ok_ids = [i for i, (ok, _w) in elig.items() if ok]
    by_id = {x["id"]: x for x in log["uploads"]}
    fin_e = by_id.get((fin or {}).get("id"))
    rule = (fin or {}).get("prereg07_5_4") or {}
    final_is_07 = bool(fin_e and fin_e.get("test07"))
    rev = rule.get("revision") or {}
    def overridden_by_rev(i):
        """A non-07 final sets a 07 entry aside only when the revision recorded with it covered that entry and was
        committed after its prepare (the time rule of final, re-checked now; as t04_status)."""
        prep = (by_id.get(i) or {}).get("prepared_at")
        if i not in (rule.get("eligible_07_entries") or []) or not rev.get("committed_at") or not prep:
            return False
        try:
            return parse_time(rev["committed_at"], "revision commit time") > parse_time(prep, "prepared_at")
        except LogError:
            return False
    final_ok = (final_is_07 and fin_e["id"] in ok_ids and rule.get("chosen") == "07") or \
        (bool(fin) and not final_is_07 and bool(rev) and all(overridden_by_rev(i) for i in ok_ids))
    final_ok = final_ok or t08_final_holds(log, fin)   # PREREG_08 5.4: a holding 08 final is not a 07 violation
    final_ok = final_ok or t09_final_holds(log, fin)   # PREREG_09 5.3: a holding 09 final is not a 07 violation
    final_ok = final_ok or t19_final_holds(log, fin)   # PREREG_19 4.3: a holding 19 final is not a 07 violation
    final_ok = final_ok or t25_final_holds(log, fin)   # PREREG_25 4.3: a holding 25 final is not a violation
    last_bad = bool(last and last.get("test07") and not elig.get(last["id"], (True, []))[0])
    final_bad = bool(final_is_07 and not elig[fin_e["id"]][0])
    return {"entries": sorted(elig), "eligible": ok_ids, "not_eligible": {str(i): w for i, (ok, w) in elig.items()
                                                                          if not ok},
            "final_set": bool(fin), "final_is_07": final_is_07, "violation": bool(ok_ids) and bool(fin) and not final_ok,
            **t08_status_key(log),         # PREREG_08 rev 3: only with 08 entries in the log
            **t09_status_key(log),         # PREREG_09 rev 3: only with 09 entries in the log
            **t19_status_key(log),         # PREREG_19 rev 3: only with 19 entries in the log
            **t25_status_key(log),         # PREREG_25 rev 3: only with 25 entries in the log
            "redo_53": not ok_ids and (last_bad or final_bad)}


def t07_status_lines(st):
    lines = [f"  07（PREREG_07 5.4）：紀錄 {st['entries']}；現在符合條件的 {st['eligible'] or '沒有'}；最終選擇"
             f"{'是 07' if st['final_is_07'] else '不是 07'}"]
    for i, why in st["not_eligible"].items():
        text = "；".join(why)
        lines.append(f"  #{i} 不符合：{text[:600]}{'…' if len(text) > 600 else ''}")
    if st.get("t25_eligible") and not st["final_set"]:   # PREREG_25 4.3: the 25 hint is printed by t04_status_lines
        st = dict(st, eligible=[])
    if st.get("t19_eligible") and not st["final_set"]:   # PREREG_19 4.3: the 19 hint is printed by t04_status_lines
        st = dict(st, eligible=[])
    if st.get("t09_eligible") and not st["final_set"]:   # PREREG_09 5.3: the 09 hint is printed by t04_status_lines
        st = dict(st, eligible=[])
    if st.get("t08_eligible") and not st["final_set"]:   # PREREG_08 5.4: the 08 hint is printed by t04_status_lines
        st = dict(st, eligible=[])
    if st["violation"]:
        lines.append("  違反 PREREG_07 5.4：有符合條件的 07，最終選擇卻不是它（沒有引用修訂，或引用的修訂沒有涵蓋它）；照 "
                     "docs/TESTDAY_07.md 第 6 步重跑 final")
    elif st["eligible"] and not st["final_set"]:
        lines.append("  有符合 PREREG_07 5.4 的 07：最終選擇要設成 07（docs/TESTDAY_07.md 第 6 步）")
    if st["redo_53"]:
        lines.append("  照 PREREG_07 5.4：現在沒有符合的 07，final 改照 PREREG_04 5.3（04，否則 R20），再 prepare 並重傳那一份"
                     "（docs/TESTDAY_07.md 第 6 步）")
    return lines


def t07_public_lines(log, e):
    """PREREG_07 5.4: 07, 04 and R20 share submission_rank.csv. Two equal public scores (|d| < 1e-9) between a 07 entry
    and a 04 / R20 entry while the 07 report counts public-question rule-B differences > 0 between them: a hint that
    the wrong file was uploaded. Empty when the log has no 07 entry."""
    if e["comp"] != "rank" or e.get("public_score") is None or not t07_entries(log):
        return []
    lines = []
    others = [x for x in uploads_of(log, "rank") if x["id"] != e["id"] and x.get("public_score") is not None
              and abs(x["public_score"] - e["public_score"]) < 1e-9]
    for x in others:
        a, b = (e, x) if e.get("test07") else (x, e)
        if not a.get("test07") or b.get("test07"):
            continue
        if b.get("test25"):              # PREREG_25 rev 3: a 25 entry is warned by t25_public_lines
            continue
        if b.get("test19"):              # PREREG_19 rev 3: a 19 entry is warned by t19_public_lines
            continue
        if b.get("test14"):              # PREREG_14: a 14 entry is warned by t14_public_lines
            continue
        if b.get("test09"):              # PREREG_09 rev 3: a 09 entry is warned by t09_public_lines
            continue
        if b.get("test08"):              # PREREG_08 rev 3: an 08 entry is warned by t08_public_lines
            continue
        kind = "04" if b.get("test04") else "R20"
        n = ((a.get("test07") or {}).get("public_rule_b_differences") or {}).get(kind)
        if _is_count(n) and n > 0:
            lines.append(f"WARNING (PREREG_07 5.4): #{a['id']}（07）和 #{b['id']}（{kind}）的 public 分數完全相同，但兩份在 "
                         f"public 題有 {n} 題的規則 B 等級不同：可能傳錯檔（三份同名）：核對 AIdea 的上傳紀錄")
    return lines


# ---------------------------------------------------------------- PREREG_08 rev 3: test08 (competition 2)
def t08_dir():
    return os.path.join(REPO, *T08_DIR)


def is_t08_file(comp, path):
    """A competition-2 file in results/test08/rank/ is an 08 file (only tools/t08_rank.py apply writes there)."""
    if comp != "rank":
        return False
    here = os.path.realpath(os.path.dirname(os.path.abspath(path)))
    return os.path.normcase(here) == os.path.normcase(os.path.realpath(t08_dir()))


def t08_h8r_problems():
    """PREREG_08 3, re-read from the committed score_H8R.json: its LF SHA-256 is the pinned one, 1,589 questions, the
    95 % lower bound of Q08 - Q02H > 0, the reference (A = Q02H) equal to H7R's B row within 1e-12, Q08 >= 0.57."""
    p = repo_path(T08_H8R)
    if not os.path.isfile(p):
        return [f"{T08_H8R} is missing: H8R was not scored (PREREG_08 3)"]
    st = git_file_state(T08_H8R)
    if not st["tracked"] or not st["clean"]:
        return [f"{T08_H8R} is not committed and clean"]
    if sha256_bound(p) != T08_H8R_SHA:
        return [f"{T08_H8R} is not the committed H8R record ({sha256_bound(p)[:12]} vs {T08_H8R_SHA[:12]})"]
    with open(refuse_forbidden(p), encoding="utf-8") as f:
        s = json.load(f)
    probs = []
    d = s.get("B_minus_A_rank_balanced") or {}
    if d.get("n") != T08_H8R_N:
        probs.append(f"H8R scored {d.get('n')!r} questions, registered {T08_H8R_N}")
    lo = (d.get("ci95") or [None])[0]
    if not _is_num(lo) or lo <= 0:
        probs.append(f"H8R lower bound {lo!r} is not > 0: 08 is not adopted (PREREG_08 3)")
    for k, want in T08_H8R_A.items():
        v = s.get(k)
        if not _is_num(v) or abs(v - want) > 1e-12:
            probs.append(f"H8R reference re-read {k} {v!r} is not {want} (PREREG_08 3)")
    b = s.get("B_rank_balanced")
    if not _is_num(b) or b < T08_H8R_FLOOR:
        probs.append(f"H8R Q08 {b!r} < {T08_H8R_FLOOR} (PREREG_08 3)")
    return probs


def t08_inputs_problems(bound):
    """Every input the 08 apply report lists under inputs_bound is committed, clean and still has its SHA-256."""
    if not isinstance(bound, dict) or not bound:
        return [f"the 08 apply report lists no inputs_bound ({T08_REDO_HINT})"]
    probs = []
    missing = [r for r in T08_INPUTS_REQUIRED if r not in bound]
    if missing:
        probs.append(f"inputs_bound lacks {missing}")
    for rel_, want in sorted(bound.items()):
        s = str(rel_)
        if os.path.isabs(s) or "\\" in s or ".." in s.split("/") or s.startswith("/"):
            probs.append(f"inputs_bound key {s!r} is not a repo-relative path")
            continue
        p = repo_path(s)
        if not os.path.isfile(p):
            probs.append(f"{s} is missing ({T08_REDO_HINT})")
            continue
        st = git_file_state(s)
        if not st["tracked"] or not st["clean"]:
            probs.append(f"{s} is not committed and clean (先 commit；內容若已改變：{T08_REDO_HINT})")
            continue
        got = sha256_bound(p)
        if got != want:
            probs.append(f"{s} is {got[:12]}, the 08 apply report bound {str(want)[:12]} ({T08_REDO_HINT})")
    return probs


def t08_conditions(rep):
    """PREREG_08 5.2 / 5.4 recomputed from the numbers of the 08 apply report (never from its own verdicts).
    -> (record, problems)."""
    probs = []
    gs = rep.get("gpu_seam") or {}
    vals = {"std_feature": gs.get("max_abs_std_feature"), "lp_p1": gs.get("max_abs_lp_p1")}
    if gs.get("counts_equal") is not True:
        probs.append("GPU seam: failed / counts / n_moves of the 400 seam games differ from val_seam_feats.npz")
    for k, v in vals.items():
        if not _is_num(v) or v > T08_TOL[k]:
            probs.append(f"GPU seam: largest {k} difference {v!r} > {T08_TOL[k]} (PREREG_08 5.2)")
    q = rep.get("q02h_vs_07") or {}
    bound = rep.get("inputs_bound") or {}
    if q.get("equal") is not True or q.get("computed_sha256") != bound.get(T08_07_PROBS):
        probs.append(f"Q02H recomputed ({str(q.get('computed_sha256'))[:12]}) is not 07's rank_test_probs.csv "
                     f"({str(bound.get(T08_07_PROBS))[:12]})")
    cross = rep.get("move_cross_check") or {}
    for s in T08_SETS:
        mm = (cross.get(s) or {}).get("mismatches")
        if mm != 0:
            probs.append(f"move cross-check of {s}: {mm!r} mismatches (registered 0)")
    sl = rep.get("seam_local") or {}
    if sl.get("required_byte_identical") is True:
        seam_ok = sl.get("byte_identical_probs") is True and sl.get("byte_identical_pred_B") is True
    else:
        seam_ok = (_is_num(sl.get("max_abs_diff")) and _is_num(sl.get("tol")) and sl["max_abs_diff"] <= sl["tol"]
                   and _is_count(sl.get("rule_b_differences")) and sl["rule_b_differences"] <= T08_SEAM_B_MAX)
    if not seam_ok:
        probs.append(f"local Q08 seam does not hold: {sl}")
    fails = rep.get("p1_failures") or {}
    rates = {}
    for s in T08_SETS:
        f_, n_ = (fails.get(s) or {}).get("failed_or_no_move"), (fails.get(s) or {}).get("games")
        if not (_is_count(f_) and _is_count(n_) and n_ > 0 and 0 <= f_ <= n_) or f_ * 200 > n_:
            probs.append(f"P1 failures of {s}: {f_!r} of {n_!r} games (> 0.5 %: no 08)")
            rates[s] = None
        else:
            rates[s] = f_ / n_
    thr = max(0.5 * T08_RHO_MONO, T08_RHO_FLOOR)
    rho = {s: ((rep.get("p1_direction") or {}).get(s) or {}).get("rho") for s in T08_SETS}
    for s, v in rho.items():
        if not _is_num(v) or v < thr:
            probs.append(f"P1 direction of {s}: rho {v!r} < {thr} (no 08)")
    return {"gpu_seam": vals, "p1_failure_rate": rates, "rho": rho, "rho_threshold": thr,
            "seam_local_max_abs_diff": sl.get("max_abs_diff")}, probs


def t08_report_problems(rep, rel, path, sha):
    """The 08 apply report's own conditions (format, mode, exit, combo, check_submission, this file) and PREREG_08
    5.2 recomputed. -> (conditions, problems)."""
    probs = []
    if rep.get("format") != T08_FORMAT or rep.get("mode") != "real":
        probs.append(f"{rel}: format {rep.get('format')!r}, mode {rep.get('mode')!r} (a real-mode {T08_FORMAT})")
    if rep.get("exit_code") != 0 or rep.get("usable") is not True:
        probs.append(f"{rel}: exit_code {rep.get('exit_code')!r}, usable {rep.get('usable')!r}")
    combo = rep.get("combo") or {}
    if combo.get("id") != T08_COMBO_ID or tuple(combo.get("sources") or ()) != T08_COMBO_SOURCES:
        probs.append(f"{rel}: combination {combo.get('id')!r} {combo.get('sources')!r} is not Q08")
    fake = [s for s, r in (rep.get("p1_sets") or {}).items() if (r or {}).get("fake_from_val")]
    if fake or set(rep.get("p1_sets") or {}) != set(T08_P1_SETS):
        probs.append(f"{rel}: P1 sets {sorted(rep.get('p1_sets') or {})}, drill records {fake}")
    cs = rep.get("check_submission") or {}
    if cs.get("mode") != "official" or cs.get("errors"):
        probs.append(f"{rel}: check_submission {cs.get('mode')!r} with errors {cs.get('errors')}")
    want = (rep.get("outputs_sha256") or {}).get(FILE_NAMES["rank"])
    if want != sha:
        try:
            hint = lf_hint(path, want, repo_rel(path)) if path and os.path.exists(path) else ""
        except LogError:
            hint = ""
        probs.append(f"{rel}: records another {FILE_NAMES['rank']} than this file ({str(sha)[:12]})" + hint)
    cond, cprobs = t08_conditions(rep)
    return cond, probs + cprobs


def check_reports_t08(comp, path, sha):
    """PREREG_08 5.2 / 5.4: bind an 08 file to the committed t08_rank.py apply report next to it, its GPU seam and
    local seam records, test02's real R20 run, the full raw T1 (PASS or NORMALIZED), H8R, every committed input it
    lists and K6-08. Returns check_reports' shape plus 'test08'."""
    out, problems = {}, []
    rpath = os.path.join(t08_dir(), T08_REPORT)
    rel, rep = _committed_json(rpath, "the test08 apply report (tools/t08_rank.py apply, docs/TESTDAY_08.md)")
    out[rel] = sha256_file(rpath)
    cond, probs = t08_report_problems(rep, rel, path, sha)
    problems += probs
    for srel, what in ((T08_GPU_SEAM, "the test08 GPU seam record"), (T08_SEAM_LOCAL, "the test08 local seam record")):
        p = repo_path(srel)
        srel_, obj = _committed_json(p, what)
        out[srel_] = sha256_file(p)
        if obj.get("passed") is not True or obj.get("mode") != "real":
            problems.append(f"{srel_}: passed {obj.get('passed')!r}, mode {obj.get('mode')!r}")
    t2 = os.path.join(REPO, "results", "test02", "rank", REPORT_OF["rank"])
    t2rel, t2rep = _committed_json(t2, "the test02 rank apply report (R20 goes first, PREREG_08 5.4)")
    out[t2rel] = sha256_file(t2)
    if t2rep.get("mode") != "real" or t2rep.get("fallback_used") is not False \
            or (t2rep.get("seam_local") or {}).get("passed") is not True:
        problems.append(f"{t2rel}: mode {t2rep.get('mode')!r}, fallback_used {t2rep.get('fallback_used')!r}: 08 needs "
                        f"test02's passed real-mode R20 run (FAIL_OPENING: 08 never plays)")
    t1_path = os.path.join(REPO, "results", "test02", "checks", comp, "t1.json")
    trel, t1 = _committed_json(t1_path, "the T1 result")
    out[trel] = sha256_file(t1_path)
    problems += t1_problems(t1, trel)
    v = t1.get("verdict")
    if v not in T1_UPLOAD_OK:
        problems.append(f"{trel}: verdict {v!r}; 08 plays only with T1 PASS or NORMALIZED (PREREG_08 5.4)")
    problems += t08_h8r_problems()
    problems += t08_inputs_problems(rep.get("inputs_bound"))
    h, t = git_commit_time(rel)
    k6_ok = t is not None and parse_time(t, "commit time") <= K6_08
    if not k6_ok:
        problems.append(f"{rel} was last committed at {t} ({h}), not by K6-08 {K6_08.isoformat()}: no 08 this season "
                        f"(PREREG_08 4); the final choice follows PREREG_07 5.4")
    if problems:
        raise LogError("the file is not backed by a usable 08 run (PREREG_08 5.2, 5.4):\n - " + "\n - ".join(problems))
    diffs = rep.get("rule_b_differences") or {}
    rec = dict(cond, combo=T08_COMBO_ID, h8r=P08_H8R, report=rel, report_sha256=out[rel],
               k6_08={"commit": h, "committed_at": t, "deadline": K6_08.isoformat(), "ok": k6_ok},
               t07_report_sha256=(rep.get("inputs_bound") or {}).get(T08_07_REPORT),
               earliest_p1_chunk_created_utc=rep.get("earliest_p1_chunk_created_utc"),
               inputs_bound_checked=len(rep.get("inputs_bound") or {}),
               public_rule_b_differences={k: (diffs.get(k) or {}).get("public") for k in ("07", "04", "R20")},
               rule_b_differences_all={k: (diffs.get(k) or {}).get("all") for k in ("07", "04", "R20")})
    return {"reports": out, "t1_verdict": v, "fallback_used": False, "test08": rec}


def method_for_t08(given):
    """An 08 file is recorded as Q08, never with R20's, 04's or 07's label."""
    if given and not re.search(r"(?<![A-Za-z0-9])%s(?![0-9A-Za-z])" % T08_COMBO_ID, given):
        raise LogError(f"--method {given!r} does not name {T08_COMBO_ID}; a results/test08/rank file is the 08 "
                       f"combination (default {T08_METHOD!r})")
    return given or T08_METHOD


def t08_entries(log):
    return [x for x in log["uploads"] if x.get("comp") == "rank" and x.get("test08")]


def t08_bound_07(log, t8):
    """The uploaded 07 entry whose report is the one the 08 report binds (its t07_apply_report.json SHA-256, LF)."""
    want = t8.get("t07_report_sha256")
    for x in uploads_of(log, "rank"):
        t7_ = x.get("test07") or {}
        rp = repo_path(t7_.get("report") or "") if t7_.get("report") else ""
        if t7_ and rp and os.path.isfile(rp) and sha256_bound(rp) == want:
            return x
    return None


def prepare_order_t08(log, t8):
    """PREREG_08 5.4 order: R20's first upload and the 07 entry the 08 report binds are logged as uploaded (that 07
    still meets PREREG_07 5.4, which needs its 04 uploaded), and every P1 chunk was written after R20's first upload."""
    r20 = r20_first_upload(log)
    if r20 is None:
        raise LogError(f"PREREG_08 5.4: log R20's first upload ({T04_R20_FILE}, status uploaded) before preparing an 08 "
                       f"upload (docs/TESTDAY.md step 8)")
    e7 = t08_bound_07(log, t8)
    if e7 is None:
        raise LogError("PREREG_08 5.4: the 07 upload the 08 report binds (results/test07/rank/t07_apply_report.json) is "
                       "not logged as uploaded: upload 04 and 07 first (docs/TESTDAY_04.md, docs/TESTDAY_07.md)")
    ok7, why7 = t07_entry_eligible(e7, log)
    if not ok7:
        raise LogError(f"PREREG_08 5.4: the bound 07 entry #{e7['id']} no longer meets PREREG_07 5.4 "
                       f"({'; '.join(why7)[:600]}): 08 is not used")
    early = t8.get("earliest_p1_chunk_created_utc")
    if not early or parse_time(early, "P1 chunk time") <= parse_time(r20["uploaded_at"], "uploaded_at"):
        raise LogError(f"PREREG_08 5.2: P1 computed at {early}, not after R20's first upload ({r20['uploaded_at']})")
    t8["r20_first_upload_id"], t8["t07_entry_id"] = r20["id"], e7["id"]
    t8["t04_entry_id"] = (e7.get("test07") or {}).get("t04_entry_id")


def print_t08_ok_lines(t8):
    d = t8.get("public_rule_b_differences") or {}
    print(f"- 這一份是 08（Q08），不是 07（{T08_07_REPORT}）、04 也不是 R20（{T04_R20_FILE}）；public 題規則 B 等級和 07 不同 "
          f"{d.get('07')} 題、和 04 不同 {d.get('04')} 題、和 R20 不同 {d.get('R20')} 題")
    print(f"- 08 的方法：{T08_METHOD}；R20 的第一次上傳是 #{t8.get('r20_first_upload_id')}，綁定的 07 是 "
          f"#{t8.get('t07_entry_id')}（它綁的 04 是 #{t8.get('t04_entry_id')}）；H8R：{P08_H8R}")
    g = t8.get("gpu_seam") or {}
    print("- 08 的檢查：GPU 接縫（標準化特徵 {std_feature}、lp_P1TO {lp_p1}）、Q02H 對得上 07、手數交叉核對 0、"
          "本機接縫都通過；P1 失敗率 ".format(**{k: (f"{v:.3g}" if _is_num(v) else v) for k, v in g.items()})
          + "、".join(f"{s} {r:.2%}" for s, r in (t8.get("p1_failure_rate") or {}).items() if r is not None)
          + "；方向 ρ " + "、".join(f"{s} {v:.4f}" for s, v in (t8.get("rho") or {}).items() if _is_num(v)))
    print(f"- K6-08：08 的報告在 {t8['k6_08']['committed_at']} commit（期限 11/19 23:59）")


def t08_entry_eligible(x, log):
    """PREREG_08 5.4 for one logged 08 entry, re-checked now: H8R, the report's last commit is the one recorded at
    prepare and is by K6-08, T1 PASS / NORMALIZED, the recomputed 5.2 conditions, the report and the file unchanged,
    every bound input unchanged, the bound 07 entry still meets PREREG_07 5.4 with the same report, and the P1 chunks
    later than R20's first upload. -> (ok, reasons)."""
    t8, why = x.get("test08") or {}, []
    if x.get("t1_verdict") not in T1_UPLOAD_OK:
        why.append(f"T1 {x.get('t1_verdict')!r}")
    why += t08_h8r_problems()
    rel_ = t8.get("report") or ""
    rep = None
    rp = repo_path(rel_) if rel_ else ""
    if not rel_ or not os.path.isfile(rp) or sha256_file(rp) != t8.get("report_sha256"):
        why.append(f"{rel_ or 'the 08 apply report'} changed since prepare ({T08_REDO_HINT})")
    else:
        with open(refuse_forbidden(rp), encoding="utf-8") as f:
            rep = json.load(f)
    fp = repo_path(x.get("file") or "") if x.get("file") else ""
    if not fp or not os.path.isfile(fp) or sha256_file(fp) != x.get("sha256"):
        why.append(f"{x.get('file')} changed since prepare")
    k6 = t8.get("k6_08") or {}
    h, t = git_commit_time(rel_) if rel_ else (None, None)
    if not k6.get("ok") or h is None or h != k6.get("commit") or parse_time(t, "commit time") > K6_08:
        why.append(f"K6-08: the report's last commit is {h} at {t}; prepare recorded {k6.get('commit')} at "
                   f"{k6.get('committed_at')} (deadline {K6_08.isoformat()})")
    if rep is not None:
        why += t08_report_problems(rep, rel_, fp if fp and os.path.isfile(fp) else None, x.get("sha256"))[1]
        why += t08_inputs_problems(rep.get("inputs_bound"))
    e7 = next((y for y in log["uploads"] if y["id"] == t8.get("t07_entry_id")), None)
    if e7 is None or e7.get("status") != "uploaded":
        why.append(f"the bound 07 entry #{t8.get('t07_entry_id')} is not logged as uploaded")
    else:
        rp7 = repo_path((e7.get("test07") or {}).get("report") or "")
        if not os.path.isfile(rp7) or sha256_bound(rp7) != t8.get("t07_report_sha256"):
            why.append(f"the 07 report of #{e7['id']} is not the one the 08 report binds")
        ok7, why7 = t07_entry_eligible(e7, log)
        if not ok7:
            why.append(f"the bound 07 #{e7['id']} no longer meets PREREG_07 5.4: {'; '.join(why7)[:300]}")
    r20 = r20_first_upload(log)
    early = t8.get("earliest_p1_chunk_created_utc")
    if r20 is None or not early or parse_time(early, "P1 chunk time") <= parse_time(r20["uploaded_at"], "uploaded_at"):
        why.append(f"P1 chunks ({early}) are not later than R20's first upload ({(r20 or {}).get('uploaded_at')})")
    return not why, why


def t08_eligibility(log):
    return {x["id"]: t08_entry_eligible(x, log) for x in t08_entries(log)}


def head_has_p08_revision_row(n, head=None):
    head = git_head_text(P08_DOC) if head is None else head
    return bool(head) and any(line.startswith(f"| rev {n} |") for line in head.splitlines())


def p08_revision(reason, overridden, what):
    """Overriding a registered 08 decision (PREREG_08 5.4): a reason naming 'PREREG_08 rev N' (N >= 4) whose row is at
    HEAD of the committed, clean docs/PREREG_08_STRENGTH_PROBE.md and was added after every overridden record."""
    text = (reason or "").strip()
    m = re.search(r"PREREG_08\s+rev\s+(\d+)", text)
    if not m:
        raise LogError(f"{what}: needs --not-08-reason \"PREREG_08 rev N: <the program error and the condition of "
                       f"PREREG_08 5.4 it breaks>\" citing a committed revision (N >= {P08_MIN_REV}); a reason "
                       f"without a revision is refused")
    n = int(m.group(1))
    if n < P08_MIN_REV:
        raise LogError(f"{what}: --not-08-reason cites PREREG_08 rev {n}; an override needs rev {P08_MIN_REV} or later "
                       f"(rev 3 is the test-day procedure itself)")
    st = git_file_state(P08_DOC)
    if not st["tracked"] or not st["clean"]:
        raise LogError(f"{what}: {P08_DOC} must be committed and unchanged (tracked={st['tracked']}, "
                       f"clean={st['clean']})")
    row = f"| rev {n} |"
    if not head_has_p08_revision_row(n):
        raise LogError(f"{what}: {P08_DOC} at HEAD has no row starting {row!r}; commit the revision first")
    h, t = git_row_added(P08_DOC, row)
    if h is None:
        raise LogError(f"{what}: no commit found that added {row!r} to {P08_DOC}")
    at = parse_time(t, "commit time")
    early = [w for w in overridden if at <= parse_time(w, "time")]
    if early:
        raise LogError(f"{what}: PREREG_08 rev {n} was committed at {t} ({h[:12]}), not after the records it "
                       f"overrides ({', '.join(early)}); a revision must come after the decision it overturns")
    return {"rev": n, "row": row, "doc": P08_DOC, "commit": h, "committed_at": t, "reason": text,
            "overridden_at": list(overridden)}


def prepare_rule_t08(log, comp, bound, not_08_reason):
    """PREREG_08 5.4 at prepare: a 07 / 04 / R20 / R10 file while a logged 08 entry meets 5.4 needs the revision check
    of --not-08-reason. None when the log has no 08 entry (everything else is exactly as before)."""
    if bound.get("test14"):              # PREREG_14 7: a 14 file is judged on its base in prepare_order_t14
        if (not_08_reason or "").strip():
            raise LogError("--not-08-reason does not apply to a 14 file")
        return None
    reason = (not_08_reason or "").strip()
    if bound.get("test25"):              # PREREG_25 rev 3: a 25 file is judged by prepare_rule_t25
        if reason:
            raise LogError("--not-08-reason does not apply to a 25 file")
        return None
    if bound.get("test19"):              # PREREG_19 rev 3: a 19 file is judged by prepare_rule_t19
        if reason:
            raise LogError("--not-08-reason applies to a competition-2 07 / 04 / R20 / R10 file only")
        return None
    if bound.get("test09"):              # PREREG_09 rev 3: a 09 file is judged by prepare_rule_t09
        if reason:
            raise LogError("--not-08-reason applies to a competition-2 07 / 04 / R20 / R10 file only")
        return None
    if comp != "rank" or bound.get("test08"):
        if reason:
            raise LogError("--not-08-reason applies to a competition-2 07 / 04 / R20 / R10 file only")
        return None
    elig = t08_eligibility(log)
    if not elig:
        if reason:
            raise LogError("--not-08-reason: the log has no 08 entry to set aside")
        return None
    ok_ids = [i for i, (ok, _w) in elig.items() if ok]
    rec = {"rule": "PREREG_08 5.4", "h8r": P08_H8R, "eligible_08_entries": ok_ids,
           "not_eligible_08_entries": {str(i): w for i, (ok, w) in elig.items() if not ok}, "checked_at": now_iso(),
           "not_08_reason": reason or None}
    if ok_ids:
        when = [x["prepared_at"] for x in t08_entries(log) if x["id"] in ok_ids]
        rec["revision"] = p08_revision(reason, when, f"08 entries {ok_ids} meet PREREG_08 5.4, so a 07 / 04 / R20 / "
                                                     f"R10 file is not the final choice")
    return rec


def final_rule_t08(log, e, not_08_reason):
    """PREREG_08 5.4 for competition 2's final choice. None when no 08 entry is logged. An 08 entry is final only
    while it holds; while some 08 entry holds, a 07 / 04 / R20 / R10 final needs --not-08-reason (p08_revision)."""
    reason = (not_08_reason or "").strip()
    if e.get("test25"):                  # PREREG_25 rev 3: a 25 final is judged by final_rule_t25
        if reason:
            raise LogError("--not-08-reason does not apply to a 25 entry")
        return None
    if e.get("test19"):                  # PREREG_19 rev 3: a 19 final is judged by final_rule_t19, never as 08
        if reason:
            raise LogError("--not-08-reason applies to a 07 / 04 / R20 / R10 final choice, not to a 19 entry")
        return None
    if e.get("test09"):                  # PREREG_09 rev 3: a 09 final is judged by final_rule_t09, never as 08
        if reason:
            raise LogError("--not-08-reason applies to a 07 / 04 / R20 / R10 final choice, not to a 09 entry")
        return None
    if e["comp"] != "rank":
        if reason:
            raise LogError("--not-08-reason applies to competition 2 (rank) only")
        return None
    elig = t08_eligibility(log)
    if not elig:
        if reason:
            raise LogError("--not-08-reason: the log has no 08 entry to set aside")
        return None
    ok_ids = [i for i, (ok, _w) in elig.items() if ok]
    rec = {"rule": "PREREG_08 5.4", "h8r": P08_H8R, "eligible_08_entries": ok_ids,
           "not_eligible_08_entries": {str(i): w for i, (ok, w) in elig.items() if not ok}, "checked_at": now_iso()}
    if e.get("test08"):
        ok, why = elig[e["id"]]
        if not ok:
            raise LogError(f"entry #{e['id']} (08) does not meet PREREG_08 5.4 now ({'; '.join(why)[:900]}): the final "
                           f"choice follows PREREG_07 5.4 (07, else 04, else R20, R10 on FAIL_OPENING); upload it "
                           f"again so that it is the last upload (docs/TESTDAY_08.md step 6)")
        if reason:
            raise LogError("--not-08-reason does not apply when the 08 entry is chosen")
        return dict(rec, chosen="08", not_08_reason=None)
    rec.update(chosen="07" if e.get("test07") else "04" if e.get("test04") else
               ("R10" if e.get("fallback_used") else "R20"), not_08_reason=reason or None)
    if ok_ids:
        when = [x["prepared_at"] for x in t08_entries(log) if x["id"] in ok_ids]
        rec["revision"] = p08_revision(reason, when, f"08 entries {ok_ids} meet every condition of PREREG_08 5.4, "
                                                     f"so the final choice is 08")
    return rec


def t08_final_holds(log, fin):
    """The final choice is an 08 entry recorded as chosen 08 that still meets PREREG_08 5.4 (False without 08
    entries)."""
    if not fin or not t08_entries(log):
        return False
    e = next((x for x in log["uploads"] if x["id"] == fin.get("id")), None)
    if not e or not e.get("test08") or (fin.get("prereg08_5_4") or {}).get("chosen") != "08":
        return False
    return t08_entry_eligible(e, log)[0]


def t08_status_key(log):
    """{'t08_eligible': [...]} for t04_status / t07_status, only when the log has 08 entries (else {})."""
    if not t08_entries(log):
        return {}
    return {"t08_eligible": [i for i, (ok, _w) in t08_eligibility(log).items() if ok]}


def t08_status(log, fin, last):
    elig = t08_eligibility(log)
    ok_ids = [i for i, (ok, _w) in elig.items() if ok]
    by_id = {x["id"]: x for x in log["uploads"]}
    fin_e = by_id.get((fin or {}).get("id"))
    rule = (fin or {}).get("prereg08_5_4") or {}
    final_is_08 = bool(fin_e and fin_e.get("test08"))
    rev = rule.get("revision") or {}

    def overridden_by_rev(i):
        """A non-08 final sets an 08 entry aside only when the revision recorded with it covered that entry and was
        committed after its prepare (the time rule of final, re-checked now; as t07_status)."""
        prep = (by_id.get(i) or {}).get("prepared_at")
        if i not in (rule.get("eligible_08_entries") or []) or not rev.get("committed_at") or not prep:
            return False
        try:
            return parse_time(rev["committed_at"], "revision commit time") > parse_time(prep, "prepared_at")
        except LogError:
            return False
    final_ok = (final_is_08 and fin_e["id"] in ok_ids and rule.get("chosen") == "08") or \
        (bool(fin) and not final_is_08 and bool(rev) and all(overridden_by_rev(i) for i in ok_ids))
    final_ok = final_ok or t09_final_holds(log, fin)   # PREREG_09 5.3: a holding 09 final is not an 08 violation
    final_ok = final_ok or t19_final_holds(log, fin)   # PREREG_19 4.3: a holding 19 final is not an 08 violation
    final_ok = final_ok or t25_final_holds(log, fin)   # PREREG_25 4.3: a holding 25 final is not a violation
    last_bad = bool(last and last.get("test08") and not elig.get(last["id"], (True, []))[0])
    final_bad = bool(final_is_08 and not elig[fin_e["id"]][0])
    return {"entries": sorted(elig), "eligible": ok_ids, "not_eligible": {str(i): w for i, (ok, w) in elig.items()
                                                                          if not ok},
            "final_set": bool(fin), "final_is_08": final_is_08, "violation": bool(ok_ids) and bool(fin) and not final_ok,
            **t09_status_key(log),         # PREREG_09 rev 3: only with 09 entries in the log
            **t19_status_key(log),         # PREREG_19 rev 3: only with 19 entries in the log
            **t25_status_key(log),         # PREREG_25 rev 3: only with 25 entries in the log
            "redo_07": not ok_ids and (last_bad or final_bad)}


def t08_status_lines(st):
    lines = [f"  08（PREREG_08 5.4）：紀錄 {st['entries']}；現在符合條件的 {st['eligible'] or '沒有'}；最終選擇"
             f"{'是 08' if st['final_is_08'] else '不是 08'}"]
    for i, why in st["not_eligible"].items():
        text = "；".join(why)
        lines.append(f"  #{i} 不符合：{text[:600]}{'…' if len(text) > 600 else ''}")
    if st["violation"]:
        lines.append("  違反 PREREG_08 5.4：有符合條件的 08，最終選擇卻不是它（沒有引用修訂，或引用的修訂沒有涵蓋它）；照 "
                     "docs/TESTDAY_08.md 第 6 步重跑 final")
    if st["redo_07"]:
        lines.append("  照 PREREG_08 5.4：現在沒有符合的 08，final 改照 PREREG_07 5.4（07，否則 04，否則 R20），再 prepare 並重傳"
                     "那一份（docs/TESTDAY_08.md 第 6 步）")
    return lines


def t08_public_lines(log, e):
    """PREREG_08 5.4: 08, 07, 04 and R20 share submission_rank.csv. Two equal public scores (|d| < 1e-9) between an 08
    entry and a 07 / 04 / R20 entry while the 08 report counts public-question rule-B differences > 0 between them: a
    hint that the wrong file was uploaded. Empty when the log has no 08 entry."""
    if e["comp"] != "rank" or e.get("public_score") is None or not t08_entries(log):
        return []
    lines = []
    others = [x for x in uploads_of(log, "rank") if x["id"] != e["id"] and x.get("public_score") is not None
              and abs(x["public_score"] - e["public_score"]) < 1e-9]
    for x in others:
        a, b = (e, x) if e.get("test08") else (x, e)
        if not a.get("test08") or b.get("test08"):
            continue
        if b.get("test25"):              # PREREG_25 rev 3: a 25 entry is warned by t25_public_lines
            continue
        if b.get("test19"):              # PREREG_19 rev 3: a 19 entry is warned by t19_public_lines
            continue
        if b.get("test14"):              # PREREG_14: a 14 entry is warned by t14_public_lines
            continue
        if b.get("test09"):              # PREREG_09 rev 3: a 09 entry is warned by t09_public_lines
            continue
        kind = "07" if b.get("test07") else "04" if b.get("test04") else "R20"
        n = ((a.get("test08") or {}).get("public_rule_b_differences") or {}).get(kind)
        if _is_count(n) and n > 0:
            lines.append(f"WARNING (PREREG_08 5.4): #{a['id']}（08）和 #{b['id']}（{kind}）的 public 分數完全相同，但兩份在 "
                         f"public 題有 {n} 題的規則 B 等級不同：可能傳錯檔（四份同名）：核對 AIdea 的上傳紀錄")
    return lines


# ---------------------------------------------------------------- PREREG_09 rev 3: test09 (competition 2)
def t09_dir():
    return os.path.join(REPO, *T09_DIR)


def is_t09_file(comp, path):
    """A competition-2 file in results/test09/rank/ is a 09 file (only tools/t09_rank.py apply writes there)."""
    if comp != "rank":
        return False
    here = os.path.realpath(os.path.dirname(os.path.abspath(path)))
    return os.path.normcase(here) == os.path.normcase(os.path.realpath(t09_dir()))


def t09_adoption_problems():
    """PREREG_09 2.7 / RESULTS.md, re-read from the committed validation apply report of prereg09: its LF SHA-256 is
    the pinned one, Q09, usable true after the audit (finalized)."""
    p = repo_path(T09_ADOPTION)
    if not os.path.isfile(p):
        return [f"{T09_ADOPTION} is missing: 09 was not adopted (PREREG_09 2.7)"]
    st = git_file_state(T09_ADOPTION)
    if not st["tracked"] or not st["clean"]:
        return [f"{T09_ADOPTION} is not committed and clean"]
    if sha256_bound(p) != T09_ADOPTION_SHA:
        return [f"{T09_ADOPTION} is not the committed adoption record ({sha256_bound(p)[:12]} vs "
                f"{T09_ADOPTION_SHA[:12]})"]
    with open(refuse_forbidden(p), encoding="utf-8") as f:
        s = json.load(f)
    if s.get("combo") != T09_COMBO_ID or s.get("usable") is not True or not s.get("finalized"):
        return [f"{T09_ADOPTION}: combo {s.get('combo')!r}, usable {s.get('usable')!r}, finalized "
                f"{s.get('finalized')!r} (PREREG_09 2.7: 09 is not adopted)"]
    return []


def t09_inputs_problems(bound):
    """Every input the 09 apply report lists under inputs_bound is committed, clean and still has its SHA-256."""
    if not isinstance(bound, dict) or not bound:
        return [f"the 09 apply report lists no inputs_bound ({T09_REDO_HINT})"]
    probs = []
    missing = [r for r in T09_INPUTS_REQUIRED if r not in bound]
    if missing:
        probs.append(f"inputs_bound lacks {missing}")
    for rel_, want in sorted(bound.items()):
        s = str(rel_)
        if os.path.isabs(s) or "\\" in s or ".." in s.split("/") or s.startswith("/"):
            probs.append(f"inputs_bound key {s!r} is not a repo-relative path")
            continue
        p = repo_path(s)
        if not os.path.isfile(p):
            probs.append(f"{s} is missing ({T09_REDO_HINT})")
            continue
        st = git_file_state(s)
        if not st["tracked"] or not st["clean"]:
            probs.append(f"{s} is not committed and clean (先 commit；內容若已改變：{T09_REDO_HINT})")
            continue
        got = sha256_bound(p)
        if got != want:
            probs.append(f"{s} is {got[:12]}, the 09 apply report bound {str(want)[:12]} ({T09_REDO_HINT})")
    return probs


def t09_conditions(rep):
    """PREREG_09 5.2 / 5.3 recomputed from the numbers of the 09 apply report (never from its own verdicts).
    -> (record, problems)."""
    probs = []
    gs = rep.get("gpu_seam") or {}
    vals = {"std_ex": gs.get("max_abs_std_ex"), "lp_p1e": gs.get("max_abs_lp_p1e")}
    if gs.get("counts_equal") is not True:
        probs.append("09 GPU seam: failed / counts / n_moves of the 400 seam games differ from the validation chunks")
    # std_ex is judged over the 42 EX features other than the two frac_lowp counts (PREREG_09 rev 4); those are in
    # the report's frac_lowp_record_only and never judged here
    for k, v in vals.items():
        if not _is_num(v) or v > T09_TOL[k]:
            probs.append(f"09 GPU seam: largest {k} difference {v!r} > {T09_TOL[k]} (PREREG_09 5.2)")
    q = rep.get("q08_vs_08") or {}
    bound = rep.get("inputs_bound") or {}
    if q.get("equal") is not True or q.get("t08_sha256") != bound.get(T09_08_PROBS) or \
            q.get("p1_ok_equal_t08") is not True or q.get("usable_equal_t08") is not True or \
            not _is_num(q.get("max_abs_diff")) or q["max_abs_diff"] > T09_TOL_Q08 or \
            q.get("rule_a_differences") != 0 or q.get("rule_b_differences") != 0:
        probs.append(f"Q08 recomputed is not 08's rank_test_probs.csv ({str(q.get('t08_sha256'))[:12]} vs "
                     f"{str(bound.get(T09_08_PROBS))[:12]}; max |dP| {q.get('max_abs_diff')!r}, usable equal "
                     f"{q.get('p1_ok_equal_t08')!r} / {q.get('usable_equal_t08')!r})")
    sl = rep.get("seam_local") or {}
    if sl.get("required_byte_identical") is True:
        seam_ok = sl.get("byte_identical_probs") is True and sl.get("byte_identical_pred_B") is True
    else:
        seam_ok = (_is_num(sl.get("max_abs_diff")) and _is_num(sl.get("tol")) and sl["max_abs_diff"] <= sl["tol"]
                   and _is_count(sl.get("rule_b_differences")) and sl["rule_b_differences"] <= T09_SEAM_B_MAX)
    if not seam_ok:
        probs.append(f"local Q09 seam does not hold: {sl}")
    fails = rep.get("p1_failures") or {}
    rates = {}
    for s in T09_SETS:
        f_, n_ = (fails.get(s) or {}).get("failed_or_no_move"), (fails.get(s) or {}).get("games")
        if not (_is_count(f_) and _is_count(n_) and n_ > 0 and 0 <= f_ <= n_) or f_ * 200 > n_:
            probs.append(f"P1 failures of {s}: {f_!r} of {n_!r} games (> 0.5 %: no 09)")
            rates[s] = None
        else:
            rates[s] = f_ / n_
    thr = max(0.5 * T09_RHO_MONO, T09_RHO_FLOOR)
    rho = {s: ((rep.get("p1_direction") or {}).get(s) or {}).get("rho") for s in T09_SETS}
    for s, v in rho.items():
        if not _is_num(v) or v < thr:
            probs.append(f"P1TO_E direction of {s}: rho {v!r} < {thr} (no 09)")
    return {"gpu_seam": vals, "p1_failure_rate": rates, "rho": rho, "rho_threshold": thr,
            "seam_local_max_abs_diff": sl.get("max_abs_diff"), "q08_max_abs_diff": q.get("max_abs_diff")}, probs


def t09_report_problems(rep, rel, path, sha):
    """The 09 apply report's own conditions (format, mode, exit, combo, check_submission, this file) and PREREG_09
    5.2 recomputed. -> (conditions, problems)."""
    probs = []
    if rep.get("format") != T09_FORMAT or rep.get("mode") != "real":
        probs.append(f"{rel}: format {rep.get('format')!r}, mode {rep.get('mode')!r} (a real-mode {T09_FORMAT})")
    if rep.get("exit_code") != 0 or rep.get("usable") is not True:
        probs.append(f"{rel}: exit_code {rep.get('exit_code')!r}, usable {rep.get('usable')!r}")
    combo = rep.get("combo") or {}
    if combo.get("id") != T09_COMBO_ID or tuple(combo.get("sources") or ()) != T09_COMBO_SOURCES:
        probs.append(f"{rel}: combination {combo.get('id')!r} {combo.get('sources')!r} is not Q09")
    fake = [s for s, r in (rep.get("p1_sets") or {}).items() if (r or {}).get("fake_from_val")]
    if fake or set(rep.get("p1_sets") or {}) != set(T09_P1_SETS):
        probs.append(f"{rel}: P1 sets {sorted(rep.get('p1_sets') or {})}, drill records {fake}")
    cs = rep.get("check_submission") or {}
    if cs.get("mode") != "official" or cs.get("errors"):
        probs.append(f"{rel}: check_submission {cs.get('mode')!r} with errors {cs.get('errors')}")
    want = (rep.get("outputs_sha256") or {}).get(FILE_NAMES["rank"])
    if want != sha:
        try:
            hint = lf_hint(path, want, repo_rel(path)) if path and os.path.exists(path) else ""
        except LogError:
            hint = ""
        probs.append(f"{rel}: records another {FILE_NAMES['rank']} than this file ({str(sha)[:12]})" + hint)
    cond, cprobs = t09_conditions(rep)
    return cond, probs + cprobs


def check_reports_t09(comp, path, sha):
    """PREREG_09 5.2 / 5.3: bind a 09 file to the committed t09_rank.py apply report next to it, its GPU seam and
    local seam records, test02's real R20 run, the full raw T1 (PASS or NORMALIZED), H8R and the 09 adoption, every
    committed input it lists and K6-09. Returns check_reports' shape plus 'test09'."""
    out, problems = {}, []
    rpath = os.path.join(t09_dir(), T09_REPORT)
    rel, rep = _committed_json(rpath, "the test09 apply report (tools/t09_rank.py apply, docs/TESTDAY_09.md)")
    out[rel] = sha256_file(rpath)
    cond, probs = t09_report_problems(rep, rel, path, sha)
    problems += probs
    for srel, what in ((T09_GPU_SEAM, "the test09 GPU seam record"), (T09_SEAM_LOCAL, "the test09 local seam record")):
        p = repo_path(srel)
        srel_, obj = _committed_json(p, what)
        out[srel_] = sha256_file(p)
        if obj.get("passed") is not True or obj.get("mode") != "real":
            problems.append(f"{srel_}: passed {obj.get('passed')!r}, mode {obj.get('mode')!r}")
    t2 = os.path.join(REPO, "results", "test02", "rank", REPORT_OF["rank"])
    t2rel, t2rep = _committed_json(t2, "the test02 rank apply report (R20 goes first, PREREG_09 5.3)")
    out[t2rel] = sha256_file(t2)
    if t2rep.get("mode") != "real" or t2rep.get("fallback_used") is not False \
            or (t2rep.get("seam_local") or {}).get("passed") is not True:
        problems.append(f"{t2rel}: mode {t2rep.get('mode')!r}, fallback_used {t2rep.get('fallback_used')!r}: 09 needs "
                        f"test02's passed real-mode R20 run (FAIL_OPENING: 09 never plays)")
    t1_path = os.path.join(REPO, "results", "test02", "checks", comp, "t1.json")
    trel, t1 = _committed_json(t1_path, "the T1 result")
    out[trel] = sha256_file(t1_path)
    problems += t1_problems(t1, trel)
    v = t1.get("verdict")
    if v not in T1_UPLOAD_OK:
        problems.append(f"{trel}: verdict {v!r}; 09 plays only with T1 PASS or NORMALIZED (PREREG_09 5.3)")
    problems += t08_h8r_problems()
    problems += t09_adoption_problems()
    problems += t09_inputs_problems(rep.get("inputs_bound"))
    h, t = git_commit_time(rel)
    k6_ok = t is not None and parse_time(t, "commit time") <= K6_09
    if not k6_ok:
        problems.append(f"{rel} was last committed at {t} ({h}), not by K6-09 {K6_09.isoformat()}: no 09 this season "
                        f"(PREREG_09 4); the final choice follows PREREG_08 5.4")
    if problems:
        raise LogError("the file is not backed by a usable 09 run (PREREG_09 5.2, 5.3):\n - " + "\n - ".join(problems))
    diffs = rep.get("rule_b_differences") or {}
    rec = dict(cond, combo=T09_COMBO_ID, adoption=P09_ADOPTED, report=rel, report_sha256=out[rel],
               k6_09={"commit": h, "committed_at": t, "deadline": K6_09.isoformat(), "ok": k6_ok},
               t08_report_sha256=(rep.get("inputs_bound") or {}).get(T09_08_REPORT),
               earliest_p1_chunk_created_utc=rep.get("earliest_p1_chunk_created_utc"),
               inputs_bound_checked=len(rep.get("inputs_bound") or {}),
               public_rule_b_differences={k: (diffs.get(k) or {}).get("public") for k in ("08", "07", "04", "R20")},
               rule_b_differences_all={k: (diffs.get(k) or {}).get("all") for k in ("08", "07", "04", "R20")})
    return {"reports": out, "t1_verdict": v, "fallback_used": False, "test09": rec}


def method_for_t09(given):
    """A 09 file is recorded as Q09, never with R20's, 04's, 07's or 08's label."""
    if given and not re.search(r"(?<![A-Za-z0-9])%s(?![0-9A-Za-z])" % T09_COMBO_ID, given):
        raise LogError(f"--method {given!r} does not name {T09_COMBO_ID}; a results/test09/rank file is the 09 "
                       f"combination (default {T09_METHOD!r})")
    return given or T09_METHOD


def t09_entries(log):
    return [x for x in log["uploads"] if x.get("comp") == "rank" and x.get("test09")]


def t09_bound_08(log, t9):
    """The uploaded 08 entry whose report is the one the 09 report binds (its t08_apply_report.json SHA-256, LF)."""
    want = t9.get("t08_report_sha256")
    for x in uploads_of(log, "rank"):
        t8_ = x.get("test08") or {}
        rp = repo_path(t8_.get("report") or "") if t8_.get("report") else ""
        if t8_ and rp and os.path.isfile(rp) and sha256_bound(rp) == want:
            return x
    return None


def prepare_order_t09(log, t9):
    """PREREG_09 5.3 order: R20's first upload and the 08 entry the 09 report binds are logged as uploaded (that 08
    still meets PREREG_08 5.4, which needs its 07 and 04 uploaded), and every P1 chunk was written after R20's first
    upload."""
    r20 = r20_first_upload(log)
    if r20 is None:
        raise LogError(f"PREREG_09 5.3: log R20's first upload ({T04_R20_FILE}, status uploaded) before preparing a 09 "
                       f"upload (docs/TESTDAY.md step 8)")
    e8 = t09_bound_08(log, t9)
    if e8 is None:
        raise LogError("PREREG_09 5.3: the 08 upload the 09 report binds (results/test08/rank/t08_apply_report.json) is "
                       "not logged as uploaded: upload 04, 07 and 08 first (docs/TESTDAY_04.md, docs/TESTDAY_07.md, "
                       "docs/TESTDAY_08.md)")
    ok8, why8 = t08_entry_eligible(e8, log)
    if not ok8:
        raise LogError(f"PREREG_09 5.3: the bound 08 entry #{e8['id']} no longer meets PREREG_08 5.4 "
                       f"({'; '.join(why8)[:600]}): 09 is not used")
    early = t9.get("earliest_p1_chunk_created_utc")
    if not early or parse_time(early, "P1 chunk time") <= parse_time(r20["uploaded_at"], "uploaded_at"):
        raise LogError(f"PREREG_09 5.2: P1 computed at {early}, not after R20's first upload ({r20['uploaded_at']})")
    t9["r20_first_upload_id"], t9["t08_entry_id"] = r20["id"], e8["id"]
    t9["t07_entry_id"] = (e8.get("test08") or {}).get("t07_entry_id")
    t9["t04_entry_id"] = (e8.get("test08") or {}).get("t04_entry_id")


def print_t09_ok_lines(t9):
    d = t9.get("public_rule_b_differences") or {}
    print(f"- 這一份是 09（Q09），不是 08（{T09_08_REPORT}）、07、04 也不是 R20（{T04_R20_FILE}）；public 題規則 B 等級和 08 "
          f"不同 {d.get('08')} 題、和 07 不同 {d.get('07')} 題、和 04 不同 {d.get('04')} 題、和 R20 不同 {d.get('R20')} 題")
    print(f"- 09 的方法：{T09_METHOD}；R20 的第一次上傳是 #{t9.get('r20_first_upload_id')}，綁定的 08 是 "
          f"#{t9.get('t08_entry_id')}（它綁的 07 是 #{t9.get('t07_entry_id')}、04 是 #{t9.get('t04_entry_id')}）；"
          f"{P09_ADOPTED}")
    g = t9.get("gpu_seam") or {}
    print("- 09 的檢查：GPU 接縫（標準化 EX {std_ex}、lp_P1TO_E {lp_p1e}）、Q08 對得上 08、本機接縫都通過；P1 失敗率 "
          .format(**{k: (f"{v:.3g}" if _is_num(v) else v) for k, v in g.items()})
          + "、".join(f"{s} {r:.2%}" for s, r in (t9.get("p1_failure_rate") or {}).items() if r is not None)
          + "；方向 ρ " + "、".join(f"{s} {v:.4f}" for s, v in (t9.get("rho") or {}).items() if _is_num(v)))
    print(f"- K6-09：09 的報告在 {t9['k6_09']['committed_at']} commit（期限 11/19 23:59）")
    print("- 注意：09 是沒在驗證集打過分的組合（PREREG_09 第 3 節的偏離），報告要寫明")


def t09_entry_eligible(x, log):
    """PREREG_09 5.3 for one logged 09 entry, re-checked now: H8R and the 09 adoption, the report's last commit is the
    one recorded at prepare and is by K6-09, T1 PASS / NORMALIZED, the recomputed 5.2 conditions, the report and the
    file unchanged, every bound input unchanged, the bound 08 entry still meets PREREG_08 5.4 with the same report, and
    the P1 chunks later than R20's first upload. -> (ok, reasons)."""
    t9, why = x.get("test09") or {}, []
    if x.get("t1_verdict") not in T1_UPLOAD_OK:
        why.append(f"T1 {x.get('t1_verdict')!r}")
    why += t08_h8r_problems()
    why += t09_adoption_problems()
    rel_ = t9.get("report") or ""
    rep = None
    rp = repo_path(rel_) if rel_ else ""
    if not rel_ or not os.path.isfile(rp) or sha256_file(rp) != t9.get("report_sha256"):
        why.append(f"{rel_ or 'the 09 apply report'} changed since prepare ({T09_REDO_HINT})")
    else:
        with open(refuse_forbidden(rp), encoding="utf-8") as f:
            rep = json.load(f)
    fp = repo_path(x.get("file") or "") if x.get("file") else ""
    if not fp or not os.path.isfile(fp) or sha256_file(fp) != x.get("sha256"):
        why.append(f"{x.get('file')} changed since prepare")
    k6 = t9.get("k6_09") or {}
    h, t = git_commit_time(rel_) if rel_ else (None, None)
    if not k6.get("ok") or h is None or h != k6.get("commit") or parse_time(t, "commit time") > K6_09:
        why.append(f"K6-09: the report's last commit is {h} at {t}; prepare recorded {k6.get('commit')} at "
                   f"{k6.get('committed_at')} (deadline {K6_09.isoformat()})")
    if rep is not None:
        why += t09_report_problems(rep, rel_, fp if fp and os.path.isfile(fp) else None, x.get("sha256"))[1]
        why += t09_inputs_problems(rep.get("inputs_bound"))
    e8 = next((y for y in log["uploads"] if y["id"] == t9.get("t08_entry_id")), None)
    if e8 is None or e8.get("status") != "uploaded":
        why.append(f"the bound 08 entry #{t9.get('t08_entry_id')} is not logged as uploaded")
    else:
        rp8 = repo_path((e8.get("test08") or {}).get("report") or "")
        if not os.path.isfile(rp8) or sha256_bound(rp8) != t9.get("t08_report_sha256"):
            why.append(f"the 08 report of #{e8['id']} is not the one the 09 report binds")
        ok8, why8 = t08_entry_eligible(e8, log)
        if not ok8:
            why.append(f"the bound 08 #{e8['id']} no longer meets PREREG_08 5.4: {'; '.join(why8)[:300]}")
    r20 = r20_first_upload(log)
    early = t9.get("earliest_p1_chunk_created_utc")
    if r20 is None or not early or parse_time(early, "P1 chunk time") <= parse_time(r20["uploaded_at"], "uploaded_at"):
        why.append(f"P1 chunks ({early}) are not later than R20's first upload ({(r20 or {}).get('uploaded_at')})")
    return not why, why


def t09_eligibility(log):
    return {x["id"]: t09_entry_eligible(x, log) for x in t09_entries(log)}


def head_has_p09_revision_row(n, head=None):
    head = git_head_text(P09_DOC) if head is None else head
    return bool(head) and any(line.startswith(f"| rev {n} |") for line in head.splitlines())


def p09_revision(reason, overridden, what):
    """Overriding a registered 09 decision (PREREG_09 5.3): a reason naming 'PREREG_09 rev N' (N >= 5) whose row is at
    HEAD of the committed, clean docs/PREREG_09_MAPPING_ENSEMBLE.md and was added after every overridden record."""
    text = (reason or "").strip()
    m = re.search(r"PREREG_09\s+rev\s+(\d+)", text)
    if not m:
        raise LogError(f"{what}: needs --not-09-reason \"PREREG_09 rev N: <the program error and the condition of "
                       f"PREREG_09 5.3 it breaks>\" citing a committed revision (N >= {P09_MIN_REV}); a reason "
                       f"without a revision is refused")
    n = int(m.group(1))
    if n < P09_MIN_REV:
        raise LogError(f"{what}: --not-09-reason cites PREREG_09 rev {n}; an override needs rev {P09_MIN_REV} or later "
                       f"(rev 3 and rev 4 are the test-day procedure itself)")
    st = git_file_state(P09_DOC)
    if not st["tracked"] or not st["clean"]:
        raise LogError(f"{what}: {P09_DOC} must be committed and unchanged (tracked={st['tracked']}, "
                       f"clean={st['clean']})")
    row = f"| rev {n} |"
    if not head_has_p09_revision_row(n):
        raise LogError(f"{what}: {P09_DOC} at HEAD has no row starting {row!r}; commit the revision first")
    h, t = git_row_added(P09_DOC, row)
    if h is None:
        raise LogError(f"{what}: no commit found that added {row!r} to {P09_DOC}")
    at = parse_time(t, "commit time")
    early = [w for w in overridden if at <= parse_time(w, "time")]
    if early:
        raise LogError(f"{what}: PREREG_09 rev {n} was committed at {t} ({h[:12]}), not after the records it "
                       f"overrides ({', '.join(early)}); a revision must come after the decision it overturns")
    return {"rev": n, "row": row, "doc": P09_DOC, "commit": h, "committed_at": t, "reason": text,
            "overridden_at": list(overridden)}


def prepare_rule_t09(log, comp, bound, not_09_reason):
    """PREREG_09 5.3 at prepare: an 08 / 07 / 04 / R20 / R10 file while a logged 09 entry meets 5.3 needs the revision
    check of --not-09-reason. None when the log has no 09 entry (everything else is exactly as before)."""
    if bound.get("test14"):              # PREREG_14 7: a 14 file is judged on its base in prepare_order_t14
        if (not_09_reason or "").strip():
            raise LogError("--not-09-reason does not apply to a 14 file")
        return None
    reason = (not_09_reason or "").strip()
    if bound.get("test25"):              # PREREG_25 rev 3: a 25 file is judged by prepare_rule_t25
        if reason:
            raise LogError("--not-09-reason does not apply to a 25 file")
        return None
    if bound.get("test19"):              # PREREG_19 rev 3: a 19 file is judged by prepare_rule_t19
        if reason:
            raise LogError("--not-09-reason applies to a competition-2 08 / 07 / 04 / R20 / R10 file only")
        return None
    if comp != "rank" or bound.get("test09"):
        if reason:
            raise LogError("--not-09-reason applies to a competition-2 08 / 07 / 04 / R20 / R10 file only")
        return None
    elig = t09_eligibility(log)
    if not elig:
        if reason:
            raise LogError("--not-09-reason: the log has no 09 entry to set aside")
        return None
    ok_ids = [i for i, (ok, _w) in elig.items() if ok]
    rec = {"rule": "PREREG_09 5.3", "adoption": P09_ADOPTED, "eligible_09_entries": ok_ids,
           "not_eligible_09_entries": {str(i): w for i, (ok, w) in elig.items() if not ok}, "checked_at": now_iso(),
           "not_09_reason": reason or None}
    if ok_ids:
        when = [x["prepared_at"] for x in t09_entries(log) if x["id"] in ok_ids]
        rec["revision"] = p09_revision(reason, when, f"09 entries {ok_ids} meet PREREG_09 5.3, so an 08 / 07 / 04 / "
                                                     f"R20 / R10 file is not the final choice")
    return rec


def final_rule_t09(log, e, not_09_reason):
    """PREREG_09 5.3 for competition 2's final choice. None when no 09 entry is logged. A 09 entry is final only
    while it holds; while some 09 entry holds, an 08 / 07 / 04 / R20 / R10 final needs --not-09-reason
    (p09_revision)."""
    reason = (not_09_reason or "").strip()
    if e.get("test25"):                  # PREREG_25 rev 3: a 25 final is judged by final_rule_t25
        if reason:
            raise LogError("--not-09-reason does not apply to a 25 entry")
        return None
    if e.get("test19"):                  # PREREG_19 rev 3: a 19 final is judged by final_rule_t19, never as 09
        if reason:
            raise LogError("--not-09-reason applies to an 08 / 07 / 04 / R20 / R10 final choice, not to a 19 entry")
        return None
    if e["comp"] != "rank":
        if reason:
            raise LogError("--not-09-reason applies to competition 2 (rank) only")
        return None
    elig = t09_eligibility(log)
    if not elig:
        if reason:
            raise LogError("--not-09-reason: the log has no 09 entry to set aside")
        return None
    ok_ids = [i for i, (ok, _w) in elig.items() if ok]
    rec = {"rule": "PREREG_09 5.3", "adoption": P09_ADOPTED, "eligible_09_entries": ok_ids,
           "not_eligible_09_entries": {str(i): w for i, (ok, w) in elig.items() if not ok}, "checked_at": now_iso()}
    if e.get("test09"):
        ok, why = elig[e["id"]]
        if not ok:
            raise LogError(f"entry #{e['id']} (09) does not meet PREREG_09 5.3 now ({'; '.join(why)[:900]}): the final "
                           f"choice follows PREREG_08 5.4 (08, else 07, else 04, else R20, R10 on FAIL_OPENING); "
                           f"upload it again so that it is the last upload (docs/TESTDAY_09.md step 5)")
        if reason:
            raise LogError("--not-09-reason does not apply when the 09 entry is chosen")
        return dict(rec, chosen="09", not_09_reason=None)
    rec.update(chosen="08" if e.get("test08") else "07" if e.get("test07") else "04" if e.get("test04") else
               ("R10" if e.get("fallback_used") else "R20"), not_09_reason=reason or None)
    if ok_ids:
        when = [x["prepared_at"] for x in t09_entries(log) if x["id"] in ok_ids]
        rec["revision"] = p09_revision(reason, when, f"09 entries {ok_ids} meet every condition of PREREG_09 5.3, "
                                                     f"so the final choice is 09")
    return rec


def t09_final_holds(log, fin):
    """The final choice is a 09 entry recorded as chosen 09 that still meets PREREG_09 5.3 (False without 09
    entries)."""
    if not fin or not t09_entries(log):
        return False
    e = next((x for x in log["uploads"] if x["id"] == fin.get("id")), None)
    if not e or not e.get("test09") or (fin.get("prereg09_5_3") or {}).get("chosen") != "09":
        return False
    return t09_entry_eligible(e, log)[0]


def t09_status_key(log):
    """{'t09_eligible': [...]} for t04_status / t07_status / t08_status, only when the log has 09 entries (else {})."""
    if not t09_entries(log):
        return {}
    return {"t09_eligible": [i for i, (ok, _w) in t09_eligibility(log).items() if ok]}


def t09_status(log, fin, last):
    elig = t09_eligibility(log)
    ok_ids = [i for i, (ok, _w) in elig.items() if ok]
    by_id = {x["id"]: x for x in log["uploads"]}
    fin_e = by_id.get((fin or {}).get("id"))
    rule = (fin or {}).get("prereg09_5_3") or {}
    final_is_09 = bool(fin_e and fin_e.get("test09"))
    rev = rule.get("revision") or {}

    def overridden_by_rev(i):
        """A non-09 final sets a 09 entry aside only when the revision recorded with it covered that entry and was
        committed after its prepare (the time rule of final, re-checked now; as t08_status)."""
        prep = (by_id.get(i) or {}).get("prepared_at")
        if i not in (rule.get("eligible_09_entries") or []) or not rev.get("committed_at") or not prep:
            return False
        try:
            return parse_time(rev["committed_at"], "revision commit time") > parse_time(prep, "prepared_at")
        except LogError:
            return False
    final_ok = (final_is_09 and fin_e["id"] in ok_ids and rule.get("chosen") == "09") or \
        (bool(fin) and not final_is_09 and bool(rev) and all(overridden_by_rev(i) for i in ok_ids))
    final_ok = final_ok or t19_final_holds(log, fin)   # PREREG_19 4.3: a holding 19 final is not a 09 violation
    final_ok = final_ok or t25_final_holds(log, fin)   # PREREG_25 4.3: a holding 25 final is not a violation
    last_bad = bool(last and last.get("test09") and not elig.get(last["id"], (True, []))[0])
    final_bad = bool(final_is_09 and not elig[fin_e["id"]][0])
    return {"entries": sorted(elig), "eligible": ok_ids, "not_eligible": {str(i): w for i, (ok, w) in elig.items()
                                                                          if not ok},
            "final_set": bool(fin), "final_is_09": final_is_09, "violation": bool(ok_ids) and bool(fin) and not final_ok,
            "redo_08": not ok_ids and (last_bad or final_bad)}


def t09_status_lines(st):
    lines = [f"  09（PREREG_09 5.3）：紀錄 {st['entries']}；現在符合條件的 {st['eligible'] or '沒有'}；最終選擇"
             f"{'是 09' if st['final_is_09'] else '不是 09'}"]
    for i, why in st["not_eligible"].items():
        text = "；".join(why)
        lines.append(f"  #{i} 不符合：{text[:600]}{'…' if len(text) > 600 else ''}")
    if st["violation"]:
        lines.append("  違反 PREREG_09 5.3：有符合條件的 09，最終選擇卻不是它（沒有引用修訂，或引用的修訂沒有涵蓋它）；照 "
                     "docs/TESTDAY_09.md 第 5 步重跑 final")
    if st["redo_08"]:
        lines.append("  照 PREREG_09 5.3：現在沒有符合的 09，final 改照 PREREG_08 5.4（08，否則 07，否則 04，否則 R20），再 prepare "
                     "並重傳那一份（docs/TESTDAY_09.md 第 5 步）")
    return lines


def t09_public_lines(log, e):
    """PREREG_09 5.3: 09, 08, 07, 04 and R20 share submission_rank.csv. Two equal public scores (|d| < 1e-9) between a
    09 entry and an 08 / 07 / 04 / R20 entry while the 09 report counts public-question rule-B differences > 0 between
    them: a hint that the wrong file was uploaded. Empty when the log has no 09 entry."""
    if e["comp"] != "rank" or e.get("public_score") is None or not t09_entries(log):
        return []
    lines = []
    others = [x for x in uploads_of(log, "rank") if x["id"] != e["id"] and x.get("public_score") is not None
              and abs(x["public_score"] - e["public_score"]) < 1e-9]
    for x in others:
        a, b = (e, x) if e.get("test09") else (x, e)
        if not a.get("test09") or b.get("test09"):
            continue
        if b.get("test25"):              # PREREG_25 rev 3: a 25 entry is warned by t25_public_lines
            continue
        if b.get("test19"):              # PREREG_19 rev 3: a 19 entry is warned by t19_public_lines
            continue
        if b.get("test14"):              # PREREG_14: a 14 entry is warned by t14_public_lines
            continue
        kind = "08" if b.get("test08") else "07" if b.get("test07") else "04" if b.get("test04") else "R20"
        n = ((a.get("test09") or {}).get("public_rule_b_differences") or {}).get(kind)
        if _is_count(n) and n > 0:
            lines.append(f"WARNING (PREREG_09 5.3): #{a['id']}（09）和 #{b['id']}（{kind}）的 public 分數完全相同，但兩份在 "
                         f"public 題有 {n} 題的規則 B 等級不同：可能傳錯檔（五份同名）：核對 AIdea 的上傳紀錄")
    return lines


# ---------------------------------------------------------------- PREREG_10 rev 3: test10 (competition 1)
def t10_dir():
    return os.path.join(REPO, *T10_DIR)


def is_t10_file(comp, path):
    """A competition-1 file in results/test10/player/ is a 10 file (only tools/t10_player.py submit writes there)."""
    if comp != "player":
        return False
    here = os.path.normcase(os.path.realpath(os.path.dirname(os.path.abspath(path))))
    return here == os.path.normcase(os.path.realpath(t10_dir()))


def head_has_p10_revision_row(n, head=None):
    head = git_head_text(P10_DOC) if head is None else head
    return bool(head) and any(line.startswith(f"| rev {n} |") for line in head.splitlines())


def p10_revision(reason, overridden, what):
    """PREREG_10 4.4 (same definition as p06_revision): another competition-1 choice than the designated one needs
    --not-10-reason naming 'PREREG_10 rev N' (N >= 2) while docs/PREREG_10_PLAYER_GUARD_P1.md is committed and clean,
    HEAD has a row starting '| rev N |', and the commit that added the row is later than every overridden record."""
    text = (reason or "").strip()
    m = re.search(r"PREREG_10\s+rev\s+(\d+)", text)
    if not m:
        raise LogError(f"{what}: needs --not-10-reason \"PREREG_10 rev N: <the program error and the condition of "
                       f"PREREG_10 4.4 it breaks>\" citing a committed revision (N >= {P10_MIN_REV}); a reason "
                       f"without a revision is refused (PREREG_10 4.4)")
    n = int(m.group(1))
    if n < P10_MIN_REV:
        raise LogError(f"{what}: --not-10-reason cites PREREG_10 rev {n}; an override needs rev {P10_MIN_REV} or "
                       f"later (PREREG_10 4.4)")
    st = git_file_state(P10_DOC)
    if not st["tracked"] or not st["clean"]:
        raise LogError(f"{what}: {P10_DOC} must be committed and unchanged (tracked={st['tracked']}, "
                       f"clean={st['clean']})")
    row = f"| rev {n} |"
    if not head_has_p10_revision_row(n):
        raise LogError(f"{what}: {P10_DOC} at HEAD has no row starting {row!r}; commit the revision first")
    h, t = git_row_added_any(P10_DOC, row)
    if h is None:
        raise LogError(f"{what}: no commit found that added {row!r} to {P10_DOC}")
    at = parse_time(t, "commit time")
    early = [w for w in overridden if at <= parse_time(w, "time")]
    if early:
        raise LogError(f"{what}: PREREG_10 rev {n} was committed at {t} ({h[:12]}), not after the records it "
                       f"overrides ({', '.join(early)}); a revision must come after the decision it overturns")
    return {"rev": n, "row": row, "doc": P10_DOC, "commit": h, "committed_at": t, "reason": text,
            "overridden_at": list(overridden)}


def git_row_added_any(rel, row):
    """(commit, committer time ISO) of the first commit that added the text row to rel, merge commits included
    (git log --reverse -m -S: a merge is diffed against each parent, so a row that first appears while resolving a
    merge is found; git_row_added skips merges). (None, None) when there is none. PREREG_10 only (rev 3: the test-day
    row first appears in the merge commit that brought main into p10-testday)."""
    rc, out = git("log", "--reverse", "-m", "-S" + row, "--format=%H %cI", "--", rel)
    if rc != 0 or not out:
        return None, None
    h, t = out.splitlines()[0].split()
    return h, t


def t10_k5_state():
    """K5-10: the PREREG_10 test-day revision row is in HEAD and was added by a commit at or before K5-10."""
    head = git_head_text(P10_DOC)
    in_head = bool(head) and any(line.startswith(P10_TESTDAY_ROW) for line in head.splitlines())
    h, t = git_row_added_any(P10_DOC, P10_TESTDAY_ROW)
    ok = bool(in_head and h is not None and parse_time(t, "commit time") <= K5_10)
    return {"row": P10_TESTDAY_ROW, "commit": h, "committed_at": t, "in_head": in_head, "deadline": K5_10.isoformat(),
            "ok": ok}


def t10_inputs_problems(bound, need_t4=False):
    """Every input the 10 submit report lists under inputs_bound is committed, clean and still has the recorded SHA-256
    (a re-run of test02 or 06 after the 10 run invalidates it, PREREG_10 4.2)."""
    if not isinstance(bound, dict) or not bound:
        return [f"the 10 submit report lists no inputs_bound ({T10_REDO_HINT})"]
    probs = []
    need = list(T10_INPUTS_REQUIRED) + (["results/test02/checks/player/t4.json"] if need_t4 else [])
    missing = [r for r in need if r not in bound]
    if missing:
        probs.append(f"inputs_bound lacks {missing}")
    for rel_, want in sorted(bound.items()):
        s = str(rel_)
        if os.path.isabs(s) or "\\" in s or ".." in s.split("/") or s.startswith("/"):
            probs.append(f"inputs_bound key {s!r} is not a repo-relative path")
            continue
        p = repo_path(s)
        if not os.path.isfile(p):
            probs.append(f"{s} is missing ({T10_REDO_HINT})")
            continue
        st = git_file_state(s)
        if not st["tracked"] or not st["clean"]:
            probs.append(f"{s} is not committed and clean (先 commit；內容若已改變：{T10_REDO_HINT})")
            continue
        got = sha256_bound(p)
        if got != want:
            probs.append(f"{s} is {got[:12]}, the 10 submit report bound {str(want)[:12]} ({T10_REDO_HINT})")
    return probs


def t10_report_problems(rep, rel, path, sha):
    """The 10 submit report's own conditions (format, mode, method, check_submission, this file, the seams, H6P, T1,
    the pool sources: at least one 10A, no 10B while B is not adopted). -> problems."""
    probs = []
    if rep.get("format") != T10_FORMAT or rep.get("mode") != "real":
        probs.append(f"{rel}: format {rep.get('format')!r}, mode {rep.get('mode')!r} (a real-mode {T10_FORMAT})")
    if rep.get("method") != T10_METHOD:
        probs.append(f"{rel}: method {rep.get('method')!r} ({T10_METHOD})")
    cs = rep.get("check_submission") or {}
    if cs.get("official") is not True or cs.get("passed") is not True:
        probs.append(f"{rel}: check_submission official={cs.get('official')!r} passed={cs.get('passed')!r}")
    want = (rep.get("output") or {}).get("sha256")
    if want != sha:
        try:
            hint = lf_hint(path, want, repo_rel(path)) if path and os.path.exists(path) else ""
        except LogError:                                  # a file outside the repository (a dry-run tree)
            hint = ""
        probs.append(f"{rel}: records another {FILE_NAMES['player']} than this file ({str(sha)[:12]})" + hint)
    if (rep.get("seam") or {}).get("passed") is not True:
        probs.append(f"{rel}: the 10 seam it records did not pass")
    if (rep.get("t06_seam") or {}).get("passed") is not True:
        probs.append(f"{rel}: the 06 seam it records did not pass")
    if (rep.get("h6p") or {}).get("adopted") is not True:
        probs.append(f"{rel}: H6P is not recorded as adopted (PREREG_10 1.1 item 4)")
    if (rep.get("t1") or {}).get("verdict") not in T1_UPLOAD_OK:
        probs.append(f"{rel}: T1 {(rep.get('t1') or {}).get('verdict')!r} (PREREG_10 1.1 item 1)")
    if not rep.get("run_id"):
        probs.append(f"{rel}: no run_id")
    src = rep.get("sources") or {}
    if set(src) != {"public", "private"} or any(v not in T10_SOURCES for v in src.values()):
        probs.append(f"{rel}: pool sources {src!r} (each one of {T10_SOURCES}; 10B needs PREREG_10 B adopted)")
    elif "10A" not in src.values():
        probs.append(f"{rel}: no pool was lifted (10A): such a 10 equals 06-OFF / Q04 and is never produced")
    if (rep.get("b") or {}).get("adopted") is not False:
        probs.append(f"{rel}: B recorded as {(rep.get('b') or {}).get('adopted')!r}; this kit has part A only")
    return probs


# ---- PREREG_10 rev 4 (test10 part B, competition 1); added lines: the rev 3 lines above are unchanged. A 10 report
# whose B is recorded as adopted is checked by the rev 3 rules with its 10B pools counted as lifted pools, plus the B
# rules below; a report without B (b.adopted false) goes through the rev 3 function exactly as before.
T10_SOURCES_B = ("06-OFF", "10A", "10B", "test02")         # 10B: PREREG_10 B adopted (rev 4)
P10_B_ROW = "| rev 4 |"                                    # B's test-day revision of PREREG_10 (K5-10 for B)
K6B_10 = datetime(2026, 11, 14, 23, 59, 59, tzinfo=TW)    # PREREG_10 3: both pools' P1 and the GPU seam committed
T10_B_FAIL_MAX = 0.005                                     # PREREG_10 4.2: P1 failed games <= 0.5 % per pool
T10_P1_DIR = "results/test10/p1"
T10_B_INPUTS_REQUIRED = ("results/prereg10/RESULTS.md", "results/prereg10/player/frozen_config.json",
                         "results/prereg10/player/p1_lda.npz", "results/prereg10/player/score_H10P.json",
                         "results/prereg10/feat/val_seam_feats.npz", T10_P1_DIR + "/p1_rehearsal.json",
                         T10_P1_DIR + "/gpu_seam.json")
_t10_report_problems_rev3 = t10_report_problems


def git_row_added_on_main(rel, row):
    """(commit, committer time) of the first commit in main's first-parent history whose diff against its first parent
    adds the text row (a merge into main counts at its own time). (None, None) when there is none."""
    rc, out = git("log", "main", "--first-parent", "--diff-merges=first-parent", "--reverse", "-S" + row,
                  "--format=%H %cI", "--", rel)
    if rc != 0 or not out:
        return None, None
    h, t = out.splitlines()[0].split()
    return h, t


def t10_k5_b_state():
    """K5-10 for B (PREREG_10 rev 4 section 14): the rev 4 row is in HEAD and reached main (first-parent history) at or
    before K5-10."""
    head = git_head_text(P10_DOC)
    in_head = bool(head) and any(line.startswith(P10_B_ROW) for line in head.splitlines())
    h, t = git_row_added_on_main(P10_DOC, P10_B_ROW)
    ok = bool(in_head and h is not None and parse_time(t, "commit time") <= K5_10)
    return {"row": P10_B_ROW, "commit": h, "committed_at": t, "in_head": in_head, "deadline": K5_10.isoformat(),
            "ok": ok}


def t10_b_problems(rep, rel):
    """PREREG_10 rev 4 for a 10 report with B adopted: K5-10 for B; the GPU seam passed with a tol_P from the committed
    rehearsal; each 10B pool's B record applied (pool gate passed, P1 failed <= 0.5 %), its P1 report and the GPU
    seam last committed by K6B-10; the B files bound (inputs_bound). -> problems."""
    probs = []
    b = rep.get("b") or {}
    src = rep.get("sources") or {}
    if set(src) != {"public", "private"} or any(v not in T10_SOURCES_B for v in src.values()):
        probs.append(f"{rel}: pool sources {src!r} (each one of {T10_SOURCES_B})")
    k5 = t10_k5_b_state()
    if not k5["ok"]:
        probs.append(f"K5-10 (B): {P10_DOC} row {P10_B_ROW!r} in HEAD {k5['in_head']}, added by {k5['commit']} at "
                     f"{k5['committed_at']} (deadline {K5_10.isoformat()}): B is not used this season (PREREG_10 3)")
    seam = b.get("gpu_seam") or {}
    tol = b.get("tol") or {}
    if seam.get("passed") is not True:
        probs.append(f"{rel}: the B GPU seam it records did not pass")
    if not isinstance(tol.get("tol_P"), (int, float)) or not isinstance(b.get("w_star"), (int, float)):
        probs.append(f"{rel}: B without tol_P (the committed rehearsal) or w*")
    bound = rep.get("inputs_bound") or {}
    need = list(T10_B_INPUTS_REQUIRED)
    for pool in ("public", "private"):
        if src.get(pool) != "10B":
            continue
        pb = ((rep.get("pools") or {}).get(pool) or {}).get("b") or {}
        gate = pb.get("pool_gate") or {}
        if pb.get("applied") is not True or gate.get("passed") is not True:
            probs.append(f"{rel}: {pool} is 10B but its B record is applied={pb.get('applied')!r}, pool gate "
                         f"{gate.get('passed')!r}")
        ff = pb.get("failed_fraction")
        if not isinstance(ff, (int, float)) or ff > T10_B_FAIL_MAX:
            probs.append(f"{rel}: {pool}: P1 failed share {ff!r} (> 0.5 %: no B)")
        need.append(f"{T10_P1_DIR}/{pool}_p1_report.json")
    for r in need:
        if r not in bound:
            probs.append(f"{rel}: inputs_bound lacks the B file {r}")
    for r in [f"{T10_P1_DIR}/gpu_seam.json"] + [f"{T10_P1_DIR}/{pool}_p1_report.json" for pool in ("public", "private")
                                                 if src.get(pool) == "10B"]:
        h, t = git_commit_time(r)
        if h is None or parse_time(t, "commit time") > K6B_10:
            probs.append(f"{r} was last committed at {t} ({h}), not by K6B-10 {K6B_10.isoformat()}: no B this season "
                         f"(PREREG_10 3)")
    return probs


def t10_report_problems(rep, rel, path, sha):
    """Rev 4 wrapper of the rev 3 function above (same name: every caller resolves to this one). The B rules apply only
    when some pool is 10B; a report with B adopted but no 10B pool (B not used in either pool) is an A report."""
    b = rep.get("b") or {}
    src = rep.get("sources") or {}
    if b.get("adopted") is not True and "10B" not in src.values():
        return _t10_report_problems_rev3(rep, rel, path, sha)
    as_a = dict(rep, b={"adopted": False}, sources={k: ("10A" if v == "10B" else v) for k, v in src.items()})
    probs = _t10_report_problems_rev3(as_a, rel, path, sha)
    if "10B" in src.values():
        if b.get("adopted") is not True:
            probs.append(f"{rel}: a 10B pool while B is recorded as not adopted (10B needs PREREG_10 B adopted)")
        probs += t10_b_problems(rep, rel)
    return probs


# ---- PREREG_10 rev 5 (section 16; PREREG_13 8.2): B adopted through PREREG_13. Added lines: the rev 3 and rev 4 lines
# above are unchanged. A report whose b.source is not 'PREREG_13' goes through the rev 4 function exactly as before;
# with b.source 'PREREG_13' and a 10B pool, inputs_bound needs results/prereg13's RESULTS.md, score_H13P.json and
# view_report.json instead of results/prereg10/player/score_H10P.json, and the rev 5 row must have reached main by K5-13.
P10_B13_ROW = "| rev 5 |"                                  # PREREG_10 rev 5 (K5-13 for the PREREG_13 path)
K5_13 = datetime(2026, 10, 31, 23, 59, 59, tzinfo=TW)     # PREREG_13 7: rev 5 and the t10 change merged into main
T10_B_H10P_FILE = "results/prereg10/player/score_H10P.json"
T10_B13_INPUTS_REQUIRED = ("results/prereg13/RESULTS.md", "results/prereg13/player/score_H13P.json",
                           "results/prereg13/player/view_report.json")
_t10_report_problems_rev4 = t10_report_problems


def t10_k5_b13_state():
    """K5-13 (PREREG_10 rev 5, 16.2): the rev 5 row is in HEAD and reached main (first-parent history) by K5-13."""
    head = git_head_text(P10_DOC)
    in_head = bool(head) and any(line.startswith(P10_B13_ROW) for line in head.splitlines())
    h, t = git_row_added_on_main(P10_DOC, P10_B13_ROW)
    ok = bool(in_head and h is not None and parse_time(t, "commit time") <= K5_13)
    return {"row": P10_B13_ROW, "commit": h, "committed_at": t, "in_head": in_head, "deadline": K5_13.isoformat(),
            "ok": ok}


def t10_report_problems(rep, rel, path, sha):
    """Rev 5 wrapper of the rev 4 function above (same name: every caller resolves to this one)."""
    b = rep.get("b") or {}
    src = rep.get("sources") or {}
    probs = _t10_report_problems_rev4(rep, rel, path, sha)
    if b.get("source") != "PREREG_13" or "10B" not in src.values():
        return probs
    h10p_missing = f"{rel}: inputs_bound lacks the B file {T10_B_H10P_FILE}"
    probs = [x for x in probs if x != h10p_missing]
    bound = rep.get("inputs_bound") or {}
    for r in T10_B13_INPUTS_REQUIRED:
        if r not in bound:
            probs.append(f"{rel}: inputs_bound lacks the B file {r} (B adopted through PREREG_13)")
    if b.get("adopted") is True:
        k5 = t10_k5_b13_state()
        if not k5["ok"]:
            probs.append(f"K5-13: {P10_DOC} row {P10_B13_ROW!r} in HEAD {k5['in_head']}, added on main by "
                         f"{k5['commit']} at {k5['committed_at']} (deadline {K5_13.isoformat()}): B through PREREG_13 "
                         f"is not used this season (PREREG_13 7)")
    return probs


def check_reports_t10(comp, path, sha):
    """PREREG_10 4.2-4.4: bind a 10 file to the committed tools/t10_player.py submit report next to it, the 10 seam and
    pool reports, the 06 seam, test02's competition-1 reports (submit, local seam, the full raw T1 with PASS or
    NORMALIZED), the committed inputs it lists, K6-10 and K5-10. Returns check_reports' shape plus 'test10'."""
    out, problems = {}, []
    rpath = os.path.join(t10_dir(), T10_REPORT)
    rel, rep = _committed_json(rpath, "the 10 submit report (tools/t10_player.py submit, docs/TESTDAY_10.md)")
    out[rel] = sha256_file(rpath)
    problems += t10_report_problems(rep, rel, path, sha)
    for srel, what, fmt in ((T10_SEAM, "the 10 seam record", T10_SEAM_FORMAT), (T06_SEAM, "the 06 seam record",
                                                                               T06_SEAM_FORMAT)) + tuple(
            (p, "a 10 pool report", T10_POOL_FORMAT) for p in T10_POOL_REPORTS):
        p = repo_path(srel)
        srel_, obj = _committed_json(p, what)
        out[srel_] = sha256_file(p)
        if obj.get("format") != fmt or obj.get("mode") != "real":
            problems.append(f"{srel_}: format {obj.get('format')!r}, mode {obj.get('mode')!r} (a real-mode {fmt})")
        if fmt in (T10_SEAM_FORMAT, T06_SEAM_FORMAT) and obj.get("passed") is not True:
            problems.append(f"{srel_}: the seam did not pass (verdict {obj.get('verdict')!r})")
    srel, sub = _committed_json(repo_path(T06_T02_REPORTS[0]), "test02's competition-1 submit report (Q04 first)")
    out[srel] = sha256_file(repo_path(T06_T02_REPORTS[0]))
    cs = sub.get("check_submission") or {}
    if sub.get("mode") != "real" or cs.get("official") is not True or cs.get("passed") is not True:
        problems.append(f"{srel}: mode {sub.get('mode')!r}, check_submission official {cs.get('official')!r} passed "
                        f"{cs.get('passed')!r}: 10 needs test02's passed real-mode Q04 run")
    lrel, seam = _committed_json(repo_path(T06_T02_REPORTS[1]), "test02's competition-1 local seam")
    out[lrel] = sha256_file(repo_path(T06_T02_REPORTS[1]))
    if seam.get("passed") is not True or seam.get("mode") != "real":
        problems.append(f"{lrel}: passed={seam.get('passed')!r}, mode={seam.get('mode')!r}")
    trel, t1 = _committed_json(repo_path(T06_T02_REPORTS[2]), "the T1 result")
    out[trel] = sha256_file(repo_path(T06_T02_REPORTS[2]))
    problems += t1_problems(t1, trel)
    v = t1.get("verdict")
    if v not in T1_UPLOAD_OK:
        problems.append(f"{trel}: verdict {v!r}; 10 plays only with T1 PASS or NORMALIZED (PREREG_10 1.1 item 1)")
    a_holds = rep.get("a_holds") or {}
    problems += t10_inputs_problems(rep.get("inputs_bound"), need_t4=any(a_holds.values()))
    h, t = git_commit_time(rel)
    k6_ok = t is not None and parse_time(t, "commit time") <= K6_10
    if not k6_ok:
        problems.append(f"{rel} was last committed at {t} ({h}), not by K6-10 {K6_10.isoformat()}: no 10 this season "
                        f"(PREREG_10 3); the final choice follows PREREG_06 5.4")
    k5 = t10_k5_state()
    if not k5["ok"]:
        problems.append(f"K5-10: {P10_DOC} row {P10_TESTDAY_ROW!r} in HEAD {k5['in_head']}, added by {k5['commit']} at "
                        f"{k5['committed_at']} (deadline {K5_10.isoformat()}): A did not make K5-10, no 10 this "
                        f"season (PREREG_10 3)")
    if problems:
        raise LogError("the file is not backed by a usable 10 run (PREREG_10 4.2-4.4):\n - " + "\n - ".join(problems))
    off = rep.get("t06_off") or {}
    rec = {"label": "10", "run_id": rep.get("run_id"), "report": rel, "report_sha256": out[rel],
           "k6_10": {"commit": h, "committed_at": t, "deadline": K6_10.isoformat(), "ok": k6_ok}, "k5_10": k5,
           "sources": rep.get("sources"), "reference": rep.get("reference"),
           "t06_off_exists": bool(off.get("exists")), "t06_off_run_id": off.get("run_id"),
           "t06_off_sha256": off.get("sha256"), "t06_on_produced": bool(off.get("on_produced")),
           "top1_differences_vs_06_off": rep.get("top1_differences_vs_06_off"),
           "top1_differences_vs_q04": rep.get("top1_differences_vs_q04"),
           "inputs_bound_checked": len(rep.get("inputs_bound") or {})}
    return {"reports": out, "t1_verdict": v, "fallback_used": None, "test10": rec}


def method_for_t10(given):
    """A 10 file is recorded under its own method; a --method that does not name prereg10 is refused."""
    if given and "prereg10" not in given:
        raise LogError(f"--method {given!r} does not name prereg10; a results/test10/player file is {T10_METHOD!r}")
    return given or T10_METHOD


def t10_entries(log):
    return [x for x in log["uploads"] if x.get("comp") == "player" and x.get("test10")]


def t10_on_allows(log, now):
    """PREREG_10 4.4: with a 06-ON produced, 10 is uploaded only when 06-ON is not the designated final upload
    (vetoed, wrong upload, check inconclusive, K7' passed, or 06-ON not eligible). -> (ok, why)."""
    st = t06_state(log, now)
    ons = [x for x in uploads_of(log, "player") if (x.get("test06") or {}).get("variant") == "on"]
    if st["designated"] == "on":
        return False, "06-ON is kept (PREREG_06 5.3 / 5.4: d >= 0): final 06-ON, no 10 (PREREG_10 4.4)"
    if st["designated"] == "undecided":
        return False, f"06-ON waits for its veto ({st['rule']}): decide it first (docs/TESTDAY_06.md step 7)"
    if not ons and now <= K7P:
        return False, ("06-ON was produced but not uploaded: PREREG_06 5.3 uploads it and decides the veto first "
                       "(or K7' passes)")
    return True, st["rule"]


def prepare_order_t10(log, t10):
    """PREREG_10 4.4 at prepare: Q04's first upload is logged; this season's 06-OFF (when there is one) is uploaded;
    with a 06-ON produced, 06-ON is not kept."""
    q04 = q04_first_upload(log)
    if q04 is None:
        raise LogError(f"PREREG_10 4.4: log Q04's first upload ({T06_Q04_FILE}, status uploaded) before preparing a "
                       f"10 upload (docs/TESTDAY.md step 8, then docs/TESTDAY_10.md)")
    t10["q04_first_upload_id"] = q04["id"]
    if t10.get("t06_off_exists"):
        offs = [x for x in uploads_of(log, "player") if (x.get("test06") or {}).get("variant") == "off"
                and x["test06"].get("run_id") == t10.get("t06_off_run_id") and x["sha256"] == t10.get("t06_off_sha256")]
        if not offs:
            raise LogError("PREREG_10 4.4: upload this season's 06-OFF (the one the 10 report was compared with) "
                           "first (docs/TESTDAY_06.md), then 10")
        t10["off_entry_id"] = offs[-1]["id"]
    if t10.get("t06_on_produced"):
        ok, why = t10_on_allows(log, t06_now())
        if not ok:
            raise LogError(f"PREREG_10 4.4: {why}")
    return t10


def print_t10_ok_lines(t10):
    d, q = t10.get("top1_differences_vs_06_off") or {}, t10.get("top1_differences_vs_q04") or {}
    print(f"- 這一份是 10（{T10_METHOD}），各池來源 {t10.get('sources')}；和 Q04 第 1 名不同的題數：public {q.get('public')}、"
          f"private {q.get('private')}" + (f"；和 06-OFF（#{t10.get('off_entry_id')}）不同：public {d.get('public')}、"
                                           f"private {d.get('private')}" if t10.get("t06_off_exists") else "")
          + f"；Q04 的第一次上傳是 #{t10.get('q04_first_upload_id')}")
    print(f"- 10 的檢查：10 和 06 的 seam、test02 的接縫都通過；K6-10：10 的報告在 {t10['k6_10']['committed_at']} commit"
          f"（期限 11/21 23:59）；K5-10 成立")


def t10_entry_eligible(x):
    """PREREG_10 4.4 for one logged 10 entry, re-checked now: T1 PASS / NORMALIZED, the report and the uploaded file
    unchanged since prepare, the report's last commit the one recorded and by K6-10, K5-10, the report's own
    conditions, every bound input unchanged. -> (ok, reasons)."""
    t10, why = x.get("test10") or {}, []
    if x.get("t1_verdict") not in T1_UPLOAD_OK:
        why.append(f"T1 {x.get('t1_verdict')!r}")
    rel_ = t10.get("report") or ""
    rep = None
    rp = repo_path(rel_) if rel_ else ""
    if not rel_ or not os.path.isfile(rp) or sha256_file(rp) != t10.get("report_sha256"):
        why.append(f"{rel_ or 'the 10 submit report'} changed since prepare ({T10_REDO_HINT})")
    else:
        with open(refuse_forbidden(rp), encoding="utf-8") as f:
            rep = json.load(f)
    fp = repo_path(x.get("file") or "") if x.get("file") else ""
    if not fp or not os.path.isfile(fp) or sha256_file(fp) != x.get("sha256"):
        why.append(f"{x.get('file')} changed since prepare")
    k6 = t10.get("k6_10") or {}
    h, t = git_commit_time(rel_) if rel_ else (None, None)
    if not k6.get("ok") or h is None or h != k6.get("commit") or parse_time(t, "commit time") > K6_10:
        why.append(f"K6-10: the report's last commit is {h} at {t}; prepare recorded {k6.get('commit')} at "
                   f"{k6.get('committed_at')} (deadline {K6_10.isoformat()})")
    if not t10_k5_state()["ok"]:
        why.append("K5-10 does not hold")
    if rep is not None:
        why += t10_report_problems(rep, rel_, fp if fp and os.path.isfile(fp) else None, x.get("sha256"))
        why += t10_inputs_problems(rep.get("inputs_bound"), need_t4=any((rep.get("a_holds") or {}).values()))
    return not why, why


def t10_state(log, now):
    """PREREG_10 4.4 from the log: 06-ON when PREREG_06 designates it; else a 10 upload that holds; else PREREG_06 5.4
    ('off' / 'q04'); 'undecided' while 06-ON waits for its veto."""
    ents = t10_entries(log)
    elig = {x["id"]: t10_entry_eligible(x) for x in ents}
    ups10 = [x for x in uploads_of(log, "player") if x.get("test10") and elig[x["id"]][0]]
    if t06_entries(log):
        s6 = t06_state(log, now)
    else:
        s6 = {"designated": "q04", "designated_entries": [], "rule": "no 06 entry: Q04 (U5)"}
    st = {"entries": sorted(elig), "eligible": [i for i, (ok, _w) in elig.items() if ok],
          "not_eligible": {str(i): w for i, (ok, w) in elig.items() if not ok}, "prereg06_designated": s6["designated"],
          "k6_10_passed": now > K6_10}
    if s6["designated"] == "on":
        st.update(designated="on", designated_entries=s6["designated_entries"],
                  rule="06-ON is kept (PREREG_06 5.3 / 5.4): final 06-ON; 10 is not uploaded (PREREG_10 4.4)")
    elif s6["designated"] == "undecided":
        st.update(designated="undecided", designated_entries=[], rule=s6["rule"])
    elif ups10:
        st.update(designated="10", designated_entries=[x["id"] for x in ups10],
                  rule="a 10 upload meets PREREG_10 4.4: final 10")
    else:
        st.update(designated=s6["designated"], designated_entries=s6["designated_entries"],
                  rule="no 10 upload meets PREREG_10 4.4: " + str(s6["rule"]))
    return st


def prepare_rule_t10(log, comp, bound, not_10_reason):
    """PREREG_10 4.4 at prepare: a competition-1 file that is not 10 while a logged 10 upload is the designated final
    upload needs the revision check of --not-10-reason. None when the log has no 10 entry (exactly as before)."""
    if bound.get("test21"):              # PREREG_21 7: a 21 file is judged on its base in prepare_order_t21
        if (not_10_reason or "").strip():
            raise LogError("--not-10-reason does not apply to a 21 file")
        return None
    reason = (not_10_reason or "").strip()
    if comp != "player" or bound.get("test10"):
        if reason:
            raise LogError("--not-10-reason applies to a competition-1 file that is not 10 (Q04, 06-OFF, 06-ON)")
        return None
    if not t10_entries(log):
        return {"rule": "PREREG_10 4.4", "eligible_10_entries": [], "not_10_reason": reason} if reason else None
    st = t10_state(log, t06_now())
    rec = {"rule": "PREREG_10 4.4", "eligible_10_entries": st["eligible"], "designated": st["designated"],
           "not_eligible_10_entries": st["not_eligible"], "checked_at": now_iso(), "not_10_reason": reason or None}
    if st["designated"] == "10":
        when = [x["prepared_at"] for x in t10_entries(log) if x["id"] in st["eligible"]]
        rec["revision"] = p10_revision(reason, when, f"10 entries {st['designated_entries']} meet PREREG_10 4.4, so "
                                                     f"another competition-1 file is not the final choice")
    return rec


def final_rule_t10(log, e, not_10_reason):
    """PREREG_10 4.4 for competition 1. None when no 10 entry is logged (the rules before are unchanged then). The
    final choice must be the designated upload (t10_state); any other needs --not-10-reason citing a committed later
    revision (p10_revision), which is recorded."""
    reason = (not_10_reason or "").strip()
    if e["comp"] != "player":
        if reason:
            raise LogError("--not-10-reason applies to competition 1 (player) only")
        return None
    if not t10_entries(log):
        return {"rule": "PREREG_10 4.4", "eligible_10_entries": [], "not_10_reason": reason} if reason else None
    st = t10_state(log, t06_now())
    chosen = "10" if e.get("test10") else ((e.get("test06") or {}).get("variant") or "q04")
    rec = {"rule": "PREREG_10 4.4", "eligible_10_entries": st["eligible"], "not_eligible_10_entries": st["not_eligible"],
           "designated": st["designated"], "designated_rule": st["rule"], "designated_entries": st["designated_entries"],
           "chosen": chosen, "checked_at": now_iso(), "not_10_reason": reason or None}
    ok = chosen == st["designated"] and (chosen == "q04" or e["id"] in st["designated_entries"])
    if ok:
        return rec
    if st["designated"] == "undecided" and not reason:
        raise LogError(f"PREREG_10 4.4 / PREREG_06 5.3: 06-ON is uploaded and its veto is not decided ({st['rule']}); "
                       f"run PY tools/t06_player.py veto and commit veto.json first (docs/TESTDAY_06.md step 7)")
    if not reason:
        names = {"q04": "Q04 (U5)", "off": "06-OFF", "on": "06-ON", "10": "10", "undecided": "undecided"}
        raise LogError(f"PREREG_10 4.4: the final choice must be {names[st['designated']]} "
                       f"{st['designated_entries'] or ''} ({st['rule']}); entry #{e['id']} is "
                       f"{names.get(chosen, chosen)}. Another choice needs --not-10-reason \"PREREG_10 rev N: ...\"")
    when = [x["prepared_at"] for x in t10_entries(log)]
    rec["revision"] = p10_revision(reason, when, f"entry #{e['id']} ({chosen}) is not the upload PREREG_10 4.4 "
                                                 f"designates ({st['designated']})")
    return rec


def t10_status(log, fin, last, now):
    st = t10_state(log, now)
    by_id = {x["id"]: x for x in log["uploads"]}
    fin_e = by_id.get((fin or {}).get("id"))
    chosen = None
    if fin_e:
        chosen = "10" if fin_e.get("test10") else ((fin_e.get("test06") or {}).get("variant") or "q04")
    rule = (fin or {}).get("prereg10_4_4") or {}
    final_ok = bool(fin_e) and ((chosen == st["designated"] and (chosen == "q04" or fin_e["id"] in
                                                                 st["designated_entries"]))
                                or bool(rule.get("revision")))
    st.update(final_set=bool(fin), final_choice=chosen, violation=bool(fin) and not final_ok,
              last_upload=(last or {}).get("id"),
              last_is_designated=bool(last) and (last["id"] in st["designated_entries"] or
                                                 (st["designated"] == "q04" and not last.get("test06")
                                                  and not last.get("test10"))))
    return st


def t10_status_lines(st, now):
    names = {"q04": "Q04（U5）", "off": "06-OFF", "on": "06-ON", "10": "10", "undecided": "還沒定（06-ON 等否決）"}
    lines = [f"  10（PREREG_10 4.4）：紀錄 {st['entries']}；現在符合條件的 {st['eligible'] or '沒有'}；06 這條線指定 "
             f"{names.get(st['prereg06_designated'], st['prereg06_designated'])}；最終應該是 {names[st['designated']]} "
             f"{st['designated_entries'] or ''}"]
    for i, why in st["not_eligible"].items():
        text = "；".join(why)
        lines.append(f"  #{i} 不符合：{text[:600]}{'…' if len(text) > 600 else ''}")
    if st["violation"]:
        lines.append("  違反 PREREG_10 4.4：最終選擇不是規則指定的那一份（也沒有引用修訂）；照 docs/TESTDAY_10.md 第 6 步重跑 final")
    elif not st["final_set"] and st["designated"] != "undecided":
        lines.append(f"  最終選擇要設成 {names[st['designated']]}（docs/TESTDAY_10.md 第 6 步）")
    if st["designated"] in ("10", "off", "on") and st["designated_entries"] and not st["last_is_designated"]:
        lines.append(f"  最後一次上傳不是 {names[st['designated']]}：再上傳一次，讓它成為最後一次上傳（docs/TESTDAY_10.md 第 6 步）")
    lines.append(f"  距 K6-10 11/21 23:59（10 的正式執行要在這之前 commit）：{_left(now, K6_10)}")
    return lines


def t10_public_lines(log, e):
    """PREREG_10 4.4: 10, 06-OFF and Q04 share submission_player.csv. Two equal public scores while the 10 report counts
    public top-1 differences > 0 between them: possibly the wrong file. A 10 score 0.05 or more below the previous
    upload: look for bugs only (never change the method because of the score). Empty when the log has no 10 entry."""
    if e["comp"] != "player" or e.get("public_score") is None or not t10_entries(log):
        return []
    lines = []
    others = [x for x in uploads_of(log, "player") if x["id"] != e["id"] and x.get("public_score") is not None
              and abs(x["public_score"] - e["public_score"]) < T06_SAME_SCORE]
    for x in others:
        a, b = (e, x) if e.get("test10") else (x, e)
        if not a.get("test10") or b.get("test10"):
            continue
        kind = "06-OFF" if (b.get("test06") or {}).get("variant") == "off" else \
            "06-ON" if (b.get("test06") or {}).get("variant") == "on" else "Q04"
        src = (a["test10"].get("top1_differences_vs_06_off") if kind == "06-OFF" else
               a["test10"].get("top1_differences_vs_q04") if kind == "Q04" else None) or {}
        n = src.get("public")
        if _is_count(n) and n > 0:
            lines.append(f"WARNING (PREREG_10 4.4): #{a['id']}（10）和 #{b['id']}（{kind}）的 public 分數完全相同"
                         f"（{a['public_score']}），但 public 池第 1 名不同 {n} 題：可能傳錯檔（同名）：核對 AIdea 的上傳紀錄")
    if e.get("test10") and e.get("public_delta_vs_prev") is not None and e["public_delta_vs_prev"] <= -DROP_ALERT + 1e-12:
        lines.append("PREREG_10 4.4：10 比上一份低 0.05 以上：只查錯（id 對應、列順序、池配對、seam）；查不到就維持 4.4 的結果，寫進報告")
    return lines


# ---------------------------------------------------------------- PREREG_14: test14 (competition 2, exact-copy layer)
def t14_dir():
    return os.path.join(REPO, *T14_DIR)


def is_t14_file(comp, path):
    """A competition-2 file in results/test14/rank/ is a 14 file (only tools/t14_copy.py apply writes there)."""
    if comp != "rank":
        return False
    here = os.path.realpath(os.path.dirname(os.path.abspath(path)))
    return os.path.normcase(here) == os.path.normcase(os.path.realpath(t14_dir()))


def t14_entries(log):
    return [x for x in log["uploads"] if x.get("comp") == "rank" and x.get("test14")]


def t14_rank_rows(path):
    """({question_id: pred_rank}, [question_id in file order]) of a rank file."""
    csv.field_size_limit(10**8)
    with open(refuse_forbidden(path), newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f)][1:]
    return {r[0]: r[1] for r in rows if len(r) == 2}, [r[0] for r in rows if r]


def t14_report_problems(rep, rel, path, sha):
    """PREREG_14 5 / 7, re-read from the committed 14 report: real mode, verdict used, check_submission official
    without errors, this file's SHA-256 (sha None: not compared), the index pins equal on part 'all', T1 PASS /
    NORMALIZED, a base in one of the competition-2 result folders whose bytes are the recorded ones, at least one
    changed row, and the file differs from the base exactly in the rows the report lists as changed."""
    probs = []
    if rep.get("format") != T14_FORMAT:
        probs.append(f"{rel}: format {rep.get('format')!r}")
    if rep.get("mode") != "real":
        probs.append(f"{rel}: mode {rep.get('mode')!r} (uploads come from real-mode runs only)")
    if rep.get("verdict") != "used":
        probs.append(f"{rel}: verdict {rep.get('verdict')!r} (not 'used')")
    cs = rep.get("check_submission") or {}
    if cs.get("mode") != "official" or cs.get("errors"):
        probs.append(f"{rel}: check_submission {cs.get('mode')!r} with errors {cs.get('errors')}")
    want = (rep.get("outputs_sha256") or {}).get(FILE_NAMES["rank"])
    if sha is not None and want != sha:
        probs.append(f"{rel}: records another {FILE_NAMES['rank']} than this file ({sha[:12]})"
                     + (lf_hint(path, want, repo_rel(path)) if path else ""))
    pins = rep.get("index_pins_check") or {}
    if pins.get("equal") is not True or pins.get("part") != "all":
        probs.append(f"{rel}: the index pins were not checked equal on part 'all' (PREREG_14 3)")
    pp = repo_path(T14_PINS)
    if os.path.isfile(pp):
        with open(refuse_forbidden(pp), encoding="utf-8") as f:
            want_pins = ((json.load(f).get("parts") or {}).get("all")) or {}
        got_pins = pins.get("pins") or {}
        off = [k for k in T14_PIN_FIELDS if got_pins.get(k) != want_pins.get(k)]
        if off:
            probs.append(f"{rel}: index pins {off} differ from the committed {T14_PINS}")
    for name in QUESTION_FILES["rank"]:
        tp = os.path.join(REPO, "data", "test", name)
        want_t = (((rep.get("files") or {}).get(name)) or {}).get("sha256")
        if not os.path.isfile(tp) or sha256_file(tp) != want_t:
            probs.append(f"{rel}: data/test/{name} is not the file the 14 report read ({str(want_t)[:12]})")
    t1v = (((rep.get("preflight") or {}).get("t1")) or {}).get("verdict")
    if t1v not in T1_UPLOAD_OK:
        probs.append(f"{rel}: T1 {t1v!r} at apply time")
    base = rep.get("base") or {}
    bf = base.get("file") or ""
    changed = {o.get("question_id"): o.get("new_level") for o in rep.get("overrides") or [] if o.get("changed")}
    if not changed or rep.get("rows_changed") != len(changed):
        probs.append(f"{rel}: rows_changed {rep.get('rows_changed')!r} and {len(changed)} changed overrides "
                     f"(a 14 file changes at least one row)")
    if os.path.dirname(bf) not in T14_BASE_DIRS:
        probs.append(f"{rel}: base file {bf!r} is not in {T14_BASE_DIRS}")
        return probs
    bp = repo_path(bf)
    if not os.path.isfile(bp) or sha256_file(bp) != base.get("sha256"):
        probs.append(f"{bf} is missing or no longer the base bytes the 14 report records "
                     f"({str(base.get('sha256'))[:12]})")
    elif path is not None and os.path.isfile(path):
        b, border = t14_rank_rows(bp)
        n, norder = t14_rank_rows(path)
        if border != norder:
            probs.append(f"{repo_rel(path)}: the question order differs from its base {bf}")
        diff = {q: n.get(q) for q in border if n.get(q) != b.get(q)}
        bad_base = [o.get("question_id") for o in rep.get("overrides") or []
                    if b.get(o.get("question_id")) != o.get("base_level")]
        if bad_base:
            probs.append(f"{rel}: base_level of {bad_base[:5]} is not the base file's level")
        if diff != changed:
            extra = sorted(set(diff) ^ set(changed))[:5]
            probs.append(f"{repo_rel(path)}: differs from its base in other rows than the report lists "
                         f"({len(diff)} vs {len(changed)}; e.g. {extra})")
    return probs


def check_reports_t14(comp, path, sha):
    """PREREG_14 7: bind a 14 file to its committed report, the raw T1, the index pins, the registration and K6-14."""
    d = os.path.dirname(os.path.abspath(path))
    rpath = os.path.join(d, T14_REPORT)
    rel_, rep = _committed_json(rpath, "the 14 report (tools/t14_copy.py apply)")
    out = {rel_: sha256_file(rpath)}
    probs = t14_report_problems(rep, rel_, path, sha)
    t1_path = os.path.join(REPO, "results", "test02", "checks", comp, "t1.json")
    trel, t1 = _committed_json(t1_path, "the T1 result")
    out[trel] = sha256_file(t1_path)
    probs += t1_problems(t1, trel)
    v = t1.get("verdict")
    if v not in T1_UPLOAD_OK:
        probs.append(f"{trel}: verdict {v!r} (PREREG_14 9: the layer is not used)")
    prel, _pins = _committed_json(repo_path(T14_PINS), "the index pins")
    out[prel] = sha256_file(repo_path(T14_PINS))
    srel, sim = _committed_json(repo_path(T14_SIM), "the simulation record")
    out[srel] = sha256_file(repo_path(T14_SIM))
    if not (sim.get("evaluation") or {}).get("passed"):
        probs.append(f"{srel}: the simulation did not pass (PREREG_14 6)")
    k5 = (((rep.get("preflight") or {}).get("k5_and_sim")) or {})
    if k5.get("k5_ok") is not True:
        probs.append(f"{rel_}: K5-14 was not confirmed at apply time (PREREG_14 8)")
    h2, t2 = git_row_added(P14_DOC, "| rev 2 |")
    if h2 is None or parse_time(t2, "commit time") > K5_14:
        probs.append(f"K5-14: the '| rev 2 |' row of {P14_DOC} was added at {t2} ({h2}); the deadline is "
                     f"{K5_14.isoformat()} (PREREG_14 8)")
    st = git_file_state(P14_DOC)
    if not st["tracked"] or not st["clean"]:
        probs.append(f"{P14_DOC} is not committed and clean")
    h, t = git_commit_time(rel_)
    k6_ok = h is not None and parse_time(t, "commit time") <= K6_14
    if not k6_ok:
        probs.append(f"K6-14: {rel_} last committed {t} ({h}); the deadline is {K6_14.isoformat()} (PREREG_14 8)")
    if probs:
        raise LogError("the 14 file is not backed by its report (PREREG_14 7):\n - " + "\n - ".join(probs))
    base = rep["base"]
    t14 = {"report": rel_, "report_sha256": out[rel_], "base_entry_id": base.get("entry_id"),
           "base_file": base.get("file"), "base_sha256": base.get("sha256"),
           "questions_overridden": rep.get("questions_overridden"), "rows_changed": rep.get("rows_changed"),
           "changed_by_pool": rep.get("changed_by_pool") or {},
           "k6_14": {"commit": h, "committed_at": t, "deadline": K6_14.isoformat(), "ok": True}}
    return {"reports": out, "t1_verdict": v, "fallback_used": False, "test14": t14}


def method_for_t14(given):
    """A 14 file is recorded as the PREREG_14 layer, never with its base's label."""
    if given and "prereg14" not in given:
        raise LogError(f"--method {given!r} does not name prereg14; a results/test14/rank file is the exact-copy layer "
                       f"(default {T14_METHOD!r})")
    return given or T14_METHOD


def t14_base_reasons(fin):
    """The --not-0x-reason values recorded with a competition-2 final choice (None where none was given)."""
    fin = fin or {}
    return {"not_04_reason": (fin.get("prereg04_5_3") or {}).get("not_04_reason"),
            "not_07_reason": (fin.get("prereg07_5_4") or {}).get("not_07_reason"),
            "not_08_reason": (fin.get("prereg08_5_4") or {}).get("not_08_reason"),
            "not_09_reason": (fin.get("prereg09_5_3") or {}).get("not_09_reason")}


def t14_base_rules(log, base, reasons):
    """The competition-2 final rules (PREREG_04 5.3, 07 5.4, 08 5.4, 09 5.3) applied to the base entry as final_rule
    applies them; raises LogError when they no longer designate it. -> {record key: chosen}."""
    out = {}
    for key, fn, rk in (("prereg04_5_3", final_rule_t04, "not_04_reason"),
                        ("prereg07_5_4", final_rule_t07, "not_07_reason"),
                        ("prereg08_5_4", final_rule_t08, "not_08_reason"),
                        ("prereg09_5_3", final_rule_t09, "not_09_reason")):
        r = fn(log, base, (reasons or {}).get(rk))
        if r:
            out[key] = r.get("chosen")
    r19 = final_rule_t19(log, base, (reasons or {}).get("not_19_reason"))   # PREREG_19 4.3 (None: no 19 entry)
    if r19:
        out["prereg19_4_3"] = r19.get("chosen")
    return out


def t14_base_entry(log, e14):
    """The uploaded, non-14 base entry a 14 entry was built on (same file and bytes)."""
    t14 = e14.get("test14") or {}
    base = next((x for x in log["uploads"] if x["id"] == t14.get("base_entry_id")), None)
    if base is None or base.get("comp") != "rank" or base.get("status") != "uploaded" or base.get("test14"):
        raise LogError(f"PREREG_14 7: the base #{t14.get('base_entry_id')} of 14 entry #{e14['id']} is not an uploaded "
                       f"competition-2 entry")
    if base.get("sha256") != t14.get("base_sha256") or base.get("file") != t14.get("base_file"):
        raise LogError(f"PREREG_14 7: entry #{base['id']} is not the base the 14 report records "
                       f"({t14.get('base_file')}, {str(t14.get('base_sha256'))[:12]})")
    return base


def prepare_order_t14(log, t14):
    """PREREG_14 7: the report's base is final_choice.rank, an uploaded non-14 entry with the same bytes, and the
    competition-2 final rules (with the reasons recorded in that final choice) still designate it."""
    fin = log["final_choice"].get("rank")
    if not fin or fin.get("id") != t14.get("base_entry_id") or fin.get("sha256") != t14.get("base_sha256"):
        raise LogError(f"PREREG_14 7: the 14 report's base #{t14.get('base_entry_id')} is not final_choice.rank "
                       f"(#{(fin or {}).get('id')}): set the final choice by the current rules first "
                       f"(docs/TESTDAY_09.md step 5), then rerun tools/t14_copy.py apply (docs/TESTDAY_14.md)")
    base = t14_base_entry(log, {"id": "(new)", "test14": t14})
    reasons = t14_base_reasons(fin)
    if (fin.get("prereg19_4_3") or {}).get("not_19_reason"):   # PREREG_19 4.3: only when one was given
        reasons["not_19_reason"] = fin["prereg19_4_3"]["not_19_reason"]
    try:
        rules = t14_base_rules(log, base, reasons)
    except LogError as e:
        raise LogError(f"PREREG_14 7: the base #{base['id']} is no longer the file the competition-2 rules designate: "
                       f"{e}")
    t14["base_rule_reasons"] = reasons
    t14["base_rules"] = rules
    t14["base_method"] = base.get("method")


def print_t14_ok_lines(t14):
    ch = t14.get("changed_by_pool") or {}
    print(f"- 這一份是 14（PREREG_14 完全複製層），底檔是 #{t14.get('base_entry_id')}（{t14.get('base_file')}，"
          f"{t14.get('base_method')}）；覆寫 {t14.get('questions_overridden')} 題，其中 {t14.get('rows_changed')} 題的等級"
          f"和底檔不同（public {ch.get('public', 0)}、private {ch.get('private', 0)}）")
    print("- 14 之外的列和底檔逐位元組相同；上傳後要 final --comp rank --id <這一筆>（docs/TESTDAY_14.md）")
    print(f"- K6-14：14 的報告在 {t14['k6_14']['committed_at']} commit（期限 11/23 18:00）")


def t14_entry_eligible(x, log, require_uploaded=True, with_base_rules=True):
    """PREREG_14 7 for one logged 14 entry, re-checked now: (uploaded,) the report and the file unchanged since prepare
    and still meeting t14_report_problems, the report's last commit the recorded one and by K6-14, the base entry
    uploaded with the recorded bytes, and (with_base_rules) the final rules still designating the base. -> (ok, why)."""
    t14, why = x.get("test14") or {}, []
    if require_uploaded and x.get("status") != "uploaded":
        why.append(f"#{x['id']} is {x.get('status')!r}, not uploaded")
    rel_ = t14.get("report") or ""
    rp = repo_path(rel_) if rel_ else ""
    rep = None
    if not rel_ or not os.path.isfile(rp) or sha256_file(rp) != t14.get("report_sha256"):
        why.append(f"{rel_ or 'the 14 report'} changed since prepare (rerun tools/t14_copy.py apply, docs/TESTDAY_14.md)")
    else:
        with open(refuse_forbidden(rp), encoding="utf-8") as f:
            rep = json.load(f)
    fp = repo_path(x.get("file") or "") if x.get("file") else ""
    if not fp or not os.path.isfile(fp) or sha256_file(fp) != x.get("sha256"):
        why.append(f"{x.get('file')} changed since prepare")
    k6 = t14.get("k6_14") or {}
    h, t = git_commit_time(rel_) if rel_ else (None, None)
    if not k6.get("ok") or h is None or h != k6.get("commit") or parse_time(t, "commit time") > K6_14:
        why.append(f"K6-14: the report's last commit is {h} at {t}; prepare recorded {k6.get('commit')} at "
                   f"{k6.get('committed_at')} (deadline {K6_14.isoformat()})")
    if rep is not None:
        why += t14_report_problems(rep, rel_, fp if fp and os.path.isfile(fp) else None, x.get("sha256"))
    try:
        base = t14_base_entry(log, x)
    except LogError as e:
        why.append(str(e))
        base = None
    if base is not None and with_base_rules:
        try:
            t14_base_rules(log, base, t14.get("base_rule_reasons") or {})
        except LogError as e:
            why.append(f"the base #{base['id']} is no longer the file the competition-2 rules designate: {e}")
        fin = log["final_choice"].get("rank")
        fin_e = next((y for y in log["uploads"] if y["id"] == (fin or {}).get("id")), None)
        on_base = fin_e is not None and (fin_e["id"] == base["id"] or (fin_e.get("test14") or {}).get(
            "base_entry_id") == base["id"])
        if fin and not on_base:
            why.append(f"the final choice moved to #{fin.get('id')} (not the base #{base['id']} or a 14 on it): the "
                       f"competition-2 rules designate another file now (PREREG_14 7)")
    return not why, why


def t14_eligibility(log):
    return {x["id"]: t14_entry_eligible(x, log) for x in t14_entries(log)}


def head_has_p14_revision_row(n, head=None):
    head = git_head_text(P14_DOC) if head is None else head
    return bool(head) and any(line.startswith(f"| rev {n} |") for line in head.splitlines())


def p14_revision(reason, overridden, what):
    """Setting aside a 14 upload that meets PREREG_14 7: a reason naming 'PREREG_14 rev N' (N >= 3) whose row is at
    HEAD of the committed, clean docs/PREREG_14_RANK_EXACT_COPY.md and was added after every overridden record."""
    text = (reason or "").strip()
    m = re.search(r"PREREG_14\s+rev\s+(\d+)", text)
    if not m:
        raise LogError(f"{what}: needs --not-14-reason \"PREREG_14 rev N: <the program error>\" citing a committed "
                       f"revision (N >= {P14_MIN_REV}); a reason without a revision is refused")
    n = int(m.group(1))
    if n < P14_MIN_REV:
        raise LogError(f"{what}: --not-14-reason cites PREREG_14 rev {n}; an override needs rev {P14_MIN_REV} or later")
    st = git_file_state(P14_DOC)
    if not st["tracked"] or not st["clean"]:
        raise LogError(f"{what}: {P14_DOC} must be committed and unchanged (tracked={st['tracked']}, "
                       f"clean={st['clean']})")
    row = f"| rev {n} |"
    if not head_has_p14_revision_row(n):
        raise LogError(f"{what}: {P14_DOC} at HEAD has no row starting {row!r}; commit the revision first")
    h, t = git_row_added(P14_DOC, row)
    if h is None:
        raise LogError(f"{what}: no commit found that added {row!r} to {P14_DOC}")
    at = parse_time(t, "commit time")
    early = [w for w in overridden if at <= parse_time(w, "time")]
    if early:
        raise LogError(f"{what}: PREREG_14 rev {n} was committed at {t} ({h[:12]}), not after the records it "
                       f"overrides ({', '.join(early)}); a revision must come after the decision it overturns")
    return {"rev": n, "row": row, "doc": P14_DOC, "commit": h, "committed_at": t, "reason": text,
            "overridden_at": list(overridden)}


def _t14_other_rule(log, reason, what, sha=None):
    """sha: the non-14 file's SHA-256; only 14 entries built on that very file need --not-14-reason (a file the
    04-09 rules designate instead of the base makes the 14 on the old base moot, PREREG_14 7)."""
    elig = {i: v for i, v in t14_eligibility(log).items()
            if sha is None or (entry(log, i).get("test14") or {}).get("base_sha256") == sha}
    if not elig:
        if reason:
            raise LogError("--not-14-reason: the log has no 14 entry to set aside")
        return None
    ok_ids = [i for i, (ok, _w) in elig.items() if ok]
    rec = {"rule": "PREREG_14 7", "eligible_14_entries": ok_ids,
           "not_eligible_14_entries": {str(i): w for i, (ok, w) in elig.items() if not ok}, "checked_at": now_iso(),
           "not_14_reason": reason or None}
    if ok_ids:
        when = [x["prepared_at"] for x in t14_entries(log) if x["id"] in ok_ids]
        rec["revision"] = p14_revision(reason, when, f"14 uploads {ok_ids} meet PREREG_14 7, so {what}")
    return rec


def prepare_rule_t14(log, comp, bound, not_14_reason):
    """PREREG_14 7 at prepare: a competition-2 file that is not 14 while a logged 14 upload meets PREREG_14 7 needs the
    revision check of --not-14-reason. None when the log has no 14 entry (exactly as before)."""
    reason = (not_14_reason or "").strip()
    if comp != "rank" or bound.get("test14"):
        if reason:
            raise LogError("--not-14-reason applies to a competition-2 file that is not 14")
        return None
    return _t14_other_rule(log, reason, "their base is not uploaded again as the final file",
                           sha=bound.get("_sha"))


def final_rule_t14_self(log, e14, not_14_reason):
    """PREREG_14 7 for a 14 final (the 04-09 rules were applied to its base by cmd_final): the 14 entry itself."""
    if (not_14_reason or "").strip():
        raise LogError("--not-14-reason does not apply when the 14 entry is chosen")
    ok, why = t14_entry_eligible(e14, log, require_uploaded=False, with_base_rules=False)
    if not ok:
        raise LogError(f"entry #{e14['id']} (14) does not meet PREREG_14 7 now ({'; '.join(why)[:900]}): the final "
                       f"choice is the file the competition-2 rules designate (docs/TESTDAY_14.md)")
    t14 = e14["test14"]
    return {"rule": "PREREG_14 7", "chosen": "14", "base_entry_id": t14.get("base_entry_id"),
            "eligible_14_entries": [i for i, (k, _w) in t14_eligibility(log).items() if k], "checked_at": now_iso(),
            "not_14_reason": None}


def final_rule_t14_other(log, e, not_14_reason):
    """PREREG_14 7 for a competition-2 final that is not 14: None without 14 entries; while a 14 upload meets the
    conditions, only with --not-14-reason (p14_revision)."""
    reason = (not_14_reason or "").strip()
    if e["comp"] != "rank":
        if reason:
            raise LogError("--not-14-reason applies to competition 2 (rank) only")
        return None
    rec = _t14_other_rule(log, reason, "the final choice is 14", sha=e.get("sha256"))
    if rec is not None:
        rec["chosen"] = "not 14"
    return rec


def t14_status_view(log, comp, fin, last):
    """(fin, last) as the 04-09 status blocks should see them when the final choice is a 14 entry: its base. None
    otherwise (the blocks see the real values, exactly as before)."""
    if comp != "rank" or not fin:
        return None
    e = next((x for x in log["uploads"] if x["id"] == fin.get("id")), None)
    if not e or not e.get("test14"):
        return None
    base = next((x for x in log["uploads"] if x["id"] == (e["test14"] or {}).get("base_entry_id")), None)
    if base is None:
        return None
    fin_v = dict(fin, id=base["id"], sha256=base["sha256"], file=base["file"], method=base.get("method"))
    return fin_v, (base if last and last["id"] == e["id"] else last)


def t14_status(log, fin, last):
    elig = t14_eligibility(log)
    ok_ids = [i for i, (ok, _w) in elig.items() if ok]
    by_id = {x["id"]: x for x in log["uploads"]}
    fin_e = by_id.get((fin or {}).get("id"))
    final_is_14 = bool(fin_e and fin_e.get("test14"))
    rule = (fin or {}).get("prereg14_7") or {}
    rev = rule.get("revision") or {}

    def overridden_by_rev(i):
        prep = (by_id.get(i) or {}).get("prepared_at")
        if i not in (rule.get("eligible_14_entries") or []) or not rev.get("committed_at") or not prep:
            return False
        try:
            return parse_time(rev["committed_at"], "revision commit time") > parse_time(prep, "prepared_at")
        except LogError:
            return False
    final_ok = (final_is_14 and fin_e["id"] in ok_ids and rule.get("chosen") == "14") or \
        (bool(fin) and not final_is_14 and bool(rev) and all(overridden_by_rev(i) for i in ok_ids))
    final_bad = bool(final_is_14 and not elig.get(fin_e["id"], (False, []))[0])
    return {"entries": sorted(elig), "eligible": ok_ids,
            "not_eligible": {str(i): w for i, (ok, w) in elig.items() if not ok},
            "final_set": bool(fin), "final_is_14": final_is_14,
            "last_is_14": bool(last and last.get("test14")),
            "violation": bool(ok_ids) and bool(fin) and not final_ok, "redo_base": final_bad}


def t14_status_lines(st):
    lines = [f"  14（PREREG_14 7）：紀錄 {st['entries']}；現在符合條件的 {st['eligible'] or '沒有'}；最終選擇"
             f"{'是 14' if st['final_is_14'] else '不是 14'}"]
    for i, why in st["not_eligible"].items():
        text = "；".join(why)
        lines.append(f"  #{i} 不符合：{text[:600]}{'…' if len(text) > 600 else ''}")
    if st["eligible"] and not st["final_is_14"]:
        if st["violation"]:
            lines.append("  違反 PREREG_14 7：有符合條件的 14，最終選擇卻不是它（沒有引用修訂）；照 docs/TESTDAY_14.md 重跑 final")
        elif not st["final_set"]:
            lines.append("  有符合 PREREG_14 7 的 14：最終選擇要設成 14（docs/TESTDAY_14.md）")
    if st["redo_base"]:
        lines.append("  照 PREREG_14 7：14 不再符合；final 改回現行規則指定的那一份，再 prepare 並重傳它（docs/TESTDAY_14.md）")
    return lines


def t14_public_lines(log, e):
    """PREREG_14: 14 and its base share submission_rank.csv. Equal public scores while the 14 report changed public
    rows, or different public scores while it changed none: a hint that the wrong file was uploaded. Empty when the
    log has no 14 entry."""
    if e["comp"] != "rank" or e.get("public_score") is None or not t14_entries(log):
        return []
    lines = []
    for x in uploads_of(log, "rank"):
        if x["id"] == e["id"] or x.get("public_score") is None:
            continue
        a, b = (e, x) if e.get("test14") else (x, e)
        if not a.get("test14") or b.get("test14") or (a["test14"] or {}).get("base_entry_id") != b["id"]:
            continue
        n_pub = ((a["test14"] or {}).get("changed_by_pool") or {}).get("public", 0)
        same = abs(a["public_score"] - b["public_score"]) < 1e-9
        if same and _is_count(n_pub) and n_pub > 0:
            lines.append(f"WARNING (PREREG_14): #{a['id']}（14）和底檔 #{b['id']} 的 public 分數完全相同，但 14 改了 public "
                         f"{n_pub} 題：可能傳錯檔（同名）：核對 AIdea 的上傳紀錄")
        if not same and n_pub == 0:
            lines.append(f"WARNING (PREREG_14): #{a['id']}（14）沒有改任何 public 題，public 分數卻和底檔 #{b['id']} 不同："
                         f"可能傳錯檔：核對 AIdea 的上傳紀錄")
    return lines


# ---------------------------------------------------------------- PREREG_19: test19 (competition 2, Q19 = Q09 + 4 x SET)
def t19_dir():
    return os.path.join(REPO, *T19_DIR)


def is_t19_file(comp, path):
    """A competition-2 file in results/test19/rank/ is a 19 file (only tools/t19_rank.py apply writes there)."""
    if comp != "rank":
        return False
    here = os.path.realpath(os.path.dirname(os.path.abspath(path)))
    return os.path.normcase(here) == os.path.normcase(os.path.realpath(t19_dir()))


def t19_adoption_problems():
    """PREREG_19 2.2 / 2.3, re-read from the committed validation apply report of prereg19: its LF SHA-256 is the
    pinned one, Q19, usable true after the audit (finalized)."""
    p = repo_path(T19_ADOPTION)
    if not os.path.isfile(p):
        return [f"{T19_ADOPTION} is missing: 19 was not adopted (PREREG_19 2.2)"]
    st = git_file_state(T19_ADOPTION)
    if not st["tracked"] or not st["clean"]:
        return [f"{T19_ADOPTION} is not committed and clean"]
    if sha256_bound(p) != T19_ADOPTION_SHA:
        return [f"{T19_ADOPTION} is not the committed adoption record ({sha256_bound(p)[:12]} vs "
                f"{str(T19_ADOPTION_SHA)[:12]})"]
    with open(refuse_forbidden(p), encoding="utf-8") as f:
        s = json.load(f)
    if s.get("combo") != T19_COMBO_ID or s.get("usable") is not True or not s.get("finalized"):
        return [f"{T19_ADOPTION}: combo {s.get('combo')!r}, usable {s.get('usable')!r}, finalized "
                f"{s.get('finalized')!r} (PREREG_19 2.2: 19 is not adopted)"]
    return []


def t19_inputs_problems(bound):
    """Every input the 19 apply report lists under inputs_bound is committed, clean and still has its SHA-256."""
    if not isinstance(bound, dict) or not bound:
        return [f"the 19 apply report lists no inputs_bound ({T19_REDO_HINT})"]
    probs = []
    missing = [r for r in T19_INPUTS_REQUIRED if r not in bound]
    if missing:
        probs.append(f"inputs_bound lacks {missing}")
    for rel_, want in sorted(bound.items()):
        s = str(rel_)
        if os.path.isabs(s) or "\\" in s or ".." in s.split("/") or s.startswith("/"):
            probs.append(f"inputs_bound key {s!r} is not a repo-relative path")
            continue
        p = repo_path(s)
        if not os.path.isfile(p):
            probs.append(f"{s} is missing ({T19_REDO_HINT})")
            continue
        st = git_file_state(s)
        if not st["tracked"] or not st["clean"]:
            probs.append(f"{s} is not committed and clean (先 commit；內容若已改變：{T19_REDO_HINT})")
            continue
        got = sha256_bound(p)
        if got != want:
            probs.append(f"{s} is {got[:12]}, the 19 apply report bound {str(want)[:12]} ({T19_REDO_HINT})")
    return probs


def t19_conditions(rep):
    """PREREG_19 4.2 recomputed from the numbers of the 19 apply report (never from its own verdicts).
    -> (record, problems)."""
    probs = []
    q = rep.get("q09_vs_09") or {}
    bound = rep.get("inputs_bound") or {}
    if q.get("equal") is not True or q.get("t09_sha256") != bound.get(T19_09_PROBS) or \
            q.get("p1_ok_equal_t09") is not True or q.get("usable_equal_t09") is not True or \
            not _is_num(q.get("max_abs_diff")) or q["max_abs_diff"] > T19_TOL_Q09 or \
            q.get("rule_a_differences") != 0 or q.get("rule_b_differences") != 0 or \
            not _is_num(q.get("lp_p1e_max_abs_diff_vs_t09")) or q["lp_p1e_max_abs_diff_vs_t09"] > T19_TOL_Q09:
        probs.append(f"Q09 recomputed is not 09's rank_test_probs.csv ({str(q.get('t09_sha256'))[:12]} vs "
                     f"{str(bound.get(T19_09_PROBS))[:12]}; max |dP| {q.get('max_abs_diff')!r}, lp_P1TO_E "
                     f"{q.get('lp_p1e_max_abs_diff_vs_t09')!r}, usable equal {q.get('p1_ok_equal_t09')!r} / "
                     f"{q.get('usable_equal_t09')!r})")
    sl = rep.get("seam_local") or {}
    if sl.get("required_byte_identical") is True:
        seam_ok = sl.get("byte_identical_probs") is True and sl.get("byte_identical_pred_B") is True
    else:
        seam_ok = (_is_num(sl.get("max_abs_diff")) and _is_num(sl.get("tol")) and sl["max_abs_diff"] <= sl["tol"]
                   and _is_count(sl.get("rule_b_differences")) and sl["rule_b_differences"] <= T19_SEAM_B_MAX)
    if not seam_ok or sl.get("usable_equal_to_prereg19") is not True:
        probs.append(f"local Q19 seam does not hold: {sl}")
    thr = max(0.5 * T19_RHO_MONO, T19_RHO_FLOOR)
    rho = {s: ((rep.get("set_direction") or {}).get(s) or {}).get("rho") for s in T19_SETS}
    for s, v in rho.items():
        if not _is_num(v) or v < thr:
            probs.append(f"SET direction of {s}: rho {v!r} < {thr} (no 19)")
    return {"rho": rho, "rho_threshold": thr, "seam_local_max_abs_diff": sl.get("max_abs_diff"),
            "q09_max_abs_diff": q.get("max_abs_diff"),
            "set_seam_record_only": (rep.get("set_seam_record_only") or {}).get("max_abs_lp_set")}, probs


def t19_report_problems(rep, rel, path, sha):
    """The 19 apply report's own conditions (format, mode, exit, combo, P1 sets, check_submission, this file) and
    PREREG_19 4.2 recomputed. -> (conditions, problems)."""
    probs = []
    if rep.get("format") != T19_FORMAT or rep.get("mode") != "real":
        probs.append(f"{rel}: format {rep.get('format')!r}, mode {rep.get('mode')!r} (a real-mode {T19_FORMAT})")
    if rep.get("exit_code") != 0 or rep.get("usable") is not True:
        probs.append(f"{rel}: exit_code {rep.get('exit_code')!r}, usable {rep.get('usable')!r}")
    combo = rep.get("combo") or {}
    if combo.get("id") != T19_COMBO_ID or tuple(combo.get("sources") or ()) != T19_COMBO_SOURCES:
        probs.append(f"{rel}: combination {combo.get('id')!r} {combo.get('sources')!r} is not Q19")
    fake = [s for s, r in (rep.get("p1_sets") or {}).items() if (r or {}).get("fake_from_val")]
    if fake or set(rep.get("p1_sets") or {}) != set(T19_P1_SETS):
        probs.append(f"{rel}: P1 sets {sorted(rep.get('p1_sets') or {})}, drill records {fake}")
    cs = rep.get("check_submission") or {}
    if cs.get("mode") != "official" or cs.get("errors"):
        probs.append(f"{rel}: check_submission {cs.get('mode')!r} with errors {cs.get('errors')}")
    want = (rep.get("outputs_sha256") or {}).get(FILE_NAMES["rank"])
    if want != sha:
        try:
            hint = lf_hint(path, want, repo_rel(path)) if path and os.path.exists(path) else ""
        except LogError:
            hint = ""
        probs.append(f"{rel}: records another {FILE_NAMES['rank']} than this file ({str(sha)[:12]})" + hint)
    cond, cprobs = t19_conditions(rep)
    return cond, probs + cprobs


def check_reports_t19(comp, path, sha):
    """PREREG_19 4.2 / 4.3: bind a 19 file to the committed t19_rank.py apply report next to it, its SET seam and
    local seam records, test02's real R20 run, the full raw T1 (PASS or NORMALIZED), H8R, the 09 and 19 adoptions,
    every committed input it lists and K6-19. Returns check_reports' shape plus 'test19'."""
    out, problems = {}, []
    rpath = os.path.join(t19_dir(), T19_REPORT)
    rel, rep = _committed_json(rpath, "the test19 apply report (tools/t19_rank.py apply, docs/TESTDAY_19.md)")
    out[rel] = sha256_file(rpath)
    cond, probs = t19_report_problems(rep, rel, path, sha)
    problems += probs
    for srel, what, need_pass in ((T19_SET_SEAM, "the test19 SET seam record (recorded only)", False),
                                  (T19_SEAM_LOCAL, "the test19 local seam record", True)):
        p = repo_path(srel)
        srel_, obj = _committed_json(p, what)
        out[srel_] = sha256_file(p)
        if obj.get("mode") != "real" or (need_pass and obj.get("passed") is not True):
            problems.append(f"{srel_}: passed {obj.get('passed')!r}, mode {obj.get('mode')!r}")
    t2 = os.path.join(REPO, "results", "test02", "rank", REPORT_OF["rank"])
    t2rel, t2rep = _committed_json(t2, "the test02 rank apply report (R20 goes first, PREREG_19 4.3)")
    out[t2rel] = sha256_file(t2)
    if t2rep.get("mode") != "real" or t2rep.get("fallback_used") is not False \
            or (t2rep.get("seam_local") or {}).get("passed") is not True:
        problems.append(f"{t2rel}: mode {t2rep.get('mode')!r}, fallback_used {t2rep.get('fallback_used')!r}: 19 needs "
                        f"test02's passed real-mode R20 run (FAIL_OPENING: 19 never plays)")
    t1_path = os.path.join(REPO, "results", "test02", "checks", comp, "t1.json")
    trel, t1 = _committed_json(t1_path, "the T1 result")
    out[trel] = sha256_file(t1_path)
    problems += t1_problems(t1, trel)
    v = t1.get("verdict")
    if v not in T1_UPLOAD_OK:
        problems.append(f"{trel}: verdict {v!r}; 19 plays only with T1 PASS or NORMALIZED (PREREG_19 4.3)")
    problems += t08_h8r_problems()
    problems += t09_adoption_problems()
    problems += t19_adoption_problems()
    problems += t19_inputs_problems(rep.get("inputs_bound"))
    h, t = git_commit_time(rel)
    k6_ok = t is not None and parse_time(t, "commit time") <= K6_19
    if not k6_ok:
        problems.append(f"{rel} was last committed at {t} ({h}), not by K6-19 {K6_19.isoformat()}: no 19 this season "
                        f"(PREREG_19 5); the final choice follows PREREG_09 5.3")
    if problems:
        raise LogError("the file is not backed by a usable 19 run (PREREG_19 4.2, 4.3):\n - " + "\n - ".join(problems))
    diffs = rep.get("rule_b_differences") or {}
    rec = dict(cond, combo=T19_COMBO_ID, adoption=P19_ADOPTED, report=rel, report_sha256=out[rel],
               k6_19={"commit": h, "committed_at": t, "deadline": K6_19.isoformat(), "ok": k6_ok},
               t09_report_sha256=(rep.get("inputs_bound") or {}).get(T19_09_REPORT),
               earliest_p1_chunk_created_utc=rep.get("earliest_p1_chunk_created_utc"),
               inputs_bound_checked=len(rep.get("inputs_bound") or {}),
               public_rule_b_differences={k: (diffs.get(k) or {}).get("public") for k in ("09", "08", "07", "04",
                                                                                          "R20")},
               rule_b_differences_all={k: (diffs.get(k) or {}).get("all") for k in ("09", "08", "07", "04", "R20")})
    return {"reports": out, "t1_verdict": v, "fallback_used": False, "test19": rec}


def method_for_t19(given):
    """A 19 file is recorded as Q19, never with R20's, 04's, 07's, 08's or 09's label."""
    if given and not re.search(r"(?<![A-Za-z0-9])%s(?![0-9A-Za-z])" % T19_COMBO_ID, given):
        raise LogError(f"--method {given!r} does not name {T19_COMBO_ID}; a results/test19/rank file is the 19 "
                       f"combination (default {T19_METHOD!r})")
    return given or T19_METHOD


def t19_entries(log):
    return [x for x in log["uploads"] if x.get("comp") == "rank" and x.get("test19")]


def t19_bound_09(log, t19):
    """The uploaded 09 entry whose report is the one the 19 report binds (its t09_apply_report.json SHA-256, LF)."""
    want = t19.get("t09_report_sha256")
    for x in uploads_of(log, "rank"):
        t9_ = x.get("test09") or {}
        rp = repo_path(t9_.get("report") or "") if t9_.get("report") else ""
        if t9_ and rp and os.path.isfile(rp) and sha256_bound(rp) == want:
            return x
    return None


def prepare_order_t19(log, t19):
    """PREREG_19 4.3 order: R20's first upload and the 09 entry the 19 report binds are logged as uploaded (that 09
    still meets PREREG_09 5.3, which needs its 08, 07 and 04 uploaded), and every P1 chunk was written after R20's
    first upload."""
    r20 = r20_first_upload(log)
    if r20 is None:
        raise LogError(f"PREREG_19 4.3: log R20's first upload ({T04_R20_FILE}, status uploaded) before preparing a 19 "
                       f"upload (docs/TESTDAY.md step 8)")
    e9 = t19_bound_09(log, t19)
    if e9 is None:
        raise LogError("PREREG_19 4.3: the 09 upload the 19 report binds (results/test09/rank/t09_apply_report.json) is "
                       "not logged as uploaded: upload 04, 07, 08 and 09 first (docs/TESTDAY_09.md)")
    ok9, why9 = t09_entry_eligible(e9, log)
    if not ok9:
        raise LogError(f"PREREG_19 4.3: the bound 09 entry #{e9['id']} no longer meets PREREG_09 5.3 "
                       f"({'; '.join(why9)[:600]}): 19 is not used")
    early = t19.get("earliest_p1_chunk_created_utc")
    if not early or parse_time(early, "P1 chunk time") <= parse_time(r20["uploaded_at"], "uploaded_at"):
        raise LogError(f"PREREG_19 4.2: P1 computed at {early}, not after R20's first upload ({r20['uploaded_at']})")
    t9_ = e9.get("test09") or {}
    t19["r20_first_upload_id"], t19["t09_entry_id"] = r20["id"], e9["id"]
    t19["t08_entry_id"] = t9_.get("t08_entry_id")
    t19["t07_entry_id"] = t9_.get("t07_entry_id")
    t19["t04_entry_id"] = t9_.get("t04_entry_id")


def print_t19_ok_lines(t19):
    d = t19.get("public_rule_b_differences") or {}
    print(f"- 這一份是 19（Q19），不是 09（{T19_09_REPORT}）、08、07、04 也不是 R20（{T04_R20_FILE}）；public 題規則 B 等級和 "
          f"09 不同 {d.get('09')} 題、和 08 不同 {d.get('08')} 題、和 R20 不同 {d.get('R20')} 題")
    print(f"- 19 的方法：{T19_METHOD}；R20 的第一次上傳是 #{t19.get('r20_first_upload_id')}，綁定的 09 是 "
          f"#{t19.get('t09_entry_id')}（它綁的 08 是 #{t19.get('t08_entry_id')}）；{P19_ADOPTED}")
    print("- 19 的檢查：Q09 對得上 09、本機接縫都通過；SET 方向 ρ "
          + "、".join(f"{s} {v:.4f}" for s, v in (t19.get("rho") or {}).items() if _is_num(v))
          + f"（門檻 {t19.get('rho_threshold'):.4f}）；SET 的 GPU 接縫只記錄（{t19.get('set_seam_record_only')}）")
    print(f"- K6-19：19 的報告在 {t19['k6_19']['committed_at']} commit（期限 11/19 23:59）")
    print("- 注意：19 和 09 一樣是沒在驗證集打過分的組合（PREREG_19 第 3 節的偏離），報告要寫明")


def t19_entry_eligible(x, log):
    """PREREG_19 4.3 for one logged 19 entry, re-checked now: H8R, the 09 and 19 adoptions, the report's last commit
    is the one recorded at prepare and is by K6-19, T1 PASS / NORMALIZED, the recomputed 4.2 conditions, the report
    and the file unchanged, every bound input unchanged, the bound 09 entry still meets PREREG_09 5.3 with the same
    report, and the P1 chunks later than R20's first upload. -> (ok, reasons)."""
    t19, why = x.get("test19") or {}, []
    if x.get("t1_verdict") not in T1_UPLOAD_OK:
        why.append(f"T1 {x.get('t1_verdict')!r}")
    why += t08_h8r_problems()
    why += t09_adoption_problems()
    why += t19_adoption_problems()
    rel_ = t19.get("report") or ""
    rep = None
    rp = repo_path(rel_) if rel_ else ""
    if not rel_ or not os.path.isfile(rp) or sha256_file(rp) != t19.get("report_sha256"):
        why.append(f"{rel_ or 'the 19 apply report'} changed since prepare ({T19_REDO_HINT})")
    else:
        with open(refuse_forbidden(rp), encoding="utf-8") as f:
            rep = json.load(f)
    fp = repo_path(x.get("file") or "") if x.get("file") else ""
    if not fp or not os.path.isfile(fp) or sha256_file(fp) != x.get("sha256"):
        why.append(f"{x.get('file')} changed since prepare")
    k6 = t19.get("k6_19") or {}
    h, t = git_commit_time(rel_) if rel_ else (None, None)
    if not k6.get("ok") or h is None or h != k6.get("commit") or parse_time(t, "commit time") > K6_19:
        why.append(f"K6-19: the report's last commit is {h} at {t}; prepare recorded {k6.get('commit')} at "
                   f"{k6.get('committed_at')} (deadline {K6_19.isoformat()})")
    if rep is not None:
        why += t19_report_problems(rep, rel_, fp if fp and os.path.isfile(fp) else None, x.get("sha256"))[1]
        why += t19_inputs_problems(rep.get("inputs_bound"))
    e9 = next((y for y in log["uploads"] if y["id"] == t19.get("t09_entry_id")), None)
    if e9 is None or e9.get("status") != "uploaded":
        why.append(f"the bound 09 entry #{t19.get('t09_entry_id')} is not logged as uploaded")
    else:
        rp9 = repo_path((e9.get("test09") or {}).get("report") or "")
        if not os.path.isfile(rp9) or sha256_bound(rp9) != t19.get("t09_report_sha256"):
            why.append(f"the 09 report of #{e9['id']} is not the one the 19 report binds")
        ok9, why9 = t09_entry_eligible(e9, log)
        if not ok9:
            why.append(f"the bound 09 #{e9['id']} no longer meets PREREG_09 5.3: {'; '.join(why9)[:300]}")
    r20 = r20_first_upload(log)
    early = t19.get("earliest_p1_chunk_created_utc")
    if r20 is None or not early or parse_time(early, "P1 chunk time") <= parse_time(r20["uploaded_at"], "uploaded_at"):
        why.append(f"P1 chunks ({early}) are not later than R20's first upload ({(r20 or {}).get('uploaded_at')})")
    return not why, why


def t19_eligibility(log):
    return {x["id"]: t19_entry_eligible(x, log) for x in t19_entries(log)}


def head_has_p19_revision_row(n, head=None):
    head = git_head_text(P19_DOC) if head is None else head
    return bool(head) and any(line.startswith(f"| rev {n} |") for line in head.splitlines())


def p19_revision(reason, overridden, what):
    """Overriding a registered 19 decision (PREREG_19 4.3): a reason naming 'PREREG_19 rev N' (N >= 4) whose row is at
    HEAD of the committed, clean docs/PREREG_19_SET_MODEL.md and was added after every overridden record."""
    text = (reason or "").strip()
    m = re.search(r"PREREG_19\s+rev\s+(\d+)", text)
    if not m:
        raise LogError(f"{what}: needs --not-19-reason \"PREREG_19 rev N: <the program error and the condition of "
                       f"PREREG_19 4.3 it breaks>\" citing a committed revision (N >= {P19_MIN_REV}); a reason "
                       f"without a revision is refused")
    n = int(m.group(1))
    if n < P19_MIN_REV:
        raise LogError(f"{what}: --not-19-reason cites PREREG_19 rev {n}; an override needs rev {P19_MIN_REV} or later "
                       f"(rev 1 to rev 3 are the registration, the freeze and the test-day procedure)")
    st = git_file_state(P19_DOC)
    if not st["tracked"] or not st["clean"]:
        raise LogError(f"{what}: {P19_DOC} must be committed and unchanged (tracked={st['tracked']}, "
                       f"clean={st['clean']})")
    row = f"| rev {n} |"
    if not head_has_p19_revision_row(n):
        raise LogError(f"{what}: {P19_DOC} at HEAD has no row starting {row!r}; commit the revision first")
    h, t = git_row_added(P19_DOC, row)
    if h is None:
        raise LogError(f"{what}: no commit found that added {row!r} to {P19_DOC}")
    at = parse_time(t, "commit time")
    early = [w for w in overridden if at <= parse_time(w, "time")]
    if early:
        raise LogError(f"{what}: PREREG_19 rev {n} was committed at {t} ({h[:12]}), not after the records it "
                       f"overrides ({', '.join(early)}); a revision must come after the decision it overturns")
    return {"rev": n, "row": row, "doc": P19_DOC, "commit": h, "committed_at": t, "reason": text,
            "overridden_at": list(overridden)}


def prepare_rule_t19(log, comp, bound, not_19_reason):
    """PREREG_19 4.3 at prepare: a 09 / 08 / 07 / 04 / R20 / R10 file while a logged 19 entry meets 4.3 needs the
    revision check of --not-19-reason. None when the log has no 19 entry (everything else is exactly as before)."""
    if bound.get("test14"):              # PREREG_14 7: a 14 file is judged on its base in prepare_order_t14
        if (not_19_reason or "").strip():
            raise LogError("--not-19-reason does not apply to a 14 file")
        return None
    if bound.get("test25"):              # PREREG_25 rev 3: a 25 file is judged by prepare_rule_t25
        if (not_19_reason or "").strip():
            raise LogError("--not-19-reason does not apply to a 25 file")
        return None
    reason = (not_19_reason or "").strip()
    if comp != "rank" or bound.get("test19"):
        if reason:
            raise LogError("--not-19-reason applies to a competition-2 09 / 08 / 07 / 04 / R20 / R10 file only")
        return None
    elig = t19_eligibility(log)
    if not elig:
        if reason:
            raise LogError("--not-19-reason: the log has no 19 entry to set aside")
        return None
    ok_ids = [i for i, (ok, _w) in elig.items() if ok]
    rec = {"rule": "PREREG_19 4.3", "adoption": P19_ADOPTED, "eligible_19_entries": ok_ids,
           "not_eligible_19_entries": {str(i): w for i, (ok, w) in elig.items() if not ok}, "checked_at": now_iso(),
           "not_19_reason": reason or None}
    if ok_ids:
        when = [x["prepared_at"] for x in t19_entries(log) if x["id"] in ok_ids]
        rec["revision"] = p19_revision(reason, when, f"19 entries {ok_ids} meet PREREG_19 4.3, so a 09 / 08 / 07 / "
                                                     f"04 / R20 / R10 file is not the final choice")
    return rec


def final_rule_t19(log, e, not_19_reason):
    """PREREG_19 4.3 for competition 2's final choice. None when no 19 entry is logged. A 19 entry is final only
    while it holds; while some 19 entry holds, a 09 / 08 / 07 / 04 / R20 / R10 final needs --not-19-reason
    (p19_revision)."""
    reason = (not_19_reason or "").strip()
    if e.get("test25"):                  # PREREG_25 rev 3: a 25 final is judged by final_rule_t25, never as 19
        if reason:
            raise LogError("--not-19-reason does not apply to a 25 entry")
        return None
    if e["comp"] != "rank":
        if reason:
            raise LogError("--not-19-reason applies to competition 2 (rank) only")
        return None
    elig = t19_eligibility(log)
    if not elig:
        if reason:
            raise LogError("--not-19-reason: the log has no 19 entry to set aside")
        return None
    ok_ids = [i for i, (ok, _w) in elig.items() if ok]
    rec = {"rule": "PREREG_19 4.3", "adoption": P19_ADOPTED, "eligible_19_entries": ok_ids,
           "not_eligible_19_entries": {str(i): w for i, (ok, w) in elig.items() if not ok}, "checked_at": now_iso()}
    if e.get("test19"):
        ok, why = elig[e["id"]]
        if not ok:
            raise LogError(f"entry #{e['id']} (19) does not meet PREREG_19 4.3 now ({'; '.join(why)[:900]}): the final "
                           f"choice follows PREREG_09 5.3 (09, else 08, else 07, else 04, else R20, R10 on "
                           f"FAIL_OPENING); upload it again so that it is the last upload (docs/TESTDAY_19.md step 5)")
        if reason:
            raise LogError("--not-19-reason does not apply when the 19 entry is chosen")
        return dict(rec, chosen="19", not_19_reason=None)
    rec.update(chosen="09" if e.get("test09") else "08" if e.get("test08") else "07" if e.get("test07") else
               "04" if e.get("test04") else ("R10" if e.get("fallback_used") else "R20"), not_19_reason=reason or None)
    if ok_ids:
        when = [x["prepared_at"] for x in t19_entries(log) if x["id"] in ok_ids]
        rec["revision"] = p19_revision(reason, when, f"19 entries {ok_ids} meet every condition of PREREG_19 4.3, "
                                                     f"so the final choice is 19")
    return rec


def t19_final_holds(log, fin):
    """The final choice is a 19 entry recorded as chosen 19 that still meets PREREG_19 4.3 (False without 19
    entries)."""
    if not fin or not t19_entries(log):
        return False
    e = next((x for x in log["uploads"] if x["id"] == fin.get("id")), None)
    if not e or not e.get("test19") or (fin.get("prereg19_4_3") or {}).get("chosen") != "19":
        return False
    return t19_entry_eligible(e, log)[0]


def t19_status_key(log):
    """{'t19_eligible': [...]} for the 04 / 07 / 08 / 09 status, only when the log has 19 entries (else {})."""
    if not t19_entries(log):
        return {}
    return {"t19_eligible": [i for i, (ok, _w) in t19_eligibility(log).items() if ok]}


def t19_status(log, fin, last):
    elig = t19_eligibility(log)
    ok_ids = [i for i, (ok, _w) in elig.items() if ok]
    by_id = {x["id"]: x for x in log["uploads"]}
    fin_e = by_id.get((fin or {}).get("id"))
    rule = (fin or {}).get("prereg19_4_3") or {}
    final_is_19 = bool(fin_e and fin_e.get("test19"))
    rev = rule.get("revision") or {}

    def overridden_by_rev(i):
        """A non-19 final sets a 19 entry aside only when the revision recorded with it covered that entry and was
        committed after its prepare (the time rule of final, re-checked now; as t09_status)."""
        prep = (by_id.get(i) or {}).get("prepared_at")
        if i not in (rule.get("eligible_19_entries") or []) or not rev.get("committed_at") or not prep:
            return False
        try:
            return parse_time(rev["committed_at"], "revision commit time") > parse_time(prep, "prepared_at")
        except LogError:
            return False
    final_ok = (final_is_19 and fin_e["id"] in ok_ids and rule.get("chosen") == "19") or \
        (bool(fin) and not final_is_19 and bool(rev) and all(overridden_by_rev(i) for i in ok_ids))
    final_ok = final_ok or t25_final_holds(log, fin)   # PREREG_25 4.3: a holding 25 final is not a 19 violation
    last_bad = bool(last and last.get("test19") and not elig.get(last["id"], (True, []))[0])
    final_bad = bool(final_is_19 and not elig[fin_e["id"]][0])
    return {"entries": sorted(elig), "eligible": ok_ids, "not_eligible": {str(i): w for i, (ok, w) in elig.items()
                                                                          if not ok},
            "final_set": bool(fin), "final_is_19": final_is_19, "violation": bool(ok_ids) and bool(fin) and not final_ok,
            "redo_09": not ok_ids and (last_bad or final_bad)}


def t19_status_lines(st):
    lines = [f"  19（PREREG_19 4.3）：紀錄 {st['entries']}；現在符合條件的 {st['eligible'] or '沒有'}；最終選擇"
             f"{'是 19' if st['final_is_19'] else '不是 19'}"]
    for i, why in st["not_eligible"].items():
        text = "；".join(why)
        lines.append(f"  #{i} 不符合：{text[:600]}{'…' if len(text) > 600 else ''}")
    if st["violation"]:
        lines.append("  違反 PREREG_19 4.3：有符合條件的 19，最終選擇卻不是它（沒有引用修訂，或引用的修訂沒有涵蓋它）；照 "
                     "docs/TESTDAY_19.md 第 5 步重跑 final")
    if st["redo_09"]:
        lines.append("  照 PREREG_19 4.3：現在沒有符合的 19，final 改照 PREREG_09 5.3（09，否則 08，否則 07，否則 04，否則 "
                     "R20），再 prepare 並重傳那一份（docs/TESTDAY_19.md 第 5 步）")
    return lines


def t19_public_lines(log, e):
    """PREREG_19 4.3: 19, 09, 08, 07, 04 and R20 share submission_rank.csv. Two equal public scores (|d| < 1e-9)
    between a 19 entry and a 09 / 08 / 07 / 04 / R20 entry while the 19 report counts public-question rule-B
    differences > 0 between them: a hint that the wrong file was uploaded. Empty when the log has no 19 entry."""
    if e["comp"] != "rank" or e.get("public_score") is None or not t19_entries(log):
        return []
    lines = []
    others = [x for x in uploads_of(log, "rank") if x["id"] != e["id"] and x.get("public_score") is not None
              and abs(x["public_score"] - e["public_score"]) < 1e-9]
    for x in others:
        a, b = (e, x) if e.get("test19") else (x, e)
        if a.get("test25") or b.get("test25"):   # PREREG_25 rev 3: a 25 entry is warned by t25_public_lines
            continue
        if not a.get("test19") or b.get("test19"):
            continue
        if b.get("test14"):              # PREREG_14: a 14 entry is warned by t14_public_lines
            continue
        kind = "09" if b.get("test09") else "08" if b.get("test08") else "07" if b.get("test07") else \
            "04" if b.get("test04") else "R20"
        n = ((a.get("test19") or {}).get("public_rule_b_differences") or {}).get(kind)
        if _is_count(n) and n > 0:
            lines.append(f"WARNING (PREREG_19 4.3): #{a['id']}（19）和 #{b['id']}（{kind}）的 public 分數完全相同，但兩份在 "
                         f"public 題有 {n} 題的規則 B 等級不同：可能傳錯檔（六份同名）：核對 AIdea 的上傳紀錄")
    return lines


# ---------------------------------------------------------------- PREREG_25: test25 (competition 2, Q25 = Q19 with S25)
def t25_dir():
    return os.path.join(REPO, *T25_DIR)


def is_t25_file(comp, path):
    """A competition-2 file in results/test25/rank/ is a 25 file (only tools/t25_rank.py apply writes there)."""
    if comp != "rank":
        return False
    here = os.path.realpath(os.path.dirname(os.path.abspath(path)))
    return os.path.normcase(here) == os.path.normcase(os.path.realpath(t25_dir()))


def t25_adoption_problems():
    """PREREG_25 2.4, re-read from the committed adoption record of prereg25: its LF SHA-256 is the pinned one, Q25,
    usable true after the independent audit (finalized)."""
    p = repo_path(T25_ADOPTION)
    if not os.path.isfile(p):
        return [f"{T25_ADOPTION} is missing: 25 was not adopted (PREREG_25 2.4)"]
    st = git_file_state(T25_ADOPTION)
    if not st["tracked"] or not st["clean"]:
        return [f"{T25_ADOPTION} is not committed and clean"]
    if sha256_bound(p) != T25_ADOPTION_SHA:
        return [f"{T25_ADOPTION} is not the committed adoption record ({sha256_bound(p)[:12]} vs "
                f"{str(T25_ADOPTION_SHA)[:12]})"]
    with open(refuse_forbidden(p), encoding="utf-8") as f:
        s = json.load(f)
    if s.get("combo") != T25_COMBO_ID or s.get("usable") is not True or not s.get("finalized"):
        return [f"{T25_ADOPTION}: combo {s.get('combo')!r}, usable {s.get('usable')!r}, finalized "
                f"{s.get('finalized')!r} (PREREG_25 2.4: 25 is not adopted)"]
    return []


def t25_inputs_problems(bound):
    """Every input the 25 apply report lists under inputs_bound is committed, clean and still has its SHA-256."""
    if not isinstance(bound, dict) or not bound:
        return [f"the 25 apply report lists no inputs_bound ({T25_REDO_HINT})"]
    probs = []
    missing = [r for r in T25_INPUTS_REQUIRED if r not in bound]
    if missing:
        probs.append(f"inputs_bound lacks {missing}")
    for rel_, want in sorted(bound.items()):
        s = str(rel_)
        if os.path.isabs(s) or "\\" in s or ".." in s.split("/") or s.startswith("/"):
            probs.append(f"inputs_bound key {s!r} is not a repo-relative path")
            continue
        p = repo_path(s)
        if not os.path.isfile(p):
            probs.append(f"{s} is missing ({T25_REDO_HINT})")
            continue
        st = git_file_state(s)
        if not st["tracked"] or not st["clean"]:
            probs.append(f"{s} is not committed and clean (先 commit；內容若已改變：{T25_REDO_HINT})")
            continue
        got = sha256_bound(p)
        if got != want:
            probs.append(f"{s} is {got[:12]}, the 25 apply report bound {str(want)[:12]} ({T25_REDO_HINT})")
    return probs


def _t25_repro_problems(q, ref_sha, what):
    if q.get("equal") is not True or q.get("reference_sha256_lf") != ref_sha or \
            not _is_num(q.get("max_abs_diff")) or q["max_abs_diff"] > T25_TOL or \
            q.get("rule_a_differences") != 0 or q.get("rule_b_differences") != 0:
        return [f"{what} recomputed is not the committed file ({str(q.get('reference_sha256_lf'))[:12]} vs "
                f"{str(ref_sha)[:12]}; max |dP| {q.get('max_abs_diff')!r}, rule A / B differences "
                f"{q.get('rule_a_differences')!r} / {q.get('rule_b_differences')!r})"]
    return []


def t25_conditions(rep):
    """PREREG_25 4.2 recomputed from the numbers of the 25 apply report (never from its own verdicts).
    -> (record, problems)."""
    probs = []
    bound = rep.get("inputs_bound") or {}
    q9 = rep.get("q09_vs_09") or {}
    probs += _t25_repro_problems(q9, bound.get(T25_09_PROBS), "Q09")
    if q9.get("p1_ok_equal_t09") is not True or q9.get("usable_equal_t09") is not True or \
            not _is_num(q9.get("lp_p1e_max_abs_diff_vs_t09")) or q9["lp_p1e_max_abs_diff_vs_t09"] > T25_TOL:
        probs.append(f"Q09's usable games / lp_P1TO_E differ from 09's test_lp_p1e.npz ({q9.get('p1_ok_equal_t09')!r}"
                     f" / {q9.get('usable_equal_t09')!r} / {q9.get('lp_p1e_max_abs_diff_vs_t09')!r})")
    q19 = rep.get("q19_vs_19") or {}
    probs += _t25_repro_problems(q19, bound.get(T25_19_PROBS), "Q19")
    lp = q19.get("lp_set19") or {}
    if not (lp.get("question_id_equal") is True and lp.get("has_equal") is True and lp.get("usable_equal") is True
            and _is_num(lp.get("lp_max_abs_diff")) and lp["lp_max_abs_diff"] <= T25_TOL):
        probs.append(f"lp_SET19 recomputed is not 19's test_lp_set.npz: {lp}")
    thr = max(0.5 * T25_RHO_MONO, T25_RHO_FLOOR)
    rho = {s: ((rep.get("set_direction") or {}).get(s) or {}).get("rho") for s in T25_SETS}
    for s, v in rho.items():
        if not _is_num(v) or v < thr:
            probs.append(f"SET25 direction of {s}: rho {v!r} < {thr} (no 25)")
    return {"rho": rho, "rho_threshold": thr, "q09_max_abs_diff": q9.get("max_abs_diff"),
            "q19_max_abs_diff": q19.get("max_abs_diff"),
            "set_seam_record_only": (rep.get("set_seam_record_only") or {}).get("max_abs_lp_set")}, probs


def t25_report_problems(rep, rel, path, sha):
    """The 25 apply report's own conditions (format, mode, exit, combo, P1 sets, check_submission, this file) and
    PREREG_25 4.2 recomputed. -> (conditions, problems)."""
    probs = []
    if rep.get("format") != T25_FORMAT or rep.get("mode") != "real":
        probs.append(f"{rel}: format {rep.get('format')!r}, mode {rep.get('mode')!r} (a real-mode {T25_FORMAT})")
    if rep.get("exit_code") != 0 or rep.get("usable") is not True:
        probs.append(f"{rel}: exit_code {rep.get('exit_code')!r}, usable {rep.get('usable')!r}")
    combo = rep.get("combo") or {}
    if combo.get("id") != T25_COMBO_ID or tuple(combo.get("sources") or ()) != T25_COMBO_SOURCES:
        probs.append(f"{rel}: combination {combo.get('id')!r} {combo.get('sources')!r} is not Q25")
    fake = [s for s, r in (rep.get("p1_sets") or {}).items() if (r or {}).get("fake_from_val")]
    if fake or set(rep.get("p1_sets") or {}) != set(T25_P1_SETS):
        probs.append(f"{rel}: P1 sets {sorted(rep.get('p1_sets') or {})}, drill records {fake}")
    cs = rep.get("check_submission") or {}
    if cs.get("mode") != "official" or cs.get("errors"):
        probs.append(f"{rel}: check_submission {cs.get('mode')!r} with errors {cs.get('errors')}")
    want = (rep.get("outputs_sha256") or {}).get(FILE_NAMES["rank"])
    if want != sha:
        try:
            hint = lf_hint(path, want, repo_rel(path)) if path and os.path.exists(path) else ""
        except LogError:
            hint = ""
        probs.append(f"{rel}: records another {FILE_NAMES['rank']} than this file ({str(sha)[:12]})" + hint)
    cond, cprobs = t25_conditions(rep)
    return cond, probs + cprobs


def check_reports_t25(comp, path, sha):
    """PREREG_25 4.2 / 4.3: bind a 25 file to the committed t25_rank.py apply report next to it, its SET25 seam record,
    test02's real R20 run, the full raw T1 (PASS or NORMALIZED), H8R, the 09, 19 and 25 adoptions, every committed
    input it lists and K6-25. Returns check_reports' shape plus 'test25'."""
    out, problems = {}, []
    rpath = os.path.join(t25_dir(), T25_REPORT)
    rel, rep = _committed_json(rpath, "the test25 apply report (tools/t25_rank.py apply, docs/TESTDAY_25.md)")
    out[rel] = sha256_file(rpath)
    cond, probs = t25_report_problems(rep, rel, path, sha)
    problems += probs
    p = repo_path(T25_SET_SEAM)
    srel_, obj = _committed_json(p, "the test25 SET25 seam record (recorded only)")
    out[srel_] = sha256_file(p)
    if obj.get("mode") != "real":
        problems.append(f"{srel_}: mode {obj.get('mode')!r}")
    t2 = os.path.join(REPO, "results", "test02", "rank", REPORT_OF["rank"])
    t2rel, t2rep = _committed_json(t2, "the test02 rank apply report (R20 goes first, PREREG_25 4.3)")
    out[t2rel] = sha256_file(t2)
    if t2rep.get("mode") != "real" or t2rep.get("fallback_used") is not False \
            or (t2rep.get("seam_local") or {}).get("passed") is not True:
        problems.append(f"{t2rel}: mode {t2rep.get('mode')!r}, fallback_used {t2rep.get('fallback_used')!r}: 25 needs "
                        f"test02's passed real-mode R20 run (FAIL_OPENING: 25 never plays)")
    t1_path = os.path.join(REPO, "results", "test02", "checks", comp, "t1.json")
    trel, t1 = _committed_json(t1_path, "the T1 result")
    out[trel] = sha256_file(t1_path)
    problems += t1_problems(t1, trel)
    v = t1.get("verdict")
    if v not in T1_UPLOAD_OK:
        problems.append(f"{trel}: verdict {v!r}; 25 plays only with T1 PASS or NORMALIZED (PREREG_25 4.3)")
    problems += t08_h8r_problems()
    problems += t09_adoption_problems()
    problems += t19_adoption_problems()
    problems += t25_adoption_problems()
    problems += t25_inputs_problems(rep.get("inputs_bound"))
    h, t = git_commit_time(rel)
    k6_ok = t is not None and parse_time(t, "commit time") <= K6_25
    if not k6_ok:
        problems.append(f"{rel} was last committed at {t} ({h}), not by K6-25 {K6_25.isoformat()}: no 25 this season "
                        f"(PREREG_25 5); the final choice follows PREREG_19 4.3")
    if problems:
        raise LogError("the file is not backed by a usable 25 run (PREREG_25 4.2, 4.3):\n - " + "\n - ".join(problems))
    diffs = rep.get("rule_b_differences") or {}
    kinds = ("19", "09", "08", "07", "04", "R20")
    rec = dict(cond, combo=T25_COMBO_ID, adoption=P25_ADOPTED, report=rel, report_sha256=out[rel],
               k6_25={"commit": h, "committed_at": t, "deadline": K6_25.isoformat(), "ok": k6_ok},
               t19_report_sha256=(rep.get("inputs_bound") or {}).get(T25_19_REPORT),
               earliest_p1_chunk_created_utc=rep.get("earliest_p1_chunk_created_utc"),
               inputs_bound_checked=len(rep.get("inputs_bound") or {}),
               public_rule_b_differences={k: (diffs.get(k) or {}).get("public") for k in kinds},
               private_rule_b_differences={k: (diffs.get(k) or {}).get("private") for k in kinds},
               rule_b_differences_all={k: (diffs.get(k) or {}).get("all") for k in kinds})
    return {"reports": out, "t1_verdict": v, "fallback_used": False, "test25": rec}


def method_for_t25(given):
    """A 25 file is recorded as Q25, never with R20's, 04's, 07's, 08's, 09's or 19's label."""
    if given and not re.search(r"(?<![A-Za-z0-9])%s(?![0-9A-Za-z])" % T25_COMBO_ID, given):
        raise LogError(f"--method {given!r} does not name {T25_COMBO_ID}; a results/test25/rank file is the 25 "
                       f"combination (default {T25_METHOD!r})")
    return given or T25_METHOD


def t25_entries(log):
    return [x for x in log["uploads"] if x.get("comp") == "rank" and x.get("test25")]


def t25_bound_19(log, t25):
    """The uploaded 19 entry whose report is the one the 25 report binds (its t19_apply_report.json SHA-256, LF)."""
    want = t25.get("t19_report_sha256")
    for x in uploads_of(log, "rank"):
        t19_ = x.get("test19") or {}
        rp = repo_path(t19_.get("report") or "") if t19_.get("report") else ""
        if t19_ and rp and os.path.isfile(rp) and sha256_bound(rp) == want:
            return x
    return None


def prepare_order_t25(log, t25):
    """PREREG_25 4.3 order: R20's first upload and the 19 entry the 25 report binds are logged as uploaded (that 19
    still meets PREREG_19 4.3, which needs its 09, 08, 07 and 04 uploaded), and every P1 chunk was written after R20's
    first upload."""
    r20 = r20_first_upload(log)
    if r20 is None:
        raise LogError(f"PREREG_25 4.3: log R20's first upload ({T04_R20_FILE}, status uploaded) before preparing a 25 "
                       f"upload (docs/TESTDAY.md step 8)")
    e19 = t25_bound_19(log, t25)
    if e19 is None:
        raise LogError("PREREG_25 4.3: the 19 upload the 25 report binds (results/test19/rank/t19_apply_report.json) is "
                       "not logged as uploaded: upload 19 first (docs/TESTDAY_19.md)")
    ok19, why19 = t19_entry_eligible(e19, log)
    if not ok19:
        raise LogError(f"PREREG_25 4.3: the bound 19 entry #{e19['id']} no longer meets PREREG_19 4.3 "
                       f"({'; '.join(why19)[:600]}): 25 is not used")
    early = t25.get("earliest_p1_chunk_created_utc")
    if not early or parse_time(early, "P1 chunk time") <= parse_time(r20["uploaded_at"], "uploaded_at"):
        raise LogError(f"PREREG_25 4.2: P1 computed at {early}, not after R20's first upload ({r20['uploaded_at']})")
    t19_ = e19.get("test19") or {}
    t25["r20_first_upload_id"], t25["t19_entry_id"] = r20["id"], e19["id"]
    t25["t09_entry_id"] = t19_.get("t09_entry_id")


def print_t25_ok_lines(t25):
    d, dp = t25.get("public_rule_b_differences") or {}, t25.get("private_rule_b_differences") or {}
    print(f"- 這一份是 25（Q25），不是 19（{T25_19_REPORT}）、09、08、07、04 也不是 R20（{T04_R20_FILE}）；規則 B 等級和 19 "
          f"不同：public {d.get('19')} 題、private {dp.get('19')} 題；和 09 不同 public {d.get('09')} 題")
    print(f"- 25 的方法：{T25_METHOD}；R20 的第一次上傳是 #{t25.get('r20_first_upload_id')}，綁定的 19 是 "
          f"#{t25.get('t19_entry_id')}（它綁的 09 是 #{t25.get('t09_entry_id')}）；{P25_ADOPTED}")
    print("- 25 的檢查：Q09 對得上 09、Q19 對得上 19；SET25 方向 ρ "
          + "、".join(f"{s} {v:.4f}" for s, v in (t25.get("rho") or {}).items() if _is_num(v))
          + f"（門檻 {t25.get('rho_threshold'):.4f}）；SET25 的 GPU 接縫只記錄（{t25.get('set_seam_record_only')}）")
    print(f"- K6-25：25 的報告在 {t25['k6_25']['committed_at']} commit（期限 11/18 23:59）")
    print("- 注意：25 用了驗證玩家訓練、沒在驗證集打過分（PREREG_25 第 3 節的偏離），報告要寫明")


def t25_entry_eligible(x, log):
    """PREREG_25 4.3 for one logged 25 entry, re-checked now: H8R, the 09, 19 and 25 adoptions, the report's last
    commit is the one recorded at prepare and is by K6-25, T1 PASS / NORMALIZED, the recomputed 4.2 conditions, the
    report and the file unchanged, every bound input unchanged, the bound 19 entry still meets PREREG_19 4.3 with the
    same report, and the P1 chunks later than R20's first upload. -> (ok, reasons)."""
    t25, why = x.get("test25") or {}, []
    if x.get("t1_verdict") not in T1_UPLOAD_OK:
        why.append(f"T1 {x.get('t1_verdict')!r}")
    why += t08_h8r_problems()
    why += t09_adoption_problems()
    why += t19_adoption_problems()
    why += t25_adoption_problems()
    rel_ = t25.get("report") or ""
    rep = None
    rp = repo_path(rel_) if rel_ else ""
    if not rel_ or not os.path.isfile(rp) or sha256_file(rp) != t25.get("report_sha256"):
        why.append(f"{rel_ or 'the 25 apply report'} changed since prepare ({T25_REDO_HINT})")
    else:
        with open(refuse_forbidden(rp), encoding="utf-8") as f:
            rep = json.load(f)
    fp = repo_path(x.get("file") or "") if x.get("file") else ""
    if not fp or not os.path.isfile(fp) or sha256_file(fp) != x.get("sha256"):
        why.append(f"{x.get('file')} changed since prepare")
    k6 = t25.get("k6_25") or {}
    h, t = git_commit_time(rel_) if rel_ else (None, None)
    if not k6.get("ok") or h is None or h != k6.get("commit") or parse_time(t, "commit time") > K6_25:
        why.append(f"K6-25: the report's last commit is {h} at {t}; prepare recorded {k6.get('commit')} at "
                   f"{k6.get('committed_at')} (deadline {K6_25.isoformat()})")
    if rep is not None:
        why += t25_report_problems(rep, rel_, fp if fp and os.path.isfile(fp) else None, x.get("sha256"))[1]
        why += t25_inputs_problems(rep.get("inputs_bound"))
    e19 = next((y for y in log["uploads"] if y["id"] == t25.get("t19_entry_id")), None)
    if e19 is None or e19.get("status") != "uploaded":
        why.append(f"the bound 19 entry #{t25.get('t19_entry_id')} is not logged as uploaded")
    else:
        rp19 = repo_path((e19.get("test19") or {}).get("report") or "")
        if not os.path.isfile(rp19) or sha256_bound(rp19) != t25.get("t19_report_sha256"):
            why.append(f"the 19 report of #{e19['id']} is not the one the 25 report binds")
        ok19, why19 = t19_entry_eligible(e19, log)
        if not ok19:
            why.append(f"the bound 19 #{e19['id']} no longer meets PREREG_19 4.3: {'; '.join(why19)[:300]}")
    r20 = r20_first_upload(log)
    early = t25.get("earliest_p1_chunk_created_utc")
    if r20 is None or not early or parse_time(early, "P1 chunk time") <= parse_time(r20["uploaded_at"], "uploaded_at"):
        why.append(f"P1 chunks ({early}) are not later than R20's first upload ({(r20 or {}).get('uploaded_at')})")
    return not why, why


def t25_eligibility(log):
    return {x["id"]: t25_entry_eligible(x, log) for x in t25_entries(log)}


def head_has_p25_revision_row(n, head=None):
    head = git_head_text(P25_DOC) if head is None else head
    return bool(head) and any(line.startswith(f"| rev {n} |") for line in head.splitlines())


def p25_revision(reason, overridden, what):
    """Overriding a registered 25 decision (PREREG_25 4.3): a reason naming 'PREREG_25 rev N' (N >= 4) whose row is at
    HEAD of the committed, clean docs/PREREG_25_SET_MORE_PLAYERS.md and was added after every overridden record."""
    text = (reason or "").strip()
    m = re.search(r"PREREG_25\s+rev\s+(\d+)", text)
    if not m:
        raise LogError(f"{what}: needs --not-25-reason \"PREREG_25 rev N: <the program error and the condition of "
                       f"PREREG_25 4.3 it breaks>\" citing a committed revision (N >= {P25_MIN_REV}); a reason "
                       f"without a revision is refused")
    n = int(m.group(1))
    if n < P25_MIN_REV:
        raise LogError(f"{what}: --not-25-reason cites PREREG_25 rev {n}; an override needs rev {P25_MIN_REV} or later "
                       f"(rev 1 to rev 3 are the registration, the freeze and the test-day procedure)")
    st = git_file_state(P25_DOC)
    if not st["tracked"] or not st["clean"]:
        raise LogError(f"{what}: {P25_DOC} must be committed and unchanged (tracked={st['tracked']}, "
                       f"clean={st['clean']})")
    row = f"| rev {n} |"
    if not head_has_p25_revision_row(n):
        raise LogError(f"{what}: {P25_DOC} at HEAD has no row starting {row!r}; commit the revision first")
    h, t = git_row_added(P25_DOC, row)
    if h is None:
        raise LogError(f"{what}: no commit found that added {row!r} to {P25_DOC}")
    at = parse_time(t, "commit time")
    early = [w for w in overridden if at <= parse_time(w, "time")]
    if early:
        raise LogError(f"{what}: PREREG_25 rev {n} was committed at {t} ({h[:12]}), not after the records it "
                       f"overrides ({', '.join(early)}); a revision must come after the decision it overturns")
    return {"rev": n, "row": row, "doc": P25_DOC, "commit": h, "committed_at": t, "reason": text,
            "overridden_at": list(overridden)}


def prepare_rule_t25(log, comp, bound, not_25_reason):
    """PREREG_25 4.3 at prepare: a 19 / 09 / 08 / 07 / 04 / R20 / R10 file while a logged 25 entry meets 4.3 needs the
    revision check of --not-25-reason. None when the log has no 25 entry (everything else is exactly as before)."""
    reason = (not_25_reason or "").strip()
    if bound.get("test14"):              # PREREG_14 7: a 14 file is judged on its base in prepare_order_t14
        if reason:
            raise LogError("--not-25-reason does not apply to a 14 file")
        return None
    if comp != "rank" or bound.get("test25"):
        if reason:
            raise LogError("--not-25-reason applies to a competition-2 19 / 09 / 08 / 07 / 04 / R20 / R10 file only")
        return None
    elig = t25_eligibility(log)
    if not elig:
        if reason:
            raise LogError("--not-25-reason: the log has no 25 entry to set aside")
        return None
    ok_ids = [i for i, (ok, _w) in elig.items() if ok]
    rec = {"rule": "PREREG_25 4.3", "adoption": P25_ADOPTED, "eligible_25_entries": ok_ids,
           "not_eligible_25_entries": {str(i): w for i, (ok, w) in elig.items() if not ok}, "checked_at": now_iso(),
           "not_25_reason": reason or None}
    if ok_ids:
        when = [x["prepared_at"] for x in t25_entries(log) if x["id"] in ok_ids]
        rec["revision"] = p25_revision(reason, when, f"25 entries {ok_ids} meet PREREG_25 4.3, so a 19 / 09 / 08 / "
                                                     f"07 / 04 / R20 / R10 file is not the final choice")
    return rec


def final_rule_t25(log, e, not_25_reason):
    """PREREG_25 4.3 for competition 2's final choice. None when no 25 entry is logged. A 25 entry is final only
    while it holds; while some 25 entry holds, a 19 / 09 / 08 / 07 / 04 / R20 / R10 final needs --not-25-reason
    (p25_revision)."""
    reason = (not_25_reason or "").strip()
    if e["comp"] != "rank":
        if reason:
            raise LogError("--not-25-reason applies to competition 2 (rank) only")
        return None
    elig = t25_eligibility(log)
    if not elig:
        if reason:
            raise LogError("--not-25-reason: the log has no 25 entry to set aside")
        return None
    ok_ids = [i for i, (ok, _w) in elig.items() if ok]
    rec = {"rule": "PREREG_25 4.3", "adoption": P25_ADOPTED, "eligible_25_entries": ok_ids,
           "not_eligible_25_entries": {str(i): w for i, (ok, w) in elig.items() if not ok}, "checked_at": now_iso()}
    if e.get("test25"):
        ok, why = elig[e["id"]]
        if not ok:
            raise LogError(f"entry #{e['id']} (25) does not meet PREREG_25 4.3 now ({'; '.join(why)[:900]}): the final "
                           f"choice follows PREREG_19 4.3 (19, else 09, else 08, else 07, else 04, else R20, R10 on "
                           f"FAIL_OPENING); upload it again so that it is the last upload (docs/TESTDAY_25.md step 5)")
        if reason:
            raise LogError("--not-25-reason does not apply when the 25 entry is chosen")
        return dict(rec, chosen="25", not_25_reason=None)
    rec.update(chosen="19" if e.get("test19") else "09" if e.get("test09") else "08" if e.get("test08") else
               "07" if e.get("test07") else "04" if e.get("test04") else
               ("R10" if e.get("fallback_used") else "R20"), not_25_reason=reason or None)
    if ok_ids:
        when = [x["prepared_at"] for x in t25_entries(log) if x["id"] in ok_ids]
        rec["revision"] = p25_revision(reason, when, f"25 entries {ok_ids} meet every condition of PREREG_25 4.3, "
                                                     f"so the final choice is 25")
    return rec


def t25_final_holds(log, fin):
    """The final choice is a 25 entry recorded as chosen 25 that still meets PREREG_25 4.3 (False without 25
    entries)."""
    if not fin or not t25_entries(log):
        return False
    e = next((x for x in log["uploads"] if x["id"] == fin.get("id")), None)
    if not e or not e.get("test25") or (fin.get("prereg25_4_3") or {}).get("chosen") != "25":
        return False
    return t25_entry_eligible(e, log)[0]


def t25_status_key(log):
    """{'t25_eligible': [...]} for the 04 / 07 / 08 status, only when the log has 25 entries (else {})."""
    if not t25_entries(log):
        return {}
    return {"t25_eligible": [i for i, (ok, _w) in t25_eligibility(log).items() if ok]}


def t25_status(log, fin, last):
    elig = t25_eligibility(log)
    ok_ids = [i for i, (ok, _w) in elig.items() if ok]
    by_id = {x["id"]: x for x in log["uploads"]}
    fin_e = by_id.get((fin or {}).get("id"))
    rule = (fin or {}).get("prereg25_4_3") or {}
    final_is_25 = bool(fin_e and fin_e.get("test25"))
    rev = rule.get("revision") or {}

    def overridden_by_rev(i):
        """A non-25 final sets a 25 entry aside only when the revision recorded with it covered that entry and was
        committed after its prepare (the time rule of final, re-checked now; as t19_status)."""
        prep = (by_id.get(i) or {}).get("prepared_at")
        if i not in (rule.get("eligible_25_entries") or []) or not rev.get("committed_at") or not prep:
            return False
        try:
            return parse_time(rev["committed_at"], "revision commit time") > parse_time(prep, "prepared_at")
        except LogError:
            return False
    final_ok = (final_is_25 and fin_e["id"] in ok_ids and rule.get("chosen") == "25") or \
        (bool(fin) and not final_is_25 and bool(rev) and all(overridden_by_rev(i) for i in ok_ids))
    last_bad = bool(last and last.get("test25") and not elig.get(last["id"], (True, []))[0])
    final_bad = bool(final_is_25 and not elig[fin_e["id"]][0])
    return {"entries": sorted(elig), "eligible": ok_ids, "not_eligible": {str(i): w for i, (ok, w) in elig.items()
                                                                          if not ok},
            "final_set": bool(fin), "final_is_25": final_is_25, "violation": bool(ok_ids) and bool(fin) and not final_ok,
            "redo_19": not ok_ids and (last_bad or final_bad)}


def t25_status_lines(st):
    lines = [f"  25（PREREG_25 4.3）：紀錄 {st['entries']}；現在符合條件的 {st['eligible'] or '沒有'}；最終選擇"
             f"{'是 25' if st['final_is_25'] else '不是 25'}"]
    for i, why in st["not_eligible"].items():
        text = "；".join(why)
        lines.append(f"  #{i} 不符合：{text[:600]}{'…' if len(text) > 600 else ''}")
    if st["violation"]:
        lines.append("  違反 PREREG_25 4.3：有符合條件的 25，最終選擇卻不是它（沒有引用修訂，或引用的修訂沒有涵蓋它）；照 "
                     "docs/TESTDAY_25.md 第 5 步重跑 final")
    if st["redo_19"]:
        lines.append("  照 PREREG_25 4.3：現在沒有符合的 25，final 改照 PREREG_19 4.3（19，否則 09，否則 08，否則 07，否則 "
                     "04，否則 R20），再 prepare 並重傳那一份（docs/TESTDAY_25.md 第 5 步）")
    return lines


def t25_public_lines(log, e):
    """PREREG_25 4.3: 25, 19, 09, 08, 07, 04 and R20 share submission_rank.csv. Two equal public scores (|d| < 1e-9)
    between a 25 entry and another entry while the 25 report counts public-question rule-B differences > 0 between
    them: a hint that the wrong file was uploaded. Empty when the log has no 25 entry."""
    if e["comp"] != "rank" or e.get("public_score") is None or not t25_entries(log):
        return []
    lines = []
    others = [x for x in uploads_of(log, "rank") if x["id"] != e["id"] and x.get("public_score") is not None
              and abs(x["public_score"] - e["public_score"]) < 1e-9]
    for x in others:
        a, b = (e, x) if e.get("test25") else (x, e)
        if not a.get("test25") or b.get("test25"):
            continue
        if b.get("test14"):              # PREREG_14: a 14 entry is warned by t14_public_lines
            continue
        kind = "19" if b.get("test19") else "09" if b.get("test09") else "08" if b.get("test08") else \
            "07" if b.get("test07") else "04" if b.get("test04") else "R20"
        n = ((a.get("test25") or {}).get("public_rule_b_differences") or {}).get(kind)
        if _is_count(n) and n > 0:
            lines.append(f"WARNING (PREREG_25 4.3): #{a['id']}（25）和 #{b['id']}（{kind}）的 public 分數完全相同，但兩份在 "
                         f"public 題有 {n} 題的規則 B 等級不同：可能傳錯檔（七份同名）：核對 AIdea 的上傳紀錄")
    return lines


# ---------------------------------------------------------------- PREREG_21: test21 (competition 1, exact-copy layer)
def t21_dir():
    return os.path.join(REPO, *T21_DIR)


def is_t21_file(comp, path):
    """A competition-1 file in results/test21/player/ is a 21 file (only tools/t21_copy.py apply writes there)."""
    if comp != "player":
        return False
    here = os.path.realpath(os.path.dirname(os.path.abspath(path)))
    return os.path.normcase(here) == os.path.normcase(os.path.realpath(t21_dir()))


def t21_entries(log):
    return [x for x in log["uploads"] if x.get("comp") == "player" and x.get("test21")]


def t21_hide(log):
    """The log as the PREREG_06 / PREREG_10 rules see it when they judge a 21 entry's base (PREREG_21 5, 7): the same
    object without 21 entries; else a shallow copy whose uploads leave the 21 entries out (final_choice is shared)."""
    if not t21_entries(log):
        return log
    return dict(log, uploads=[x for x in log["uploads"] if not (x.get("comp") == "player" and x.get("test21"))])


def t21_player_rows(path):
    """({question_id: [top_1..top_5]}, [question_id in file order]) of a player file."""
    csv.field_size_limit(10**8)
    with open(refuse_forbidden(path), newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f)][1:]
    return {r[0]: r[1:] for r in rows if len(r) == 6}, [r[0] for r in rows if r]


def t21_report_problems(rep, rel, path, sha):
    """PREREG_21 5 / 7, re-read from the committed 21 report: real mode, verdict used, check_submission official
    without errors, this file's SHA-256 (sha None: not compared), the four competition-1 test files the report read
    equal to data/test, T1 PASS / NORMALIZED, a base in one of the competition-1 result folders whose bytes are the
    recorded ones, at least one changed row, every override row = the candidate then the base row without it, and the
    file differs from the base exactly in the rows the report lists as changed."""
    probs = []
    if rep.get("format") != T21_FORMAT:
        probs.append(f"{rel}: format {rep.get('format')!r}")
    if rep.get("mode") != "real":
        probs.append(f"{rel}: mode {rep.get('mode')!r} (uploads come from real-mode runs only)")
    if rep.get("verdict") != "used":
        probs.append(f"{rel}: verdict {rep.get('verdict')!r} (not 'used')")
    cs = rep.get("check_submission") or {}
    if cs.get("mode") != "official" or cs.get("errors"):
        probs.append(f"{rel}: check_submission {cs.get('mode')!r} with errors {cs.get('errors')}")
    want = (rep.get("outputs_sha256") or {}).get(FILE_NAMES["player"])
    if sha is not None and want != sha:
        probs.append(f"{rel}: records another {FILE_NAMES['player']} than this file ({sha[:12]})"
                     + (lf_hint(path, want, repo_rel(path)) if path else ""))
    read = {n: (((rep.get("files") or {}).get(n)) or {}).get("sha256") for n in QUESTION_FILES["player"]}
    for pool, n in zip(("public", "private"), CANDIDATE_FILES):
        p = ((rep.get("pools") or {}).get(pool)) or {}
        read[n] = p.get("sha256") if p.get("file") == n else None
    for n, want_t in read.items():
        tp = os.path.join(REPO, "data", "test", n)
        if not os.path.isfile(tp) or sha256_file(tp) != want_t:
            probs.append(f"{rel}: data/test/{n} is not the file the 21 report read ({str(want_t)[:12]})")
    t1v = (((rep.get("preflight") or {}).get("t1")) or {}).get("verdict")
    if t1v not in T1_UPLOAD_OK:
        probs.append(f"{rel}: T1 {t1v!r} at apply time")
    base = rep.get("base") or {}
    bf = base.get("file") or ""
    overrides = rep.get("overrides") or []
    changed = {o.get("question_id"): o.get("new_top5") for o in overrides if o.get("changed")}
    if not changed or rep.get("rows_changed") != len(changed):
        probs.append(f"{rel}: rows_changed {rep.get('rows_changed')!r} and {len(changed)} changed overrides "
                     f"(a 21 file changes at least one row)")
    bad_rule = [o.get("question_id") for o in overrides
                if not isinstance(o.get("base_top5"), list) or o.get("new_top5") != (
                    [o.get("candidate")] + [c for c in o.get("base_top5") or [] if c != o.get("candidate")][:4])
                or bool(o.get("changed")) != (o.get("new_top5") != o.get("base_top5"))]
    if bad_rule:
        probs.append(f"{rel}: overrides {bad_rule[:5]} are not 'the candidate first, then the base row' (PREREG_21 4)")
    if os.path.dirname(bf) not in T21_BASE_DIRS:
        probs.append(f"{rel}: base file {bf!r} is not in {T21_BASE_DIRS}")
        return probs
    bp = repo_path(bf)
    if not os.path.isfile(bp) or sha256_file(bp) != base.get("sha256"):
        probs.append(f"{bf} is missing or no longer the base bytes the 21 report records "
                     f"({str(base.get('sha256'))[:12]})")
    elif path is not None and os.path.isfile(path):
        b, border = t21_player_rows(bp)
        n, norder = t21_player_rows(path)
        if border != norder:
            probs.append(f"{repo_rel(path)}: the question order differs from its base {bf}")
        diff = {q: n.get(q) for q in border if n.get(q) != b.get(q)}
        bad_base = [o.get("question_id") for o in overrides if b.get(o.get("question_id")) != o.get("base_top5")]
        if bad_base:
            probs.append(f"{rel}: base_top5 of {bad_base[:5]} is not the base file's row")
        if diff != changed:
            extra = sorted(set(diff) ^ set(changed))[:5]
            probs.append(f"{repo_rel(path)}: differs from its base in other rows than the report lists "
                         f"({len(diff)} vs {len(changed)}; e.g. {extra})")
    return probs


def check_reports_t21(comp, path, sha):
    """PREREG_21 7: bind a 21 file to its committed report, the raw T1, the simulation record, the registration
    (K5-21: the rev 2 row on main's first-parent history) and K6-21."""
    d = os.path.dirname(os.path.abspath(path))
    rpath = os.path.join(d, T21_REPORT)
    rel_, rep = _committed_json(rpath, "the 21 report (tools/t21_copy.py apply)")
    out = {rel_: sha256_file(rpath)}
    probs = t21_report_problems(rep, rel_, path, sha)
    t1_path = os.path.join(REPO, "results", "test02", "checks", comp, "t1.json")
    trel, t1 = _committed_json(t1_path, "the T1 result")
    out[trel] = sha256_file(t1_path)
    probs += t1_problems(t1, trel)
    v = t1.get("verdict")
    if v not in T1_UPLOAD_OK:
        probs.append(f"{trel}: verdict {v!r} (PREREG_21 9: the layer is not used)")
    srel, sim = _committed_json(repo_path(T21_SIM), "the simulation record")
    out[srel] = sha256_file(repo_path(T21_SIM))
    if not (sim.get("evaluation") or {}).get("passed"):
        probs.append(f"{srel}: the simulation did not pass (PREREG_21 6)")
    k5 = (((rep.get("preflight") or {}).get("k5_and_sim")) or {})
    if k5.get("k5_ok") is not True:
        probs.append(f"{rel_}: K5-21 was not confirmed at apply time (PREREG_21 8)")
    h2, t2 = git_row_added_on_main(P21_DOC, "| rev 2 |")
    if not head_has_p21_revision_row(2):
        probs.append(f"{P21_DOC} at HEAD has no '| rev 2 |' row (PREREG_21 8)")
    if h2 is None or parse_time(t2, "commit time") > K5_21:
        probs.append(f"K5-21: the '| rev 2 |' row of {P21_DOC} reached main at {t2} ({h2}); the deadline is "
                     f"{K5_21.isoformat()} (PREREG_21 8)")
    st = git_file_state(P21_DOC)
    if not st["tracked"] or not st["clean"]:
        probs.append(f"{P21_DOC} is not committed and clean")
    h, t = git_commit_time(rel_)
    k6_ok = h is not None and parse_time(t, "commit time") <= K6_21
    if not k6_ok:
        probs.append(f"K6-21: {rel_} last committed {t} ({h}); the deadline is {K6_21.isoformat()} (PREREG_21 8)")
    if probs:
        raise LogError("the 21 file is not backed by its report (PREREG_21 7):\n - " + "\n - ".join(probs))
    base = rep["base"]
    t21 = {"report": rel_, "report_sha256": out[rel_], "base_entry_id": base.get("entry_id"),
           "base_file": base.get("file"), "base_sha256": base.get("sha256"),
           "questions_overridden": rep.get("questions_overridden"), "rows_changed": rep.get("rows_changed"),
           "changed_by_pool": rep.get("changed_by_pool") or {},
           "k6_21": {"commit": h, "committed_at": t, "deadline": K6_21.isoformat(), "ok": True}}
    return {"reports": out, "t1_verdict": v, "fallback_used": None, "test21": t21}


def method_for_t21(given):
    """A 21 file is recorded as the PREREG_21 layer, never with its base's label."""
    if given and "prereg21" not in given:
        raise LogError(f"--method {given!r} does not name prereg21; a results/test21/player file is the exact-copy "
                       f"layer (default {T21_METHOD!r})")
    return given or T21_METHOD


def t21_base_reasons(fin):
    """The --not-06-reason / --not-10-reason recorded with a competition-1 final choice (None where none was given)."""
    fin = fin or {}
    return {"not_06_reason": (fin.get("prereg06_5_4") or {}).get("not_06_reason"),
            "not_10_reason": (fin.get("prereg10_4_4") or {}).get("not_10_reason")}


def t21_moot_reasons(log, reasons):
    """With 10 entries in the log PREREG_10 4.4 replaces PREREG_06 5.4 (final_rule_t06 refuses a --not-06-reason then):
    a --not-06-reason recorded before the 10 entries is dropped (review 10/20, MINOR 8)."""
    reasons = dict(reasons or {})
    if reasons.get("not_06_reason") and t10_entries(log):
        reasons["not_06_reason"] = None
    return reasons


def t21_base_rules(log, base, reasons):
    """The competition-1 final rules (PREREG_06 5.4, PREREG_10 4.4) applied to the base entry as final_rule applies
    them, on the log without 21 entries (PREREG_21 5); raises LogError when they no longer designate it."""
    view = t21_hide(log)
    out = {}
    reasons = t21_moot_reasons(view, reasons)
    r6 = final_rule_t06(view, base, (reasons or {}).get("not_06_reason"))
    if r6:
        out["prereg06_5_4"] = r6.get("chosen")
    r10 = final_rule_t10(view, base, (reasons or {}).get("not_10_reason"))
    if r10:
        out["prereg10_4_4"] = r10.get("chosen")
    return out


def t21_base_entry(log, e21):
    """The uploaded, non-21 base entry a 21 entry was built on (same file and bytes)."""
    t21 = e21.get("test21") or {}
    base = next((x for x in log["uploads"] if x["id"] == t21.get("base_entry_id")), None)
    if base is None or base.get("comp") != "player" or base.get("status") != "uploaded" or base.get("test21"):
        raise LogError(f"PREREG_21 7: the base #{t21.get('base_entry_id')} of 21 entry #{e21['id']} is not an uploaded "
                       f"competition-1 entry")
    if base.get("sha256") != t21.get("base_sha256") or base.get("file") != t21.get("base_file"):
        raise LogError(f"PREREG_21 7: entry #{base['id']} is not the base the 21 report records "
                       f"({t21.get('base_file')}, {str(t21.get('base_sha256'))[:12]})")
    return base


def prepare_order_t21(log, t21):
    """PREREG_21 7: the report's base is final_choice.player, an uploaded non-21 entry with the same bytes, and the
    competition-1 final rules (with the reasons recorded in that final choice, 21 entries hidden) still designate it."""
    fin = log["final_choice"].get("player")
    if not fin or fin.get("id") != t21.get("base_entry_id") or fin.get("sha256") != t21.get("base_sha256"):
        raise LogError(f"PREREG_21 7: the 21 report's base #{t21.get('base_entry_id')} is not final_choice.player "
                       f"(#{(fin or {}).get('id')}): set the final choice by the current rules first "
                       f"(docs/TESTDAY_10.md step 6 / docs/TESTDAY_06.md step 9), then rerun tools/t21_copy.py apply "
                       f"(docs/TESTDAY_21.md)")
    base = t21_base_entry(log, {"id": "(new)", "test21": t21})
    reasons = t21_moot_reasons(t21_hide(log), t21_base_reasons(fin))
    try:
        rules = t21_base_rules(log, base, reasons)
    except LogError as e:
        raise LogError(f"PREREG_21 7: the base #{base['id']} is no longer the file the competition-1 rules designate: "
                       f"{e}")
    t21["base_rule_reasons"] = reasons
    t21["base_rules"] = rules
    t21["base_method"] = base.get("method")


def print_t21_ok_lines(t21):
    ch = t21.get("changed_by_pool") or {}
    print(f"- 這一份是 21（PREREG_21 完全複製層），底檔是 #{t21.get('base_entry_id')}（{t21.get('base_file')}，"
          f"{t21.get('base_method')}）；覆寫 {t21.get('questions_overridden')} 題，其中 {t21.get('rows_changed')} 題的 "
          f"Top-5 和底檔不同（public {ch.get('public', 0)}、private {ch.get('private', 0)}）")
    print("- 21 之外的列和底檔逐位元組相同；21 是競賽一最後一次上傳；上傳後要 final --comp player --id <這一筆>"
          "（docs/TESTDAY_21.md）")
    print(f"- K6-21：21 的報告在 {t21['k6_21']['committed_at']} commit（期限 11/23 18:00）")


def t21_entry_eligible(x, log, require_uploaded=True, with_base_rules=True):
    """PREREG_21 7 for one logged 21 entry, re-checked now: (uploaded,) the report and the file unchanged since prepare
    and still meeting t21_report_problems, the report's last commit the recorded one and by K6-21, the base entry
    uploaded with the recorded bytes, and (with_base_rules) the final rules still designating the base (21 hidden) and
    the final choice on the base or a 21 on it. -> (ok, why)."""
    t21, why = x.get("test21") or {}, []
    if require_uploaded and x.get("status") != "uploaded":
        why.append(f"#{x['id']} is {x.get('status')!r}, not uploaded")
    rel_ = t21.get("report") or ""
    rp = repo_path(rel_) if rel_ else ""
    rep = None
    if not rel_ or not os.path.isfile(rp) or sha256_file(rp) != t21.get("report_sha256"):
        why.append(f"{rel_ or 'the 21 report'} changed since prepare (rerun tools/t21_copy.py apply, docs/TESTDAY_21.md)")
    else:
        with open(refuse_forbidden(rp), encoding="utf-8") as f:
            rep = json.load(f)
    fp = repo_path(x.get("file") or "") if x.get("file") else ""
    if not fp or not os.path.isfile(fp) or sha256_file(fp) != x.get("sha256"):
        why.append(f"{x.get('file')} changed since prepare")
    k6 = t21.get("k6_21") or {}
    h, t = git_commit_time(rel_) if rel_ else (None, None)
    if not k6.get("ok") or h is None or h != k6.get("commit") or parse_time(t, "commit time") > K6_21:
        why.append(f"K6-21: the report's last commit is {h} at {t}; prepare recorded {k6.get('commit')} at "
                   f"{k6.get('committed_at')} (deadline {K6_21.isoformat()})")
    if rep is not None:
        why += t21_report_problems(rep, rel_, fp if fp and os.path.isfile(fp) else None, x.get("sha256"))
    try:
        base = t21_base_entry(log, x)
    except LogError as e:
        why.append(str(e))
        base = None
    if base is not None and with_base_rules:
        try:
            t21_base_rules(log, base, t21.get("base_rule_reasons") or {})
        except LogError as e:
            why.append(f"the base #{base['id']} is no longer the file the competition-1 rules designate: {e}")
        fin = log["final_choice"].get("player")
        fin_e = next((y for y in log["uploads"] if y["id"] == (fin or {}).get("id")), None)
        on_base = fin_e is not None and (fin_e["id"] == base["id"] or (fin_e.get("test21") or {}).get(
            "base_entry_id") == base["id"])
        if fin and not on_base:
            why.append(f"the final choice moved to #{fin.get('id')} (not the base #{base['id']} or a 21 on it): the "
                       f"competition-1 rules designate another file now (PREREG_21 7)")
    return not why, why


def t21_eligibility(log):
    return {x["id"]: t21_entry_eligible(x, log) for x in t21_entries(log)}


def head_has_p21_revision_row(n, head=None):
    head = git_head_text(P21_DOC) if head is None else head
    return bool(head) and any(line.startswith(f"| rev {n} |") for line in head.splitlines())


def p21_revision(reason, overridden, what):
    """Setting aside a 21 upload that meets PREREG_21 7: a reason naming 'PREREG_21 rev N' (N >= 3) whose row is at
    HEAD of the committed, clean docs/PREREG_21_PLAYER_EXACT_COPY.md and was added after every overridden record."""
    text = (reason or "").strip()
    m = re.search(r"PREREG_21\s+rev\s+(\d+)", text)
    if not m:
        raise LogError(f"{what}: needs --not-21-reason \"PREREG_21 rev N: <the program error>\" citing a committed "
                       f"revision (N >= {P21_MIN_REV}); a reason without a revision is refused")
    n = int(m.group(1))
    if n < P21_MIN_REV:
        raise LogError(f"{what}: --not-21-reason cites PREREG_21 rev {n}; an override needs rev {P21_MIN_REV} or later")
    st = git_file_state(P21_DOC)
    if not st["tracked"] or not st["clean"]:
        raise LogError(f"{what}: {P21_DOC} must be committed and unchanged (tracked={st['tracked']}, "
                       f"clean={st['clean']})")
    row = f"| rev {n} |"
    if not head_has_p21_revision_row(n):
        raise LogError(f"{what}: {P21_DOC} at HEAD has no row starting {row!r}; commit the revision first")
    h, t = git_row_added(P21_DOC, row)
    if h is None:
        raise LogError(f"{what}: no commit found that added {row!r} to {P21_DOC}")
    at = parse_time(t, "commit time")
    early = [w for w in overridden if at <= parse_time(w, "time")]
    if early:
        raise LogError(f"{what}: PREREG_21 rev {n} was committed at {t} ({h[:12]}), not after the records it "
                       f"overrides ({', '.join(early)}); a revision must come after the decision it overturns")
    return {"rev": n, "row": row, "doc": P21_DOC, "commit": h, "committed_at": t, "reason": text,
            "overridden_at": list(overridden)}


def _t21_other_rule(log, reason, what):
    """While a logged 21 upload meets PREREG_21 7, any other competition-1 file or final choice (its base included)
    needs --not-21-reason citing a later revision. None when the log has no 21 entry."""
    elig = t21_eligibility(log)
    if not elig:
        if reason:
            raise LogError("--not-21-reason: the log has no 21 entry to set aside")
        return None
    ok_ids = [i for i, (ok, _w) in elig.items() if ok]
    rec = {"rule": "PREREG_21 7", "eligible_21_entries": ok_ids,
           "not_eligible_21_entries": {str(i): w for i, (ok, w) in elig.items() if not ok}, "checked_at": now_iso(),
           "not_21_reason": reason or None}
    if ok_ids:
        when = [x["prepared_at"] for x in t21_entries(log) if x["id"] in ok_ids]
        rec["revision"] = p21_revision(reason, when, f"21 uploads {ok_ids} meet PREREG_21 7, so {what}")
    return rec


def prepare_rule_t21(log, comp, bound, not_21_reason):
    """PREREG_21 7 at prepare: a competition-1 file that is not 21 while a logged 21 upload meets PREREG_21 7 needs the
    revision check of --not-21-reason. None when the log has no 21 entry (exactly as before)."""
    reason = (not_21_reason or "").strip()
    if comp != "player" or bound.get("test21"):
        if reason:
            raise LogError("--not-21-reason applies to a competition-1 file that is not 21")
        return None
    return _t21_other_rule(log, reason, "another competition-1 file is not uploaded after the 21")


def final_rule_t21_self(log, e21, not_21_reason):
    """PREREG_21 7 for a 21 final (the 06 / 10 rules were applied to its base by cmd_final): the 21 entry itself."""
    if (not_21_reason or "").strip():
        raise LogError("--not-21-reason does not apply when the 21 entry is chosen")
    ok, why = t21_entry_eligible(e21, log, require_uploaded=False, with_base_rules=False)
    if not ok:
        raise LogError(f"entry #{e21['id']} (21) does not meet PREREG_21 7 now ({'; '.join(why)[:900]}): the final "
                       f"choice is the file the competition-1 rules designate (docs/TESTDAY_21.md)")
    t21 = e21["test21"]
    return {"rule": "PREREG_21 7", "chosen": "21", "base_entry_id": t21.get("base_entry_id"),
            "eligible_21_entries": [i for i, (k, _w) in t21_eligibility(log).items() if k], "checked_at": now_iso(),
            "not_21_reason": None}


def final_rule_t21_other(log, e, not_21_reason):
    """PREREG_21 7 for a competition-1 final that is not 21: None without 21 entries; while a 21 upload meets the
    conditions, only with --not-21-reason (p21_revision)."""
    reason = (not_21_reason or "").strip()
    if e["comp"] != "player":
        if reason:
            raise LogError("--not-21-reason applies to competition 1 (player) only")
        return None
    rec = _t21_other_rule(log, reason, "the final choice is 21")
    if rec is not None:
        rec["chosen"] = "not 21"
    return rec


def t21_status_view(log, comp, fin, last):
    """(log, fin, last) as the 06 / 10 status blocks should see them (PREREG_21 5, 7): the log without 21 entries,
    the base as the final choice and the base as the last upload when the last upload is a 21 on it -- when the final
    choice is a 21 entry, or is the base of an uploaded 21 entry that meets PREREG_21 7 (the step before final --id
    <21>). None otherwise (the blocks see the real values, exactly as before; e.g. a 21 set aside on a 06-ON base:
    06-ON is no longer the last upload, final 06-OFF)."""
    if comp != "player" or not fin or not t21_entries(log):
        return None
    e = next((x for x in log["uploads"] if x["id"] == fin.get("id")), None)
    if not e:
        return None
    if e.get("test21"):
        base = next((x for x in log["uploads"] if x["id"] == (e["test21"] or {}).get("base_entry_id")), None)
    else:
        on_it = [x for x in t21_entries(log) if x.get("status") == "uploaded"
                 and (x["test21"] or {}).get("base_entry_id") == e["id"]]
        base = e if any(t21_entry_eligible(x, log)[0] for x in on_it) else None
    if base is None:
        return None
    view = t21_hide(log)
    fin_v = dict(fin, id=base["id"], sha256=base["sha256"], file=base["file"], method=base.get("method"))
    if last and last.get("test21"):
        if (last["test21"] or {}).get("base_entry_id") == base["id"]:
            last = base
        else:
            ups = uploads_of(view, comp)
            last = ups[-1] if ups else None
    return view, fin_v, last


def t21_status(log, fin, last):
    elig = t21_eligibility(log)
    ok_ids = [i for i, (ok, _w) in elig.items() if ok]
    by_id = {x["id"]: x for x in log["uploads"]}
    fin_e = by_id.get((fin or {}).get("id"))
    final_is_21 = bool(fin_e and fin_e.get("test21"))
    rule = (fin or {}).get("prereg21_7") or {}
    rev = rule.get("revision") or {}

    def overridden_by_rev(i):
        prep = (by_id.get(i) or {}).get("prepared_at")
        if i not in (rule.get("eligible_21_entries") or []) or not rev.get("committed_at") or not prep:
            return False
        try:
            return parse_time(rev["committed_at"], "revision commit time") > parse_time(prep, "prepared_at")
        except LogError:
            return False
    final_ok = (final_is_21 and fin_e["id"] in ok_ids and rule.get("chosen") == "21") or \
        (bool(fin) and not final_is_21 and bool(rev) and all(overridden_by_rev(i) for i in ok_ids))
    final_bad = bool(final_is_21 and not elig.get(fin_e["id"], (False, []))[0])
    return {"entries": sorted(elig), "eligible": ok_ids,
            "not_eligible": {str(i): w for i, (ok, w) in elig.items() if not ok},
            "final_set": bool(fin), "final_is_21": final_is_21,
            "last_is_21": bool(last and last.get("test21")),
            "violation": bool(ok_ids) and bool(fin) and not final_ok, "redo_base": final_bad}


def t21_status_lines(st):
    lines = [f"  21（PREREG_21 7）：紀錄 {st['entries']}；現在符合條件的 {st['eligible'] or '沒有'}；最終選擇"
             f"{'是 21' if st['final_is_21'] else '不是 21'}"]
    for i, why in st["not_eligible"].items():
        text = "；".join(why)
        lines.append(f"  #{i} 不符合：{text[:600]}{'…' if len(text) > 600 else ''}")
    if st["eligible"] and not st["final_is_21"]:
        if st["violation"]:
            lines.append("  違反 PREREG_21 7：有符合條件的 21，最終選擇卻不是它（沒有引用修訂）；照 docs/TESTDAY_21.md 重跑 final")
        elif not st["final_set"]:
            lines.append("  有符合 PREREG_21 7 的 21：最終選擇要設成 21（docs/TESTDAY_21.md）")
    if st["redo_base"]:
        lines.append("  照 PREREG_21 7：21 不再符合；final 改回現行規則指定的那一份（底檔是 06-ON 時是 06-OFF），再 prepare 並"
                     "重傳它（docs/TESTDAY_21.md）")
    return lines


def t21_public_lines(log, e):
    """PREREG_21: 21 and its base share submission_player.csv. Equal public scores while the 21 report changed public
    rows, or different public scores while it changed none: a hint that the wrong file was uploaded. Empty when the
    log has no 21 entry."""
    if e["comp"] != "player" or e.get("public_score") is None or not t21_entries(log):
        return []
    lines = []
    for x in uploads_of(log, "player"):
        if x["id"] == e["id"] or x.get("public_score") is None:
            continue
        a, b = (e, x) if e.get("test21") else (x, e)
        if not a.get("test21") or b.get("test21") or (a["test21"] or {}).get("base_entry_id") != b["id"]:
            continue
        n_pub = ((a["test21"] or {}).get("changed_by_pool") or {}).get("public", 0)
        same = abs(a["public_score"] - b["public_score"]) < 1e-9
        if same and _is_count(n_pub) and n_pub > 0:
            lines.append(f"WARNING (PREREG_21): #{a['id']}（21）和底檔 #{b['id']} 的 public 分數完全相同，但 21 改了 public "
                         f"{n_pub} 題：可能傳錯檔（同名）：核對 AIdea 的上傳紀錄")
        if not same and n_pub == 0:
            lines.append(f"WARNING (PREREG_21): #{a['id']}（21）沒有改任何 public 題，public 分數卻和底檔 #{b['id']} 不同："
                         f"可能傳錯檔：核對 AIdea 的上傳紀錄")
    return lines


# ---------------------------------------------------------------- commands
def check_own_tools():
    """prepare runs check_submission --official and prints that it passed: both files must be the committed ones,
    as t02_rank / t02_player require for check_submission (review finding 2026-09-26)."""
    for rel in OWN_TOOLS:
        st = git_file_state(rel)
        if not st["tracked"] or not st["clean"]:
            raise LogError(f"{rel} must be committed and unchanged before an upload (tracked={st['tracked']}, "
                           f"clean={st['clean']})")


def cmd_prepare(args):
    comp, path = args.comp, args.file
    if os.path.basename(path) != FILE_NAMES[comp]:
        raise LogError(f"--comp {comp} uploads {FILE_NAMES[comp]}; got {os.path.basename(path)}")
    if not os.path.exists(path):
        raise LogError(f"{path} does not exist")
    if not (args.reason or "").strip():
        raise LogError("--reason TEXT is required (why this upload: first upload, bug fix, final choice, ...)")
    rel = repo_rel(path)
    st = git_file_state(rel)
    if not st["tracked"] or not st["clean"]:
        raise LogError(f"{rel} must be committed and unchanged before an upload (tracked={st['tracked']}, "
                       f"clean={st['clean']})")
    check_own_tools()
    tests_dir = args.tests_dir or default_tests()
    check_tests_dir(tests_dir, comp)
    qs = [os.path.join(tests_dir, n) for n in QUESTION_FILES[comp]]
    cs = [os.path.join(tests_dir, n) for n in CANDIDATE_FILES] if comp == "player" else []
    errors = check_submission.check(comp, path, qs, cs, official=True)
    if errors:
        raise LogError("check_submission --official failed:\n - " + "\n - ".join(errors[:20]))
    sha = sha256_file(path)
    bound = check_reports(comp, path, sha)
    method = method_for(comp, bound["fallback_used"], args.method)
    if bound.get("test04"):              # PREREG_04 rev 3: a 04 file never carries R20's label
        method = method_for_t04(args.method)
    if bound.get("test06"):              # PREREG_06 rev 3: a 06 file carries its own method (06-OFF / 06-ON)
        method = method_for_t06(bound["test06"]["variant"], args.method)
    if bound.get("test07"):              # PREREG_07 rev 4: a 07 file is Q02H, never R20's label
        method = method_for_t07(args.method)
    if bound.get("test08"):              # PREREG_08 rev 3: an 08 file is Q08, never 07's, 04's or R20's label
        method = method_for_t08(args.method)
    if bound.get("test09"):              # PREREG_09 rev 3: a 09 file is Q09, never 08's, 07's, 04's or R20's label
        method = method_for_t09(args.method)
    if bound.get("test10"):              # PREREG_10 rev 3: a 10 file carries its own method, never Q04's
        method = method_for_t10(args.method)
    if bound.get("test14"):              # PREREG_14: a 14 file carries its own method, never its base's
        method = method_for_t14(args.method)
    if bound.get("test19"):              # PREREG_19 rev 3: a 19 file is Q19, never 09's, 08's, 07's, 04's or R20's label
        method = method_for_t19(args.method)
    if bound.get("test25"):              # PREREG_25 rev 3: a 25 file is Q25, never 19's, 09's ... or R20's label
        method = method_for_t25(args.method)
    if bound.get("test21"):              # PREREG_21: a 21 file carries its own method, never its base's
        method = method_for_t21(args.method)
    later = check_t2_t5(comp, args.checks_missing_reason)
    log_path = args.log or default_log()
    log = load_log(log_path)
    if bound.get("test04"):              # PREREG_04 5.3: R20's first upload is logged before any 04 upload
        r20 = r20_first_upload(log)
        if r20 is None:
            raise LogError(f"PREREG_04 5.3: log R20's first upload ({T04_R20_FILE}, status uploaded) before "
                           f"preparing a 04 upload (docs/TESTDAY.md step 8, then docs/TESTDAY_04.md)")
        bound["test04"]["r20_first_upload_id"] = r20["id"]
    if bound.get("test06"):              # PREREG_06 5.3 / 5.5 E1: Q04 first; 06-ON after its run's 06-OFF, once
        prepare_order_t06(log, bound["test06"], sha)
    if bound.get("test07"):              # PREREG_07 5.4: R20 and the bound 04 uploaded first; chunks after R20
        prepare_order_t07(log, bound["test07"])
    if bound.get("test08"):              # PREREG_08 5.4: R20, 04 and the bound 07 uploaded first; chunks after R20
        prepare_order_t08(log, bound["test08"])
    if bound.get("test09"):              # PREREG_09 5.3: R20, 04, 07 and the bound 08 uploaded first; chunks after R20
        prepare_order_t09(log, bound["test09"])
    if bound.get("test19"):              # PREREG_19 4.3: R20 and the bound 09 uploaded first; chunks after R20
        prepare_order_t19(log, bound["test19"])
    if bound.get("test25"):              # PREREG_25 4.3: R20 and the bound 19 uploaded first; chunks after R20
        prepare_order_t25(log, bound["test25"])
    if bound.get("test10"):              # PREREG_10 4.4: Q04 first; this season's 06-OFF uploaded; 06-ON not kept
        prepare_order_t10(log, bound["test10"])
    if bound.get("test14"):              # PREREG_14 7: the base is final_choice.rank and still designated
        prepare_order_t14(log, bound["test14"])
    if bound.get("test21"):              # PREREG_21 7: the base is final_choice.player and still designated
        prepare_order_t21(log, bound["test21"])
    not04 = prepare_rule_t04(log, comp, bound, getattr(args, "not_04_reason", None))   # None: no 04 entry logged
    not06 = prepare_rule_t06(log, comp, bound, getattr(args, "not_06_reason", None))   # None: no 06 entry logged
    not07 = prepare_rule_t07(log, comp, bound, getattr(args, "not_07_reason", None))   # None: no 07 entry logged
    not08 = prepare_rule_t08(log, comp, bound, getattr(args, "not_08_reason", None))   # None: no 08 entry logged
    not09 = prepare_rule_t09(log, comp, bound, getattr(args, "not_09_reason", None))   # None: no 09 entry logged
    not19 = prepare_rule_t19(log, comp, bound, getattr(args, "not_19_reason", None))   # None: no 19 entry logged
    not25 = prepare_rule_t25(log, comp, bound, getattr(args, "not_25_reason", None))   # None: no 25 entry logged
    not10 = prepare_rule_t10(log, comp, bound, getattr(args, "not_10_reason", None))   # None: no 10 entry logged
    bound["_sha"] = sha                  # PREREG_14 7: prepare_rule_t14 asks whether this file is a 14's base
    not14 = prepare_rule_t14(log, comp, bound, getattr(args, "not_14_reason", None))   # None: no 14 entry logged
    bound.pop("_sha", None)
    not21 = prepare_rule_t21(log, comp, bound, getattr(args, "not_21_reason", None))   # None: no 21 entry logged
    rows = count_rows(path)
    eid = max([e["id"] for e in log["uploads"]], default=0) + 1
    eid = max([eid] + [v["id"] + 1 for v in log.get("voided_uploads") or []])   # PREREG_27 1: numbering continues
    e = {"id": eid, "comp": comp, "team": TEAMS[comp], "status": "prepared", "file": rel, "sha256": sha,
         "rows": rows, "git_commit": st["head"], "file_commit": st["file_commit"],
         "method": method, "reason": args.reason,
         "check_submission": "OK (official)", "t1_verdict": bound["t1_verdict"],
         "fallback_used": bound["fallback_used"], "reports_sha256": dict(bound["reports"], **later["reports"]),
         "t2_t5": {k: later[k] for k in ("recorded", "missing", "missing_reason")},
         "prepared_at": now_iso(), "user_ok_at": None, "uploaded_at": None,
         "public_score": None, "public_delta_vs_prev": None, "lb_rank": None, "investigate": False, "note": ""}
    if bound.get("test04"):
        e["test04"] = bound["test04"]
    if bound.get("test06"):
        e["test06"] = bound["test06"]
    if bound.get("test07"):
        e["test07"] = bound["test07"]
    if bound.get("test08"):
        e["test08"] = bound["test08"]
    if bound.get("test09"):
        e["test09"] = bound["test09"]
    if bound.get("test19"):
        e["test19"] = bound["test19"]
    if bound.get("test25"):
        e["test25"] = bound["test25"]
    if bound.get("test10"):
        e["test10"] = bound["test10"]
    if bound.get("test14"):
        e["test14"] = bound["test14"]
    if bound.get("test21"):
        e["test21"] = bound["test21"]
    if not04:
        e["prereg04_5_3"] = not04
    if not06:
        e["prereg06_5_4"] = not06
    if not07:
        e["prereg07_5_4"] = not07
    if not08:
        e["prereg08_5_4"] = not08
    if not09:
        e["prereg09_5_3"] = not09
    if not19:
        e["prereg19_4_3"] = not19
    if not25:
        e["prereg25_4_3"] = not25
    if not10:
        e["prereg10_4_4"] = not10
    if not14:
        e["prereg14_7"] = not14
    if not21:
        e["prereg21_7"] = not21
    same = [x["id"] for x in log["uploads"] if x["comp"] == comp and x["sha256"] == sha]
    log["uploads"].append(e)
    save_log(log_path, log)
    print(f"prepared entry #{eid} in {log_path}" + (f" (same file as entries {same})" if same else ""))
    print()
    print("請確認這一份檔案（只適用這一次上傳）：")
    print(f"- 比賽：{COMP_LABEL[comp]} {TEAMS[comp]}")
    print(f"- 檔案：{os.path.abspath(path)}")
    print(f"- 列數：{rows}（不含標頭）")
    print(f"- SHA-256 前 12 碼：{sha[:12]}")
    print(f"- 方法：{e['method']}；原因：{args.reason}")
    print(f"- 檢查：T1 {bound['t1_verdict']}"
          + ("（競賽二用 R10 備案）" if bound["fallback_used"] else "")
          + "；notebook 與本機的接縫都通過；check_submission --official 通過")
    print("- T2–T5（只記錄，不擋上傳）：" + ("、".join(f"{c} {v}" for c, v in later["recorded"].items()) or "沒有")
          + (f"；缺 {later['missing']}（原因：{later['missing_reason']}）" if later["missing"] else ""))
    if bound.get("test04"):
        print_t04_ok_lines(bound["test04"])
    if bound.get("test06"):
        print_t06_ok_lines(bound["test06"])
    if bound.get("test07"):
        print_t07_ok_lines(bound["test07"])
    if bound.get("test08"):
        print_t08_ok_lines(bound["test08"])
    if bound.get("test09"):
        print_t09_ok_lines(bound["test09"])
    if bound.get("test19"):
        print_t19_ok_lines(bound["test19"])
    if bound.get("test25"):
        print_t25_ok_lines(bound["test25"])
    if bound.get("test10"):
        print_t10_ok_lines(bound["test10"])
    if bound.get("test14"):
        print_t14_ok_lines(bound["test14"])
    if bound.get("test21"):
        print_t21_ok_lines(bound["test21"])
    if not04 and not04.get("revision"):
        print(f"- 注意：04 紀錄 {not04['eligible_04_entries']} 符合 PREREG_04 5.3；這一份不是 04，依 PREREG_04 rev "
              f"{not04['revision']['rev']}（{not04['revision']['commit'][:12]}）改交：{not04['not_04_reason']}")
    if not06 and not06.get("revision"):
        print(f"- 注意：06 紀錄 {not06['eligible_06_entries']} 符合 PREREG_06 5.4；這一份是 Q04，依 PREREG_06 rev "
              f"{not06['revision']['rev']}（{not06['revision']['commit'][:12]}）改交：{not06['not_06_reason']}")
    if not07 and not07.get("revision"):
        print(f"- 注意：07 紀錄 {not07['eligible_07_entries']} 符合 PREREG_07 5.4；這一份不是 07，依 PREREG_07 rev "
              f"{not07['revision']['rev']}（{not07['revision']['commit'][:12]}）改交：{not07['not_07_reason']}")
    if not08 and not08.get("revision"):
        print(f"- 注意：08 紀錄 {not08['eligible_08_entries']} 符合 PREREG_08 5.4；這一份不是 08，依 PREREG_08 rev "
              f"{not08['revision']['rev']}（{not08['revision']['commit'][:12]}）改交：{not08['not_08_reason']}")
    if not09 and not09.get("revision"):
        print(f"- 注意：09 紀錄 {not09['eligible_09_entries']} 符合 PREREG_09 5.3；這一份不是 09，依 PREREG_09 rev "
              f"{not09['revision']['rev']}（{not09['revision']['commit'][:12]}）改交：{not09['not_09_reason']}")
    if not19 and not19.get("revision"):
        print(f"- 注意：19 紀錄 {not19['eligible_19_entries']} 符合 PREREG_19 4.3；這一份不是 19，依 PREREG_19 rev "
              f"{not19['revision']['rev']}（{not19['revision']['commit'][:12]}）改交：{not19['not_19_reason']}")
    if not25 and not25.get("revision"):
        print(f"- 注意：25 紀錄 {not25['eligible_25_entries']} 符合 PREREG_25 4.3；這一份不是 25，依 PREREG_25 rev "
              f"{not25['revision']['rev']}（{not25['revision']['commit'][:12]}）改交：{not25['not_25_reason']}")
    if not10 and not10.get("revision"):
        print(f"- 注意：10 紀錄 {not10['eligible_10_entries']} 符合 PREREG_10 4.4；這一份不是 10，依 PREREG_10 rev "
              f"{not10['revision']['rev']}（{not10['revision']['commit'][:12]}）改交：{not10['not_10_reason']}")
    if not14 and not14.get("revision"):
        print(f"- 注意：14 紀錄 {not14['eligible_14_entries']} 符合 PREREG_14 7；這一份不是 14，依 PREREG_14 rev "
              f"{not14['revision']['rev']}（{not14['revision']['commit'][:12]}）改交：{not14['not_14_reason']}")
    if not21 and not21.get("revision"):
        print(f"- 注意：21 紀錄 {not21['eligible_21_entries']} 符合 PREREG_21 7；這一份不是 21，依 PREREG_21 rev "
              f"{not21['revision']['rev']}（{not21['revision']['commit'][:12]}）改交：{not21['not_21_reason']}")
    print(f"- 登記編號：#{eid}")
    return e


def cmd_uploaded(args):
    log_path = args.log or default_log()
    log = load_log(log_path)
    e = entry(log, args.id)
    if e["status"] != "prepared":
        raise LogError(f"entry #{e['id']} is {e['status']!r}; only a prepared entry can be marked uploaded")
    ok, up = parse_time(args.ok_at, "--ok-at"), parse_time(args.uploaded_at, "--uploaded-at")
    prep = parse_time(e["prepared_at"], "prepared_at")
    if ok < prep:
        raise LogError(f"--ok-at {ok.isoformat()} is before the entry was prepared ({e['prepared_at']}); "
                       f"the OK must be for this prepared file")
    if up < ok:
        raise LogError(f"--uploaded-at {up.isoformat()} is before the OK ({ok.isoformat()})")
    path = os.path.join(REPO, *e["file"].split("/"))
    if not os.path.exists(path) or sha256_file(path) != e["sha256"]:
        raise LogError(f"{e['file']} is missing or no longer the prepared bytes ({e['sha256'][:12]}); restore the "
                       f"committed bytes with  git -c core.autocrlf=false checkout -- {e['file']}  (a plain checkout "
                       f"with core.autocrlf=true writes CRLF, i.e. other bytes), then log this upload")
    e.update(status="uploaded", user_ok_at=ok.isoformat(), uploaded_at=up.isoformat())
    day = local_date(e["uploaded_at"])
    n_day = sum(1 for x in uploads_of(log, e["comp"]) if local_date(x["uploaded_at"]) == day)
    save_log(log_path, log)
    print(f"entry #{e['id']} uploaded {e['uploaded_at']} ({COMP_LABEL[e['comp']]}, {day}: upload {n_day} of "
          f"at most {MAX_PER_DAY})")
    if n_day > MAX_PER_DAY:
        print(f"WARNING: {n_day} uploads on {day} for {e['comp']}; the platform allows {MAX_PER_DAY} per day")
    return e


def cmd_public(args):
    log_path = args.log or default_log()
    log = load_log(log_path)
    e = entry(log, args.id)
    if e["status"] != "uploaded":
        raise LogError(f"entry #{e['id']} is {e['status']!r}; log the upload first")
    if e["public_score"] is not None:
        raise LogError(f"entry #{e['id']} already has public score {e['public_score']}")
    score = float(args.score)
    if not 0.0 <= score <= 1.0:
        raise LogError(f"--score {score} is outside 0..1")
    t = parse_time(e["uploaded_at"], "uploaded_at")
    prev = [x for x in uploads_of(log, e["comp"])
            if x["id"] != e["id"] and x["public_score"] is not None
            and (parse_time(x["uploaded_at"], "uploaded_at"), x["id"]) < (t, e["id"])]
    delta = round(score - prev[-1]["public_score"], 10) if prev else None
    below = score < BASELINES[e["comp"]]
    drop = delta is not None and delta <= -DROP_ALERT + 1e-12
    e.update(public_score=score, public_delta_vs_prev=delta, investigate=bool(below or drop))
    if args.lb_rank is not None:
        e["lb_rank"] = int(args.lb_rank)
    if args.note:
        e["note"] = (e["note"] + " | " if e["note"] else "") + args.note
    save_log(log_path, log)
    ref = f"previous #{prev[-1]['id']} {prev[-1]['public_score']}" if prev else "no previous upload"
    print(f"entry #{e['id']} public {score} (delta {delta} vs {ref}; baseline {BASELINES[e['comp']]}); "
          f"investigate={e['investigate']}")
    if e["investigate"]:
        why = []
        if below:
            why.append(f"below the official baseline {BASELINES[e['comp']]}")
        if drop:
            why.append(f"{delta:+.4f} vs the previous upload (<= -{DROP_ALERT})")
        print("INVESTIGATE (U3): " + "; ".join(why) + ". Look for bugs in id mapping, row order, pool pairing, "
              "seam and guard level before any further upload. Never change the method because of the score.")
    log_all21, e_all21 = log, e          # PREREG_21: the 06 / 10 warnings never see a 21 entry (t21_public_lines does)
    if t21_entries(log):
        log, e = t21_hide(log), (dict(e, public_score=None) if e.get("test21") else e)
    for line in t04_public_lines(log, e):   # PREREG_04 rev 3: 04 and R20 share the file name submission_rank.csv
        print(line)
    for line in t06_public_lines(log, e):   # PREREG_06 5.3: Q04, 06-OFF and 06-ON share submission_player.csv
        print(line)
    for line in t07_public_lines(log, e):   # PREREG_07 5.4: 07, 04 and R20 share submission_rank.csv
        print(line)
    for line in t08_public_lines(log, e):   # PREREG_08 5.4: 08, 07, 04 and R20 share submission_rank.csv
        print(line)
    for line in t09_public_lines(log, e):   # PREREG_09 5.3: 09, 08, 07, 04 and R20 share submission_rank.csv
        print(line)
    for line in t19_public_lines(log, e):   # PREREG_19 4.3: 19, 09, 08, 07, 04 and R20 share submission_rank.csv
        print(line)
    for line in t25_public_lines(log, e):   # PREREG_25 4.3: 25, 19, 09, 08, 07, 04 and R20 share submission_rank.csv
        print(line)
    for line in t10_public_lines(log, e):   # PREREG_10 4.4: 10, 06-OFF and Q04 share submission_player.csv
        print(line)
    for line in t14_public_lines(log, e):   # PREREG_14: 14 and its base share submission_rank.csv
        print(line)
    log, e = log_all21, e_all21
    for line in t21_public_lines(log, e):   # PREREG_21: 21 and its base share submission_player.csv
        print(line)
    return e


def cmd_final(args):
    log_path = args.log or default_log()
    log = load_log(log_path)
    e = entry(log, args.id)
    if e["comp"] != args.comp:
        raise LogError(f"entry #{e['id']} is for {e['comp']}, not {args.comp}")
    if not (args.basis or "").strip():
        raise LogError("--basis TEXT is required (which registration / local-validation result makes it final)")
    e14 = None
    if e.get("test14"):                  # PREREG_14 7: the 04-09 rules below judge a 14 final through its base
        e14, e = e, t14_base_entry(log, e)
        rec_r = (e14["test14"] or {}).get("base_rule_reasons") or {}
        args = argparse.Namespace(**dict(vars(args), **{k: v for k, v in rec_r.items() if v
                                                         and not (getattr(args, k, None) or "").strip()}))
    e21 = None
    if e.get("test21"):                  # PREREG_21 7: the 06 / 10 rules below judge a 21 final through its base
        e21, e = e, t21_base_entry(log, e)
        rec_21 = (e21["test21"] or {}).get("base_rule_reasons") or {}
        args = argparse.Namespace(**dict(vars(args), **{k: v for k, v in rec_21.items() if v
                                                         and not (getattr(args, k, None) or "").strip()}))
        log_all21, log = log, t21_hide(log)      # ... on the log without 21 entries (PREREG_21 5)
    rule04 = final_rule_t04(log, e, getattr(args, "not_04_reason", None))      # PREREG_04 5.3 (None: no 04 entry)
    rule06 = final_rule_t06(log, e, getattr(args, "not_06_reason", None))      # PREREG_06 5.4 (None: no 06 entry)
    rule07 = final_rule_t07(log, e, getattr(args, "not_07_reason", None))      # PREREG_07 5.4 (None: no 07 entry)
    rule08 = final_rule_t08(log, e, getattr(args, "not_08_reason", None))      # PREREG_08 5.4 (None: no 08 entry)
    rule09 = final_rule_t09(log, e, getattr(args, "not_09_reason", None))      # PREREG_09 5.3 (None: no 09 entry)
    rule19 = final_rule_t19(log, e, getattr(args, "not_19_reason", None))      # PREREG_19 4.3 (None: no 19 entry)
    rule25 = final_rule_t25(log, e, getattr(args, "not_25_reason", None))      # PREREG_25 4.3 (None: no 25 entry)
    rule10 = final_rule_t10(log, e, getattr(args, "not_10_reason", None))      # PREREG_10 4.4 (None: no 10 entry)
    if e21 is not None:                  # PREREG_21 7: back to the full log and the 21 entry, which is checked itself
        log, e = log_all21, e21
        rule21 = final_rule_t21_self(log, e21, getattr(args, "not_21_reason", None))
    else:
        rule21 = final_rule_t21_other(log, e, getattr(args, "not_21_reason", None))   # None: no 21 entry
    if e14 is not None:                  # PREREG_14 7: back to the 14 entry, which is checked itself
        e, rule14 = e14, final_rule_t14_self(log, e14, getattr(args, "not_14_reason", None))
    else:
        rule14 = final_rule_t14_other(log, e, getattr(args, "not_14_reason", None))   # None: no 14 entry
    old = log["final_choice"].get(args.comp)
    if old:
        log.setdefault("final_choice_history", []).append(dict(old, replaced_at=now_iso()))
    log["final_choice"][args.comp] = {"id": e["id"], "sha256": e["sha256"], "file": e["file"],
                                      "method": e["method"], "basis": args.basis, "set_at": now_iso()}
    if rule04:
        log["final_choice"][args.comp]["prereg04_5_3"] = rule04
    if rule06:
        log["final_choice"][args.comp]["prereg06_5_4"] = rule06
    if rule07:
        log["final_choice"][args.comp]["prereg07_5_4"] = rule07
    if rule08:
        log["final_choice"][args.comp]["prereg08_5_4"] = rule08
    if rule09:
        log["final_choice"][args.comp]["prereg09_5_3"] = rule09
    if rule19:
        log["final_choice"][args.comp]["prereg19_4_3"] = rule19
    if rule25:
        log["final_choice"][args.comp]["prereg25_4_3"] = rule25
    if rule10:
        log["final_choice"][args.comp]["prereg10_4_4"] = rule10
    if rule14:
        log["final_choice"][args.comp]["prereg14_7"] = rule14
    if rule21:
        log["final_choice"][args.comp]["prereg21_7"] = rule21
    save_log(log_path, log)
    print(f"final choice for {args.comp}: entry #{e['id']} {e['file']} sha256 {e['sha256'][:12]}")
    if rule04:
        rev = rule04.get("revision")
        print(f"PREREG_04 5.3：最終選擇是 {rule04.get('chosen')}；符合條件的 04 紀錄 {rule04.get('eligible_04_entries')}"
              + (f"；依 PREREG_04 rev {rev['rev']}（{rev['commit'][:12]}）不選 04" if rev else ""))
        if e["status"] == "prepared":
            print(f"注意：#{e['id']} 這一筆還沒上傳（status prepared）；最後一次上傳要是它，最終選擇才算數")
    if rule06:
        rev = rule06.get("revision")
        print(f"PREREG_06 5.4：最終選擇是 {rule06.get('chosen')}；規則指定 {rule06.get('designated')}"
              + (f"；依 PREREG_06 rev {rev['rev']}（{rev['commit'][:12]}）改選" if rev else ""))
        if e["status"] == "prepared":
            print(f"注意：#{e['id']} 這一筆還沒上傳（status prepared）；最後一次上傳要是它，最終選擇才算數")
    if rule07:
        rev = rule07.get("revision")
        print(f"PREREG_07 5.4：最終選擇是 {rule07.get('chosen')}；符合條件的 07 紀錄 {rule07.get('eligible_07_entries')}"
              + (f"；依 PREREG_07 rev {rev['rev']}（{rev['commit'][:12]}）不選 07" if rev else ""))
        if e["status"] == "prepared" and not rule04:
            print(f"注意：#{e['id']} 這一筆還沒上傳（status prepared）；最後一次上傳要是它，最終選擇才算數")
    if rule08:
        rev = rule08.get("revision")
        print(f"PREREG_08 5.4：最終選擇是 {rule08.get('chosen')}；符合條件的 08 紀錄 {rule08.get('eligible_08_entries')}"
              + (f"；依 PREREG_08 rev {rev['rev']}（{rev['commit'][:12]}）不選 08" if rev else ""))
        if e["status"] == "prepared" and not rule04 and not rule07:
            print(f"注意：#{e['id']} 這一筆還沒上傳（status prepared）；最後一次上傳要是它，最終選擇才算數")
    if rule09:
        rev = rule09.get("revision")
        print(f"PREREG_09 5.3：最終選擇是 {rule09.get('chosen')}；符合條件的 09 紀錄 {rule09.get('eligible_09_entries')}"
              + (f"；依 PREREG_09 rev {rev['rev']}（{rev['commit'][:12]}）不選 09" if rev else ""))
        if e["status"] == "prepared" and not rule04 and not rule07 and not rule08:
            print(f"注意：#{e['id']} 這一筆還沒上傳（status prepared）；最後一次上傳要是它，最終選擇才算數")
    if rule19:
        rev = rule19.get("revision")
        print(f"PREREG_19 4.3：最終選擇是 {rule19.get('chosen')}；符合條件的 19 紀錄 {rule19.get('eligible_19_entries')}"
              + (f"；依 PREREG_19 rev {rev['rev']}（{rev['commit'][:12]}）不選 19" if rev else ""))
        if e["status"] == "prepared" and not rule04 and not rule07 and not rule08 and not rule09:
            print(f"注意：#{e['id']} 這一筆還沒上傳（status prepared）；最後一次上傳要是它，最終選擇才算數")
    if rule25:
        rev = rule25.get("revision")
        print(f"PREREG_25 4.3：最終選擇是 {rule25.get('chosen')}；符合條件的 25 紀錄 {rule25.get('eligible_25_entries')}"
              + (f"；依 PREREG_25 rev {rev['rev']}（{rev['commit'][:12]}）不選 25" if rev else ""))
        if e["status"] == "prepared" and not rule04 and not rule07 and not rule08 and not rule09 and not rule19:
            print(f"注意：#{e['id']} 這一筆還沒上傳（status prepared）；最後一次上傳要是它，最終選擇才算數")
    if rule10:
        rev = rule10.get("revision")
        print(f"PREREG_10 4.4：最終選擇是 {rule10.get('chosen')}；規則指定 {rule10.get('designated')}"
              + (f"；依 PREREG_10 rev {rev['rev']}（{rev['commit'][:12]}）改選" if rev else ""))
        if e["status"] == "prepared":
            print(f"注意：#{e['id']} 這一筆還沒上傳（status prepared）；最後一次上傳要是它，最終選擇才算數")
    if rule14:
        rev = rule14.get("revision")
        print(f"PREREG_14 7：最終選擇是 {rule14.get('chosen')}；符合條件的 14 紀錄 {rule14.get('eligible_14_entries')}"
              + (f"；底檔 #{rule14.get('base_entry_id')}" if rule14.get("base_entry_id") else "")
              + (f"；依 PREREG_14 rev {rev['rev']}（{rev['commit'][:12]}）不選 14" if rev else ""))
        if e["status"] == "prepared" and e.get("test14"):
            print(f"注意：#{e['id']} 這一筆還沒上傳（status prepared）；最後一次上傳要是它，最終選擇才算數")
    if rule21:
        rev = rule21.get("revision")
        print(f"PREREG_21 7：最終選擇是 {rule21.get('chosen')}；符合條件的 21 紀錄 {rule21.get('eligible_21_entries')}"
              + (f"；底檔 #{rule21.get('base_entry_id')}" if rule21.get("base_entry_id") else "")
              + (f"；依 PREREG_21 rev {rev['rev']}（{rev['commit'][:12]}）不選 21" if rev else ""))
        if e["status"] == "prepared" and e.get("test21"):
            print(f"注意：#{e['id']} 這一筆還沒上傳（status prepared）；最後一次上傳要是它，最終選擇才算數")
    return log["final_choice"][args.comp]


def _left(now, until):
    s = (until - now).total_seconds()
    if s <= 0:
        return "已過"
    d, rest = divmod(int(s), 86400)
    return f"{d} 天 {rest // 3600} 小時"


def cmd_status(args):
    log = load_log(args.log or default_log())
    now = parse_time(args.now, "--now") if args.now else datetime.now(TW)
    lines, out = [], {}
    for comp in ("rank", "player"):
        ups = uploads_of(log, comp)
        fin = log["final_choice"].get(comp)
        last = ups[-1] if ups else None
        per_day = Counter(local_date(x["uploaded_at"]) for x in ups)
        pending = [x["id"] for x in log["uploads"] if x["comp"] == comp and x["status"] == "prepared"]
        last_is_final = bool(last and fin and last["sha256"] == fin["sha256"])
        out[comp] = {"last_upload": last and last["id"], "last_is_final": last_is_final,
                     "uploads_per_day": dict(sorted(per_day.items())), "prepared_not_uploaded": pending}
        fin14, last14 = fin, last            # PREREG_14 7: the 04-09 blocks judge a 14 final through its base
        v14 = t14_status_view(log, comp, fin, last)
        if v14:
            fin, last = v14
        log_all21 = log                      # PREREG_21 7: the 06 / 10 blocks judge a 21 final through its base
        v21 = t21_status_view(log, comp, fin, last)
        if v21:
            log, fin, last = v21
        if comp == "rank":
            out[comp]["prereg04"] = t04_status(log, fin, last)
        if comp == "player" and t06_entries(log):     # PREREG_06 rev 3: only with 06 entries in the log
            out[comp]["prereg06"] = t06_status(log, fin, last, now)
        if comp == "rank" and t07_entries(log):       # PREREG_07 rev 4: only with 07 entries in the log
            out[comp]["prereg07"] = t07_status(log, fin, last)
        if comp == "rank" and t08_entries(log):       # PREREG_08 rev 3: only with 08 entries in the log
            out[comp]["prereg08"] = t08_status(log, fin, last)
        if comp == "rank" and t09_entries(log):       # PREREG_09 rev 3: only with 09 entries in the log
            out[comp]["prereg09"] = t09_status(log, fin, last)
        if comp == "rank" and t19_entries(log):       # PREREG_19 rev 3: only with 19 entries in the log
            out[comp]["prereg19"] = t19_status(log, fin, last)
        if comp == "rank" and t25_entries(log):       # PREREG_25 rev 3: only with 25 entries in the log
            out[comp]["prereg25"] = t25_status(log, fin, last)
        if comp == "player" and t10_entries(log):     # PREREG_10 rev 3: only with 10 entries in the log
            out[comp]["prereg10"] = t10_status(log, fin, last, now)
        log = log_all21
        fin, last = fin14, last14
        if comp == "rank" and t14_entries(log):       # PREREG_14: only with 14 entries in the log
            out[comp]["prereg14"] = t14_status(log, fin, last)
        if comp == "player" and t21_entries(log):     # PREREG_21: only with 21 entries in the log
            out[comp]["prereg21"] = t21_status(log, fin, last)
        lines.append(f"{COMP_LABEL[comp]} {TEAMS[comp]}")
        if last:
            lines.append(f"  最後一次上傳：#{last['id']} {last['uploaded_at']} sha256 {last['sha256'][:12]} "
                         f"public {last['public_score']}")
        else:
            lines.append("  還沒有上傳")
        if fin:
            lines.append(f"  最終選擇：#{fin['id']} sha256 {fin['sha256'][:12]}；最後一次上傳"
                         f"{'就是' if last_is_final else '不是'}它")
        else:
            lines.append("  最終選擇：還沒定")
        for day, n in sorted(per_day.items()):
            lines.append(f"  {day}：{n} 次{'（超過每天 %d 次）' % MAX_PER_DAY if n > MAX_PER_DAY else ''}")
        if pending:
            lines.append(f"  已 prepare 但沒記上傳：{pending}")
        if comp == "rank" and out[comp]["prereg04"]["entries"]:
            lines += t04_status_lines(out[comp]["prereg04"])
        if comp == "player" and out[comp].get("prereg06"):
            lines += t06_status_lines(out[comp]["prereg06"], now)
        if comp == "rank" and out[comp].get("prereg07"):
            lines += t07_status_lines(out[comp]["prereg07"])
        if comp == "rank" and out[comp].get("prereg08"):
            lines += t08_status_lines(out[comp]["prereg08"])
        if comp == "rank" and out[comp].get("prereg09"):
            lines += t09_status_lines(out[comp]["prereg09"])
        if comp == "rank" and out[comp].get("prereg19"):
            lines += t19_status_lines(out[comp]["prereg19"])
        if comp == "rank" and out[comp].get("prereg25"):
            lines += t25_status_lines(out[comp]["prereg25"])
        if comp == "player" and out[comp].get("prereg10"):
            lines += t10_status_lines(out[comp]["prereg10"], now)
        if comp == "rank" and out[comp].get("prereg14"):
            lines += t14_status_lines(out[comp]["prereg14"])
        if comp == "player" and out[comp].get("prereg21"):
            lines += t21_status_lines(out[comp]["prereg21"])
    lines.append(f"距 11/23 23:59（最後一次上傳的目標）：{_left(now, FINAL_TARGET)}；"
                 f"距 11/24 23:59（截止）：{_left(now, DEADLINE)}")
    lines.append(f"距 11/16 23:59（K6：04 的正式執行要在這之前 commit）：{_left(now, K6)}")
    if t07_entries(log):
        lines.append(f"距 11/19 23:59（K6-07：07 的正式執行要在這之前 commit）：{_left(now, K6_07)}")
    if t08_entries(log):
        lines.append(f"距 11/19 23:59（K6-08：08 的正式執行要在這之前 commit）：{_left(now, K6_08)}")
    if t09_entries(log):
        lines.append(f"距 11/19 23:59（K6-09：09 的正式執行要在這之前 commit）：{_left(now, K6_09)}")
    if t19_entries(log):
        lines.append(f"距 11/19 23:59（K6-19：19 的正式執行要在這之前 commit）：{_left(now, K6_19)}")
    if t25_entries(log):
        lines.append(f"距 11/18 23:59（K6-25：25 的正式執行要在這之前 commit）：{_left(now, K6_25)}")
    if t14_entries(log):
        lines.append(f"距 11/23 18:00（K6-14：14 的報告要在這之前 commit）：{_left(now, K6_14)}")
    if t21_entries(log):
        lines.append(f"距 11/23 18:00（K6-21：21 的報告要在這之前 commit）：{_left(now, K6_21)}")
    print("\n".join(lines))
    return out


# ---------------------------------------------------------------- CLI
def main(argv=None):
    for stream in (sys.stdout, sys.stderr):     # the OK request is Chinese; never cp950 on a pipe
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="official_log.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--comp", required=True, choices=("rank", "player"))
    p.add_argument("--file", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--method")
    p.add_argument("--tests-dir")
    p.add_argument("--checks-missing-reason", help="prepare although T2-T5 are not all recorded (stored in the entry)")
    p.add_argument("--not-04-reason", help="rank R20 / R10 file while a 04 entry meets PREREG_04 5.3: "
                                           "'PREREG_04 rev N: ...' citing a committed later revision (recorded)")
    p.add_argument("--not-06-reason", help="player Q04 file while a 06 entry meets PREREG_06 5.4: "
                                           "'PREREG_06 rev N: ...' citing a committed later revision (recorded)")
    p.add_argument("--not-07-reason", help="rank 04 / R20 / R10 file while a 07 entry meets PREREG_07 5.4: "
                                           "'PREREG_07 rev N: ...' citing a committed later revision (recorded)")
    p.add_argument("--not-08-reason", help="rank 07 / 04 / R20 / R10 file while an 08 entry meets PREREG_08 5.4: "
                                           "'PREREG_08 rev N: ...' citing a committed later revision (recorded)")
    p.add_argument("--not-09-reason", help="rank 08 / 07 / 04 / R20 / R10 file while a 09 entry meets PREREG_09 "
                                           "5.3: 'PREREG_09 rev N: ...' citing a committed later revision "
                                           "(recorded)")
    p.add_argument("--not-10-reason", help="player Q04 / 06-OFF / 06-ON file while a 10 upload is the final choice "
                                           "PREREG_10 4.4 designates: 'PREREG_10 rev N: ...' citing a committed later "
                                           "revision (recorded)")
    p.add_argument("--not-14-reason", help="rank file that is not 14 while a 14 upload meets PREREG_14 7: "
                                           "'PREREG_14 rev N: ...' citing a committed later revision (recorded)")
    p.add_argument("--not-19-reason", help="rank: a 09 / 08 / 07 / 04 / R20 / R10 file or final choice while a 19 "
                                           "entry meets PREREG_19 4.3; 'PREREG_19 rev N: ...' citing a committed "
                                           "later revision (recorded)")
    p.add_argument("--not-25-reason", help="rank: a 19 / 09 / 08 / 07 / 04 / R20 / R10 file or final choice "
                                           "while a 25 entry meets PREREG_25 4.3; 'PREREG_25 rev N: ...' citing a "
                                           "committed later revision (recorded)")
    p.add_argument("--not-21-reason", help="player: a file or final choice that is not 21 while a 21 upload meets "
                                           "PREREG_21 7; 'PREREG_21 rev N: ...' citing a committed later revision "
                                           "(recorded)")
    p = sub.add_parser("uploaded")
    p.add_argument("--id", required=True, type=int)
    p.add_argument("--ok-at", required=True)
    p.add_argument("--uploaded-at", required=True)
    p = sub.add_parser("public")
    p.add_argument("--id", required=True, type=int)
    p.add_argument("--score", required=True, type=float)
    p.add_argument("--lb-rank", type=int)
    p.add_argument("--note")
    p = sub.add_parser("final")
    p.add_argument("--comp", required=True, choices=("rank", "player"))
    p.add_argument("--id", required=True, type=int)
    p.add_argument("--basis", required=True)
    p.add_argument("--not-04-reason", help="rank: an R20 / R10 final choice although a 04 entry meets PREREG_04 5.3; "
                                           "'PREREG_04 rev N: ...' citing a committed later revision (recorded)")
    p.add_argument("--not-06-reason", help="player: a final choice other than the upload PREREG_06 5.4 designates; "
                                           "'PREREG_06 rev N: ...' citing a committed later revision (recorded)")
    p.add_argument("--not-07-reason", help="rank: a 04 / R20 / R10 final choice although a 07 entry meets PREREG_07 "
                                           "5.4; 'PREREG_07 rev N: ...' citing a committed later revision (recorded)")
    p.add_argument("--not-08-reason", help="rank: a 07 / 04 / R20 / R10 final choice although an 08 entry meets "
                                           "PREREG_08 5.4; 'PREREG_08 rev N: ...' citing a committed later revision "
                                           "(recorded)")
    p.add_argument("--not-09-reason", help="rank: an 08 / 07 / 04 / R20 / R10 final choice although a 09 entry "
                                           "meets PREREG_09 5.3; 'PREREG_09 rev N: ...' citing a committed later "
                                           "revision (recorded)")
    p.add_argument("--not-10-reason", help="player: a final choice other than the upload PREREG_10 4.4 designates; "
                                           "'PREREG_10 rev N: ...' citing a committed later revision (recorded)")
    p.add_argument("--not-14-reason", help="rank: a final choice other than a 14 upload that meets PREREG_14 7; "
                                           "'PREREG_14 rev N: ...' citing a committed later revision (recorded)")
    p.add_argument("--not-19-reason", help="rank: a 09 / 08 / 07 / 04 / R20 / R10 file or final choice while a 19 "
                                           "entry meets PREREG_19 4.3; 'PREREG_19 rev N: ...' citing a committed "
                                           "later revision (recorded)")
    p.add_argument("--not-25-reason", help="rank: a 19 / 09 / 08 / 07 / 04 / R20 / R10 file or final choice "
                                           "while a 25 entry meets PREREG_25 4.3; 'PREREG_25 rev N: ...' citing a "
                                           "committed later revision (recorded)")
    p.add_argument("--not-21-reason", help="player: a file or final choice that is not 21 while a 21 upload meets "
                                           "PREREG_21 7; 'PREREG_21 rev N: ...' citing a committed later revision "
                                           "(recorded)")
    p = sub.add_parser("status")
    p.add_argument("--now", help="ISO time for the countdown (default: now)")
    for name in ("prepare", "uploaded", "public", "final", "status"):
        sub.choices[name].add_argument("--log", help="default results/official_scores.json")
    args = ap.parse_args(argv)
    return {"prepare": cmd_prepare, "uploaded": cmd_uploaded, "public": cmd_public, "final": cmd_final,
            "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    main()
