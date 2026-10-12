# Paper: quality checklist, open items, figures, and a Reviewer #2 critique
Draft: `paper.tex` + `references.bib` (IEEEtran journal). Compiles with tectonic 0.15 to 11 pages,
with no overfull boxes and no undefined references. A built copy is `paper.pdf`.

## Quality checklist
| Item | Status |
|---|---|
| One central claim, stated identically in abstract, intro and conclusion | ✅ Checked by string match: 3 identical occurrences |
| Every contribution backed by a number and a table or figure | ✅ C1 Fig. 1; C2 Table IV + Fig. 3; C3 Fig. 4; C4 Table V + Figs. 1, 5; C5 Table VI |
| Theory assumptions listed and each tested or flagged | ✅ A1 (separable a_t) is flagged in "When the result does not apply"; its failure mode (floored a_t) is observed at S1 and on VisDrone. ⚠️ A2 (additive cost) is approximate under subsampling; A3 only uses the shape and rescales with the exact accountant. ⚠️ Prop. 2's attacker threshold is labelled post hoc |
| Paired / controlled comparisons; per-trial values | ✅ CRN with 20 seeds everywhere; Fig. 6 shows per-seed differences for 9 headline comparisons |
| At least one failed or null result reported as such | ✅ Section VII and Table VI; nulls listed |
| Baselines reproduced fairly; prior work not misrepresented | ⚠️ FLTrust, median, Krum and the attacks are implemented as specified. Kiani et al.'s rule is realised via σ, not client subsampling. **Their benchmark is NOT reproduced** ([MISSING] in Sec. VII) |
| Data leakage checked; splits explained | ✅ Session-level split; official-split leak (23/89) documented |
| Privacy / security parameters stated plainly | ✅ Frame-level ε, δ 1e-5, worst client; unaccounted tuning cost disclosed |
| Limitations and open problem sections | ✅ |
| No [MISSING]/[VERIFY] left without your knowledge | ⚠️ See "Open items" below (3 MISSING in the text; 3 VERIFY notes in the .bib) |
| Abstract numbers = results = conclusion | ✅ 0.0075, +0.0139, 23.5%, 16/20, 75%, 0.0005, 0.0058, 0.059 |
| Title consistent | ✅ One title (\title and \markboth short form) |

## Open items you must resolve
1. **[MISSING] Authors and affiliations** (title block).
2. **[MISSING] Reproduction of Kiani et al. on their own benchmark** (MNIST/FMNIST/Adult, client-level
   DP), if the venue expects it. Alternatively, keep the sentence saying it was not done.
3. **[MISSING] Public repository URL and dataset licence statement** (Data availability).
4. **[VERIFY] in references.bib**: the Krum author list, the Anti-UAV report's full author list, and
   Holm's issue number. Everything else was checked online on 2026-10-12. The source of each check
   is in the `checked` field.
5. **Privacy unit**: phase0/PHASE0.md still lists it as PENDING. The paper states frame-level
   plainly. Confirm this is acceptable.
6. Abstract is ~200 words after trimming. The title is long; a shorter alternative is
   "Mostly Noise: Allocation and Robustness in DP Federated Learning".

## Figures
| Fig. | File | Status |
|---|---|---|
| 1 Mechanism vs σ | figs/fig_mechanism.pdf | New, vector, from `make_figures.py` (values from the Phase 5b and 5c tables) |
| 2 Round sensitivity | figs/fig_pulse.pdf | New, vector, from the stored pulse analyses |
| 3 Schedules | figs/phase4_schedules.png | **Regenerate as vector** (from `code/analyse_phase4*.py`; matplotlib, savefig .pdf) |
| 4 Rounds | figs/phase4g_rounds.png | **Regenerate as vector** (`code/analyse_phase4g.py`) |
| 5 Trust cap | figs/phase5d_trust_cap.png | **Regenerate as vector** (`code/analyse_phase5d.py`) |
| 6 Per-seed differences | figs/fig_perseed.pdf | New, vector |
| (spare) | figs/phase4h_relative_agent.png | Not used in the text; available for an appendix |

The PNGs also use the analysis scripts' default fonts. For the final version, set the font size to
8 pt and use one column (3.5 in) or two columns (7 in).

## Reviewer #2: the five strongest objections and how to answer them
1. **"Your ε is not the ε that matters."** Frame-level DP on video protects one frame, not a
   recording. A recording of 1,500 correlated frames gets a far weaker guarantee by group privacy.
   And the schedules were tuned using runs on client data (A2's k on val, A3's a_t from 3,120
   client-data runs) without accounting.
   *Address:*
   - (a) Report a recording-level bound (group privacy with the per-recording frame count), or
     re-run the main P comparison with session-level sampling.
   - (b) Show that A2 with an untuned k (e.g. 0.5 fixed in advance on public proxy data) keeps the
     gain.
   - (c) Cite Papernot & Steinke and give an upper bound on the tuning cost.

2. **"Absolute accuracy is too low to matter; this is a toy regime."** Uniform scores 0.059 on test
   and the gain is +0.014. Effects on a 65k-parameter head may not transfer to real detectors.
   *Address:*
   - Show the validation-pool numbers (~0.29) alongside.
   - Explain the test shift (tiny targets: median box 110 vs 621 px²).
   - Add a larger-D configuration (the 116k variant exists in the pulse test) for the main
     comparison, or frame the paper explicitly as the strong-privacy, small-D regime.
   - Note that VisDrone replicates the direction.

3. **"The test pool was reused across eight phases."** The same 12 sessions served many families,
   with Holm only within families. That is a garden of forking paths.
   *Address:*
   - Point to the pre-registration commits (decisions fixed before each run).
   - Add a global Holm or Benjamini-Hochberg correction across all primary families as a robustness
     table.
   - Or hold out a fresh session set for a single confirmatory re-run of the three headline
     comparisons.

4. **"The adaptive attacker is unrealistically strong, and you did not try DP-aware robust
   aggregators."** The trust-capture attacker knows g0 exactly. Real attackers estimate it. No
   DP-native Byzantine-robust method from the literature was compared.
   *Address:*
   - Add an attacker that estimates g0 from its own clean data or from the honest mean.
   - Add at least one DP-aware robust aggregator from the literature as a baseline.
   - State that Proposition 2's threshold (~2.5%) predicts that even an estimated direction suffices
     once it is coherent, and test that prediction.

5. **"The 'unifying' claim is a narrative, not a result."** The allocation gain is explained by a_t
   (Prop. 1), not by the cosine (Prop. 2). The attacker threshold was derived after seeing the cap
   results. And CIFAR-10 shows a_t can rise without a meaningful gain.
   *Address:*
   - Keep Prop. 2's threshold explicitly labelled post hoc (done).
   - Pre-register and run one confirmatory test of it: a cap at κ ≈ 2% should begin to protect, at
     an accuracy cost.
   - Narrow the claim's allocation half to "no feedback signal beyond the schedule" (Sec. V-F
     evidence) rather than to the cosine itself.
   - Present CIFAR-10 as a limit on the claim, not a footnote (it is already in Sec. V-E).
