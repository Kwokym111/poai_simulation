"""
efficiency_pipeline.py
----------------------
Plug in an efficiency formula, get back a miner-level label and a feature ranking.

The three things that vary between efficiency definitions are separated out:

    1. the formula      - a per-block calculation
    2. the normalisation - percentile rank within a time window, or none
    3. the threshold    - median split of the miner-level score

Everything else (cutoff, aggregation, minimum block count, long_tail exclusion)
is held fixed so that two formulas are always compared on the same footing.

Usage from a notebook, after `merged_df` and `drop_features` exist:

    import efficiency_pipeline as ep

    scores = ep.build_scores(merged_df, lambda d: d['total_fee_satoshis'] / d['size'])
    table, panel = ep.rank_features(drop_features, scores, ep.FEATURES_REAL)
    print(table)

Or to compare several at once:

    rankings, meta = ep.compare_formulas(merged_df, drop_features,
                                         ep.FEATURES_REAL, ep.FORMULAS)

Verify the pipeline reproduces a logged result before trusting a new one:

    ep.validate(merged_df, drop_features)
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

# 2011-01-01 per §103: pre-2011 fees are ~zero, so a low value there says
# nothing about the miner. Costs 12.38% of blocks and zero trackable miners.
DEFAULT_CUTOFF = pd.Timestamp('2011-01-01', tz='UTC')

FEATURES_REAL = ['blocks_mined', 'avg_fee_real', 'fee_volatility_real',
                 'avg_block_size', 'difficulty', 'profitability_real', 'age']


# ----------------------------------------------------------------------
# 1. Formula -> miner-level score and label
# ----------------------------------------------------------------------

def build_scores(blocks, formula, window='M', cutoff=DEFAULT_CUTOFF,
                 min_blocks=10, group_col='miner_group', exclude='long_tail',
                 verbose=True):
    """
    Turn a per-block formula into one score and one binary label per miner.

    blocks   : block-level frame (i.e. merged_df)
    formula  : callable taking the block frame and returning a per-block Series,
               e.g. lambda d: d['total_fee_satoshis'] / d['size']
    window   : 'M', 'Y' or 'W' to percentile-rank each block against all blocks
               in the same window before aggregating; None to use raw values.
               None leaves era drift in the label -- that is sometimes the point
               (reproducing a flawed label), but it is never the safe default.
    cutoff   : drop blocks before this timestamp; None to keep everything
    min_blocks : minimum blocks a miner must have AFTER the cutoff

    Returns a frame: miner_id, score, n_blocks, label.
    """
    df = blocks.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    if cutoff is not None:
        n0 = len(df)
        df = df[df['timestamp'] >= cutoff]
        if verbose:
            dropped = n0 - len(df)
            print(f"  cutoff {cutoff.date()}: dropped {dropped:,} of {n0:,} "
                  f"blocks ({dropped / n0:.2%})")

    # --- apply the formula ---
    values = pd.to_numeric(formula(df), errors='coerce')
    bad = ~np.isfinite(values)
    if bad.any():
        if verbose:
            print(f"  warning: {bad.sum():,} non-finite values from formula, dropped")
        df, values = df[~bad], values[~bad]
    df = df.assign(_value=values)

    # --- normalise within time window (this is what removes era drift) ---
    if window is None:
        df['_score'] = df['_value']
    else:
        # long_tail blocks stay in the ranking population, as in §84.4/§103
        df['_period'] = df['timestamp'].dt.tz_convert(None).dt.to_period(window)
        df['_score'] = df.groupby('_period')['_value'].rank(pct=True)

    # --- aggregate to miner level ---
    tracked = df[df[group_col] != exclude]
    out = (tracked.groupby(group_col)['_score']
                  .agg(score='mean', n_blocks='count')
                  .reset_index()
                  .rename(columns={group_col: 'miner_id'}))

    out = out[out['n_blocks'] >= min_blocks].reset_index(drop=True)
    out['label'] = (out['score'] > out['score'].median()).astype(int)
    return out


# ----------------------------------------------------------------------
# 2. Score -> feature ranking
# ----------------------------------------------------------------------

def rank_features(miner_features, scores, features=None, id_col='miner_id'):
    """
    Rank features by Spearman correlation with the score.

    Spearman rather than Pearson because the features are badly skewed (the
    reason §77 needed log1p). Rank correlation is invariant to monotone
    transforms, so the ranking does not depend on the transform scheme.

    Univariate correlation cannot see interactions -- a feature with near-zero
    marginal correlation can still matter inside a model. Treat this as a
    screen, not as a replacement for permutation importance.

    Returns (ranking_table, merged_panel).
    """
    features = features or FEATURES_REAL
    panel = miner_features.merge(scores, on=id_col, how='inner')

    rows = []
    for f in features:
        rho, p = spearmanr(panel[f], panel['score'])
        rows.append({'feature': f, 'spearman_rho': rho,
                     'abs_rho': abs(rho), 'p_value': p})

    table = (pd.DataFrame(rows)
             .sort_values('abs_rho', ascending=False)
             .reset_index(drop=True))
    return table, panel


def time_check(panel, score_col='score', time_col='last_block_time'):
    """
    How much is this label just a clock?

    Both statistics are reported deliberately. Pearson measures linear
    association; Spearman measures monotone association of any shape. An MLP
    can fit any monotone shape, so Spearman is the more relevant number when
    the question is whether the model can use the label as a proxy for date
    (§108.2). Reporting only one invites an unexplained discrepancy with §80.
    """
    t = pd.to_datetime(panel[time_col], utc=True).astype('int64')
    rho, _ = spearmanr(panel[score_col], t)
    r, _ = pearsonr(panel[score_col], t)
    return {'n': len(panel), 'spearman': rho, 'pearson': r}


# ----------------------------------------------------------------------
# 3. Compare several formulas side by side
# ----------------------------------------------------------------------

def compare_formulas(blocks, miner_features, features=None, formulas=None,
                     **kwargs):
    """
    formulas: dict of name -> (formula_callable, window)

    Returns (rankings, meta):
      rankings - one column per formula, one row per feature, Spearman rho
      meta     - per formula: n_miners, and the label's own correlation with time
    """
    features = features or FEATURES_REAL
    formulas = formulas or FORMULAS

    ranking_cols, meta = {}, []
    for name, (formula, window) in formulas.items():
        print(f"[{name}]  window={window}")
        scores = build_scores(blocks, formula, window=window, **kwargs)
        table, panel = rank_features(miner_features, scores, features)
        ranking_cols[name] = table.set_index('feature')['spearman_rho']

        tc = time_check(panel)
        meta.append({'formula': name, 'window': window, 'n_miners': tc['n'],
                     'score_vs_time_spearman': tc['spearman'],
                     'score_vs_time_pearson': tc['pearson']})
        print(f"  {tc['n']} miners | score vs time: "
              f"rho={tc['spearman']:+.4f}, r={tc['pearson']:+.4f}\n")

    return pd.DataFrame(ranking_cols).reindex(features), pd.DataFrame(meta)


# ----------------------------------------------------------------------
# 4. Formula registry
# ----------------------------------------------------------------------
#
# NOTE: ratio_of_means (avg_transactions / avg_volume) CANNOT go here. It is a
# ratio of miner-level aggregates, not a mean of per-block values, so it is not
# expressible as a per-block formula. It needs a separate path if ever tested.

FORMULAS = {
    # §103's grounded label, and the §108 headline configuration
    'fee_density_month':  (lambda d: d['total_fee_satoshis'] / d['size'], 'M'),

    # same label, coarser window -- §103 shows the window matters
    'fee_density_year':   (lambda d: d['total_fee_satoshis'] / d['size'], 'Y'),

    # absolute fee: partly genuine competence, partly a re-import of the
    # avg_block_size time confound (§103)
    'fee_raw_month':      (lambda d: d['total_fee_satoshis'], 'M'),

    # the original flawed label, era-corrected -- what mean_of_ratios looks
    # like once the global-median timestamp effect is removed
    'mean_of_ratios_month': (lambda d: d['transaction_count'] / d['total_output_satoshis'], 'M'),

    # the original flawed label as-is, for contrast. window=None keeps the
    # era drift, which is exactly what §83/§85/§87 identify as the problem.
    'mean_of_ratios_raw': (lambda d: d['transaction_count'] / d['total_output_satoshis'], None),
}


# ----------------------------------------------------------------------
# 5. Validation against a logged result
# ----------------------------------------------------------------------

# §108.1, fee_density / month / >=10 blocks / 2011 cutoff / drop grouping
EXPECTED_108 = {
    'fee_volatility_real':  0.2855,
    'profitability_real':   0.2742,
    'avg_fee_real':         0.2727,
    'blocks_mined':         0.1742,
    'age':                  0.1341,
    'difficulty':          -0.0875,
    'avg_block_size':      -0.0696,
}
EXPECTED_108_N = 975
EXPECTED_108_TIME_RHO = -0.0733


def validate(blocks, miner_features, tol=0.001):
    """
    Re-derive §108's ranking through this pipeline and compare to the logged
    values. Run this after any change to the pipeline, before trusting a new
    formula's output.
    """
    scores = build_scores(blocks, FORMULAS['fee_density_month'][0], window='M')
    table, panel = rank_features(miner_features, scores)
    tc = time_check(panel)

    print(f"\n{'feature':<22}{'got':>10}{'expected':>10}{'':>4}")
    ok = True
    for feat, exp in EXPECTED_108.items():
        got = table.loc[table['feature'] == feat, 'spearman_rho'].iloc[0]
        hit = abs(got - exp) < tol
        ok &= hit
        print(f"{feat:<22}{got:>10.4f}{exp:>10.4f}{'  ok' if hit else '  MISMATCH':>4}")

    n_hit = tc['n'] == EXPECTED_108_N
    t_hit = abs(tc['spearman'] - EXPECTED_108_TIME_RHO) < tol
    ok &= n_hit and t_hit
    print(f"{'n_miners':<22}{tc['n']:>10}{EXPECTED_108_N:>10}{'  ok' if n_hit else '  MISMATCH':>4}")
    print(f"{'score_vs_time':<22}{tc['spearman']:>10.4f}{EXPECTED_108_TIME_RHO:>10.4f}"
          f"{'  ok' if t_hit else '  MISMATCH':>4}")

    print("\nPASS -- pipeline reproduces §108\n" if ok else
          "\nFAIL -- do not trust new results until resolved\n")
    return ok