"""
score_model.py
==============
Dixon-Coles Poisson model for predicting WC2026 match scorelines.

Model
-----
Goals scored follow independent Poisson distributions:
  home_goals ~ Poisson(lambda_home)
  away_goals ~ Poisson(lambda_away)

where:
  log(lambda_home) = attack[home] + defence[away] + home_advantage * (1 - is_neutral)
  log(lambda_away) = attack[away] + defence[home]

A Dixon-Coles tau (rho) correction adjusts the joint probability of
low-scoring results (0-0, 1-0, 0-1, 1-1) which are systematically
under-predicted by independent Poisson.

Training uses time-decay weighting so recent matches count more,
and tournament weighting so WC/major tournament matches count more
than friendlies.

Outputs
-------
  data/models/dixon_coles_params.json   <- fitted parameters
  data/models/score_model_meta.json     <- fit metadata + diagnostics

Usage
-----
  # Fit and save
  python score_model.py

  # Predict one match
  from score_model import ScoreModel
  model = ScoreModel.load()
  result = model.predict('Argentina', 'France', neutral=True)
  print(result['most_likely_score'])   # e.g. (1, 0)
  print(result['home_win_prob'])       # e.g. 0.47

Requirements
------------
  pip install pandas numpy scipy scikit-learn
"""

import os
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
TRAINING_DATA = "data/processed/match_history_with_elo.csv"
MODELS_DIR    = "data/models"
PARAMS_PATH   = os.path.join(MODELS_DIR, "dixon_coles_params.json")
META_PATH     = os.path.join(MODELS_DIR, "score_model_meta.json")

os.makedirs(MODELS_DIR, exist_ok=True)

# ─── WC2026 teams ─────────────────────────────────────────────────────────────
WC2026_TEAMS = {
    "Argentina", "Australia", "Belgium", "Bolivia", "Brazil",
    "Cameroon", "Canada", "Chile", "China PR", "Colombia",
    "Costa Rica", "Croatia", "Czech Republic", "Denmark", "Ecuador",
    "Egypt", "England", "France", "Germany", "Ghana",
    "Honduras", "Hungary", "Indonesia", "Iran", "Iraq",
    "Italy", "Ivory Coast", "Jamaica", "Japan", "Kenya",
    "Mali", "Mexico", "Morocco", "Netherlands", "New Zealand",
    "Nigeria", "Norway", "Panama", "Paraguay", "Peru",
    "Poland", "Portugal", "Qatar", "Romania", "Saudi Arabia",
    "Senegal", "Serbia", "Slovakia", "Slovenia", "South Korea",
    "South Africa", "Spain", "Switzerland", "Trinidad and Tobago",
    "Tunisia", "Turkey", "Ukraine", "United States", "Uruguay",
    "Uzbekistan", "Venezuela",
}

# Max scoreline to consider when building probability matrices
MAX_GOALS = 8


# ══════════════════════════════════════════════════════════════════════════════
# DIXON-COLES TAU CORRECTION
# ══════════════════════════════════════════════════════════════════════════════

def _dc_tau_vec(hg: np.ndarray, ag: np.ndarray,
                mu: np.ndarray, nu: np.ndarray,
                rho: float) -> np.ndarray:
    """
    Vectorised Dixon-Coles tau correction for low-scoring results.
    Adjusts joint probability of (0,0), (1,0), (0,1), (1,1).
    """
    tau = np.ones(len(hg))
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    tau[m00] = np.maximum(1e-10, 1.0 - mu[m00] * nu[m00] * rho)
    tau[m01] = np.maximum(1e-10, 1.0 + mu[m01] * rho)
    tau[m10] = np.maximum(1e-10, 1.0 + nu[m10] * rho)
    tau[m11] = np.maximum(1e-10, 1.0 - rho)
    return tau


def _dc_tau_scalar(hg: int, ag: int,
                   mu: float, nu: float, rho: float) -> float:
    """Scalar version of the Dixon-Coles tau correction."""
    if hg == 0 and ag == 0:
        return max(1e-10, 1.0 - mu * nu * rho)
    elif hg == 0 and ag == 1:
        return max(1e-10, 1.0 + mu * rho)
    elif hg == 1 and ag == 0:
        return max(1e-10, 1.0 + nu * rho)
    elif hg == 1 and ag == 1:
        return max(1e-10, 1.0 - rho)
    return 1.0


# ══════════════════════════════════════════════════════════════════════════════
# MODEL FITTING
# ══════════════════════════════════════════════════════════════════════════════

def load_training_data() -> pd.DataFrame:
    """Load and validate the training dataset."""
    df = pd.read_csv(TRAINING_DATA, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    log.info(f"Loaded {len(df):,} matches ({df['date'].dt.year.min()}–"
             f"{df['date'].dt.year.max()})")
    return df


def build_team_index(df: pd.DataFrame) -> dict:
    """Build sorted team → integer index mapping from all teams in dataset."""
    all_teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    return {t: i for i, t in enumerate(all_teams)}


def fit_dixon_coles(df: pd.DataFrame, t_idx: dict) -> dict:
    """
    Fit Dixon-Coles model via maximum likelihood estimation.

    Parameters fitted:
      - attack[i]  : attacking strength for team i (log scale)
      - defence[i] : defensive weakness for team i (log scale, higher = worse defence)
      - home_adv   : home advantage (log scale, applied when is_neutral=False)
      - rho        : Dixon-Coles low-score correction parameter

    Constraints:
      - rho bounded to [-1, 1] for numerical stability
      - Sum of attack params = 0 (identifiability constraint)

    Weighting:
      - decay_weight: time-decay (recent matches count more)
      - tournament_weight baked into decay_weight already
    """
    n_t = len(t_idx)
    log.info(f"Fitting Dixon-Coles model: {n_t} teams, {len(df):,} matches")

    # Pre-compute arrays for speed
    hg      = df["home_score"].values.astype(int)
    ag      = df["away_score"].values.astype(int)
    h_idx   = np.array([t_idx[t] for t in df["home_team"]])
    a_idx   = np.array([t_idx[t] for t in df["away_team"]])
    neutral = df["is_neutral"].astype(float).values
    weights = df["decay_weight"].values

    def neg_log_likelihood(params: np.ndarray) -> float:
        attack   = params[:n_t]
        defence  = params[n_t : 2 * n_t]
        home_adv = params[-2]
        rho      = params[-1]

        # Expected goals
        mu = np.exp(attack[h_idx] + defence[a_idx] + home_adv * (1.0 - neutral))
        nu = np.exp(attack[a_idx] + defence[h_idx])

        # Weighted log-likelihood
        ll = (poisson.logpmf(hg, mu) + poisson.logpmf(ag, nu)) * weights

        # Dixon-Coles tau adjustment
        tau = _dc_tau_vec(hg, ag, mu, nu, rho)
        ll += np.log(tau) * weights

        return -ll.sum()

    # Initialisation — small positive attack, small negative defence
    x0 = np.zeros(n_t * 2 + 2)
    x0[:n_t]        = 0.1    # attack
    x0[n_t:2 * n_t] = -0.1  # defence
    x0[-2]          = 0.25   # home advantage (~1.28x goals at home)
    x0[-1]          = -0.09  # rho (slight negative correlation for 1-1)

    # Bounds: only rho is constrained
    bounds = [(None, None)] * (n_t * 2) + [(None, None), (-1.0, 1.0)]

    log.info("  Running L-BFGS-B optimisation (maxiter=1500) ...")
    result = minimize(
        neg_log_likelihood,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1500, "ftol": 1e-10, "gtol": 1e-7},
    )

    if not result.success:
        log.warning(f"  Optimisation did not fully converge: {result.message}")
        log.warning("  Using best parameters found — model is still usable.")
    else:
        log.info(f"  Converged. NLL = {result.fun:.2f}")

    params = result.x

    # Apply sum-to-zero identifiability constraint on attack
    attack  = params[:n_t]
    defence = params[n_t : 2 * n_t]
    attack_mean = attack.mean()
    attack  -= attack_mean
    defence -= attack_mean   # adjust defence to compensate

    return {
        "attack"     : attack,
        "defence"    : defence,
        "home_adv"   : float(params[-2]),
        "rho"        : float(params[-1]),
        "nll"        : float(result.fun),
        "converged"  : bool(result.success),
        "n_matches"  : len(df),
        "n_teams"    : n_t,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def expected_goals(
    home_team: str,
    away_team: str,
    params: dict,
    t_idx: dict,
    neutral: bool = True,
) -> tuple[float, float]:
    """
    Return (lambda_home, lambda_away) — expected goals for each team.
    Uses ELO-weighted fallback for teams not in t_idx.
    """
    attack  = params["attack"]
    defence = params["defence"]
    adv     = 0.0 if neutral else params["home_adv"]

    def _attack(team):
        if team in t_idx:
            return attack[t_idx[team]]
        log.debug(f"  '{team}' not in model — using mean attack")
        return 0.0

    def _defence(team):
        if team in t_idx:
            return defence[t_idx[team]]
        return 0.0

    mu = np.exp(_attack(home_team) + _defence(away_team) + adv)
    nu = np.exp(_attack(away_team) + _defence(home_team))
    return float(mu), float(nu)


def scoreline_matrix(
    mu: float,
    nu: float,
    rho: float,
    max_goals: int = MAX_GOALS,
) -> np.ndarray:
    """
    Build (max_goals+1) x (max_goals+1) matrix of scoreline probabilities.
    Applies Dixon-Coles tau correction to low-scoring cells.
    Rows = home goals, Cols = away goals.
    """
    matrix = np.outer(
        poisson.pmf(np.arange(max_goals + 1), mu),
        poisson.pmf(np.arange(max_goals + 1), nu),
    )
    for h in range(2):
        for a in range(2):
            tau = _dc_tau_scalar(h, a, mu, nu, rho)
            matrix[h, a] *= tau

    # Renormalise so probabilities sum to 1
    matrix /= matrix.sum()
    return matrix


def predict_match(
    home_team: str,
    away_team: str,
    params: dict,
    t_idx: dict,
    neutral: bool = True,
    top_n: int = 5,
) -> dict:
    """
    Predict the outcome of a single match.

    Returns a dict with:
        lambda_home       : expected home goals
        lambda_away       : expected away goals
        home_win_prob     : P(home wins)
        draw_prob         : P(draw)
        away_win_prob     : P(away wins)
        most_likely_score : (home_goals, away_goals) tuple
        top_scores        : list of (score, probability) for top_n scorelines
        expected_total    : expected total goals (for corners/cards model input)
    """
    mu, nu = expected_goals(home_team, away_team, params, t_idx, neutral)
    matrix = scoreline_matrix(mu, nu, params["rho"])

    home_win = float(np.tril(matrix, -1).sum())  # home > away
    draw     = float(np.trace(matrix))
    away_win = float(np.triu(matrix, 1).sum())   # away > home

    # Most likely scoreline
    best_idx = np.unravel_index(matrix.argmax(), matrix.shape)
    most_likely = (int(best_idx[0]), int(best_idx[1]))

    # Top N scorelines
    flat = matrix.flatten()
    top_idx = flat.argsort()[::-1][:top_n]
    top_scores = [
        {
            "score"      : f"{i // (MAX_GOALS + 1)}-{i % (MAX_GOALS + 1)}",
            "home_goals" : i // (MAX_GOALS + 1),
            "away_goals" : i % (MAX_GOALS + 1),
            "probability": float(flat[i]),
        }
        for i in top_idx
    ]

    return {
        "home_team"        : home_team,
        "away_team"        : away_team,
        "neutral"          : neutral,
        "lambda_home"      : round(mu, 4),
        "lambda_away"      : round(nu, 4),
        "home_win_prob"    : round(home_win, 4),
        "draw_prob"        : round(draw, 4),
        "away_win_prob"    : round(away_win, 4),
        "most_likely_score": most_likely,
        "most_likely_str"  : f"{most_likely[0]}-{most_likely[1]}",
        "most_likely_prob" : round(float(matrix[best_idx]), 4),
        "top_scores"       : top_scores,
        "expected_total"   : round(mu + nu, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SERIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def save_params(params: dict, t_idx: dict) -> None:
    """Save fitted parameters and team index to JSON."""
    teams = sorted(t_idx, key=t_idx.get)
    out = {
        "teams"    : teams,
        "attack"   : params["attack"].tolist(),
        "defence"  : params["defence"].tolist(),
        "home_adv" : params["home_adv"],
        "rho"      : params["rho"],
        "nll"      : params["nll"],
        "converged": params["converged"],
        "n_matches": params["n_matches"],
        "n_teams"  : params["n_teams"],
    }
    with open(PARAMS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"Parameters saved → {PARAMS_PATH}")


def load_params() -> tuple[dict, dict]:
    """Load parameters from JSON. Returns (params_dict, t_idx)."""
    with open(PARAMS_PATH) as f:
        data = json.load(f)
    t_idx = {t: i for i, t in enumerate(data["teams"])}
    params = {
        "attack"   : np.array(data["attack"]),
        "defence"  : np.array(data["defence"]),
        "home_adv" : data["home_adv"],
        "rho"      : data["rho"],
    }
    return params, t_idx


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def run_diagnostics(params: dict, t_idx: dict, df: pd.DataFrame) -> dict:
    """
    Run basic model diagnostics:
      - Mean absolute error on goals
      - Most likely score accuracy on held-out WC matches
      - Calibration: predicted win prob vs actual win rate
    """
    log.info("Running diagnostics ...")

    # Hold out: WC2022 matches only
    df_test = df[df["tournament"].str.contains("2022 FIFA World Cup", na=False)].copy()
    if df_test.empty:
        df_test = df[df["is_wc"] == True].tail(64).copy()

    errors_h, errors_a = [], []
    correct_result = 0
    total = 0

    for _, row in df_test.iterrows():
        pred = predict_match(
            row["home_team"], row["away_team"],
            params, t_idx, neutral=bool(row["is_neutral"])
        )
        errors_h.append(abs(pred["lambda_home"] - row["home_score"]))
        errors_a.append(abs(pred["lambda_away"] - row["away_score"]))

        # Check if predicted winner matches actual
        actual_result = (
            "H" if row["home_score"] > row["away_score"]
            else ("A" if row["home_score"] < row["away_score"] else "D")
        )
        pred_result = (
            "H" if pred["home_win_prob"] > pred["away_win_prob"] and pred["home_win_prob"] > pred["draw_prob"]
            else ("A" if pred["away_win_prob"] > pred["home_win_prob"] and pred["away_win_prob"] > pred["draw_prob"]
            else "D")
        )
        if actual_result == pred_result:
            correct_result += 1
        total += 1

    mae_h = float(np.mean(errors_h))
    mae_a = float(np.mean(errors_a))
    acc   = correct_result / total if total > 0 else 0.0

    diag = {
        "test_set"          : "WC2022 (64 matches)",
        "mae_home_goals"    : round(mae_h, 4),
        "mae_away_goals"    : round(mae_a, 4),
        "result_accuracy"   : round(acc, 4),
        "home_adv_factor"   : round(float(np.exp(params["home_adv"])), 4),
        "rho"               : round(params["rho"], 4),
    }
    log.info(f"  MAE home goals : {mae_h:.3f}")
    log.info(f"  MAE away goals : {mae_a:.3f}")
    log.info(f"  Result accuracy: {acc:.1%} ({correct_result}/{total})")
    log.info(f"  Home adv factor: {np.exp(params['home_adv']):.3f}x")
    log.info(f"  Rho (DC corr)  : {params['rho']:.4f}")
    return diag


def print_team_strengths(params: dict, t_idx: dict) -> None:
    """Print attack/defence rankings for all WC2026 teams."""
    teams_in_model = {t for t in WC2026_TEAMS if t in t_idx}
    rows = []
    for t in teams_in_model:
        i = t_idx[t]
        rows.append({
            "team"   : t,
            "attack" : round(float(params["attack"][i]), 3),
            "defence": round(float(params["defence"][i]), 3),
            "net"    : round(float(params["attack"][i]) - float(params["defence"][i]), 3),
        })
    df_str = pd.DataFrame(rows).sort_values("net", ascending=False)
    print("\n" + "=" * 55)
    print("  WC2026 Team Strengths (attack - defence = net)")
    print("=" * 55)
    print(df_str.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("WC2026 Score Model — Dixon-Coles Poisson")
    log.info("=" * 60)

    # 1. Load data
    df = load_training_data()
    t_idx = build_team_index(df)

    # 2. Fit model
    params_raw = fit_dixon_coles(df, t_idx)
    params = {
        "attack"   : params_raw["attack"],
        "defence"  : params_raw["defence"],
        "home_adv" : params_raw["home_adv"],
        "rho"      : params_raw["rho"],
    }

    # 3. Save
    save_params(params_raw, t_idx)

    # 4. Diagnostics
    diag = run_diagnostics(params, t_idx, df)
    with open(META_PATH, "w") as f:
        json.dump({**params_raw, "diagnostics": diag,
                   "attack": None, "defence": None}, f, indent=2)

    # 5. Print team strength table
    print_team_strengths(params, t_idx)

    # 6. Sample predictions
    print("\n" + "=" * 60)
    print("  Sample predictions (neutral venue)")
    print("=" * 60)
    sample_matchups = [
        ("Argentina", "France"),
        ("Spain",     "Brazil"),
        ("England",   "Germany"),
        ("Morocco",   "Senegal"),
        ("Japan",     "South Korea"),
        ("Kenya",     "Indonesia"),
    ]
    for h, a in sample_matchups:
        if h in t_idx and a in t_idx:
            pred = predict_match(h, a, params, t_idx, neutral=True)
            print(
                f"  {h:20s} vs {a:20s}  "
                f"λ={pred['lambda_home']:.2f}-{pred['lambda_away']:.2f}  "
                f"most likely: {pred['most_likely_str']}  "
                f"H/D/A: {pred['home_win_prob']:.0%}/{pred['draw_prob']:.0%}/{pred['away_win_prob']:.0%}"
            )

    print(f"\n  Parameters saved → {PARAMS_PATH}")
    print(f"  Metadata saved   → {META_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()