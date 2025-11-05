import numpy as np


def monte_carlo_simulation(mean: float,
                           variability: float = 0.25,
                           runs: int = 1000,
                           black_swan: dict | None = None) -> dict:
    """
    Perform Monte Carlo simulation with optional Black Swan event injection.

    Args:
        mean (float): Expected average outcome (e.g., forecasted sales or profit).
        variability (float): Standard deviation as a fraction of mean (default=0.25).
        runs (int): Number of simulation iterations.
        black_swan (dict): Optional rare-event config, e.g.
            {'prob': 0.05, 'multiplier': 0.3} meaning 5% of cases drop to 30% value.

    Returns:
        dict: Contains mean, std, percentiles, Value at Risk (VaR), Conditional VaR (CVaR).
    """
    if mean < 0 or runs <= 0:
        # Return 0s if mean is negative (e.g., profit loss) or runs are 0
        return {'mean': 0.0, 'std': 0.0, 'p10': 0.0, 'p90': 0.0, 'VaR90': 0.0, 'CVaR90': 0.0}

    # Generate base simulation outcomes
    # Use abs(mean) to ensure scale is positive
    scale = max(1.0, abs(mean) * variability)
    preds = np.random.normal(loc=mean, scale=scale, size=runs)

    # Inject rare "Black Swan" disruptions
    if black_swan and black_swan.get('prob', 0) > 0:
        n_swans = int(runs * black_swan['prob'])
        if n_swans > 0:
            idx = np.random.choice(runs, size=n_swans, replace=False)
            multiplier = black_swan.get('multiplier', 0.2)
            preds[idx] *= multiplier

    # Filter out invalid (negative) predictions
    preds = preds[preds >= 0]
    if len(preds) == 0:
        return {'mean': 0.0, 'std': 0.0, 'p10': 0.0, 'p90': 0.0, 'VaR90': 0.0, 'CVaR90': 0.0}

    # Compute risk statistics
    mean_val = float(np.mean(preds))
    std_val = float(np.std(preds))
    p10 = float(np.percentile(preds, 10))
    p90 = float(np.percentile(preds, 90))
    var90 = p10  # 10th percentile = 90% Value-at-Risk

    # More robust CVaR calculation
    cvar90 = float(np.mean(preds[preds <= p10])) if np.any(preds <= p10) else p10

    return {
        'mean': mean_val,
        'std': std_val,
        'p10': p10,
        'p90': p90,
        'VaR90': var90,
        'CVaR90': cvar90
    }
