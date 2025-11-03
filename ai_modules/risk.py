try:
    from scipy.stats import norm

    SCIPY = True
except Exception:
    SCIPY = False
    from math import erf, sqrt


def stockout_prob(sim_results, current_stock):
    """
    Calculates the probability of stocking out (demand > stock)
    given simulation results and current stock level, assuming a Normal distribution.
    """
    mean = sim_results.get('mean', 0.0)
    std = sim_results.get('std', 0.0)

    if std <= 0:
        # No variability, probability is 1.0 if mean demand > stock, 0.0 otherwise
        return 1.0 if mean > current_stock else 0.0

    if SCIPY:
        # Use SciPy's Normal Distribution Cumulative Dstribution Function (CDF)
        # cdf = P(demand <= current_stock)
        cdf = norm.cdf(current_stock, loc=mean, scale=std)
    else:
        # Fallback to math.erf if scipy is not installed
        z = (current_stock - mean) / (std * sqrt(2))
        cdf = 0.5 * (1 + erf(z))

    # Stockout probability = 1 - P(demand <= current_stock)
    prob = 1.0 - cdf

    return max(0.0, min(1.0, float(prob)))