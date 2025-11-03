import numpy as np
def monte_carlo(mean, variability=0.25, runs=1000, black_swan=None):
    preds = np.random.normal(loc=mean, scale=max(1.0, mean*variability), size=runs)
    if black_swan and 'prob' in black_swan and black_swan['prob']>0:
        n = runs
        idx = np.random.choice(n, size=int(n*black_swan['prob']), replace=False)
        for i in idx:
            preds[i] = preds[i] * black_swan.get('multiplier', 0.2)
    preds = preds[preds>=0]
    if len(preds)==0:
        return {'mean':0.0,'std':0.0,'p10':0.0,'p90':0.0,'VaR90':0.0,'CVaR90':0.0}
    mean = float(preds.mean()); std = float(preds.std())
    p10 = float(np.percentile(preds,10)); p90 = float(np.percentile(preds,90))
    var90 = float(np.percentile(preds,10))
    threshold = np.percentile(preds,10)
    cvar = float(preds[preds<=threshold].mean()) if len(preds[preds<=threshold])>0 else threshold
    return {'mean':mean,'std':std,'p10':p10,'p90':p90,'VaR90':var90,'CVaR90':cvar}
