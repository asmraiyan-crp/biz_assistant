def explain_simple(insight):
    reasons=[]
    if insight.get('risk_index',0)>0.6:
        reasons.append('High predicted demand vs current stock')
    if insight.get('sentiment',0)<-0.3:
        reasons.append('Negative customer sentiment')
    return reasons
