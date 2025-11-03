from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime, timedelta

# Import all your AI modules
from ai_modules.data_loader import load_sales_data, load_reviews
from ai_modules.forecast import ForecastModel
from ai_modules.simulation import monte_carlo
from ai_modules.risk import stockout_prob
from ai_modules.sentiment import Sentiment
from ai_modules.rag import SimpleRAG
from ai_modules.llm import llm_summary
from ai_modules.explain import explain_simple  # Import explain_simple

app = FastAPI(title='Business AI Assistant - Risk-style')


# --- Pydantic Models ---
# These define the structure of your API requests

class SalesRow(BaseModel):
    date: str
    product_name: str
    quantity_sold: int
    price: float
    total_revenue: float
    stock_quantity: int = 0


class ReviewRow(BaseModel):
    product_name: str
    review_text: str


class ForecastRequest(BaseModel):
    sales: List[SalesRow]
    reviews: List[ReviewRow] = []  # Add reviews to the main request
    products: List[str]
    scenario: Dict[str, Any] = None


class QueryRequest(BaseModel):
    query: str
    reviews: List[ReviewRow]  # Pass reviews for RAG


# --- Helper Functions ---
# (We can move logic out of endpoints to make them cleaner)

def _build_sales_df(sales_rows: List[SalesRow]) -> pd.DataFrame:
    """Converts sales rows into a processed DataFrame."""
    df = pd.DataFrame([r.dict() for r in sales_rows])
    df['date'] = pd.to_datetime(df['date'])

    # --- THIS IS THE FIX ---
    # The model's train() method requires these columns.
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['date'].dt.dayofweek
    if 'month' not in df.columns:
        df['month'] = df['date'].dt.month
    # --- END FIX ---

    return df


def _build_reviews_df(review_rows: List[ReviewRow]) -> pd.DataFrame:
    """Converts review rows into a DataFrame."""
    return pd.DataFrame([r.dict() for r in review_rows])


# --- API Endpoints ---

@app.post('/simulate')
def simulate(req: ForecastRequest):
    """
    (Internal) Trains forecast model and runs simulation.
    This is called by other endpoints.
    """
    try:
        df = _build_sales_df(req.sales)  # Use helper

        model = ForecastModel()
        metrics = model.train(df)

        out = {}
        # Predict for 7 days from now
        date_str = (datetime.utcnow() + timedelta(days=7)).date().isoformat()

        for p in req.products:
            try:
                fc = model.predict_with_confidence(p, date_str)
            except Exception:
                fc = {'mean': 0.0, 'std': 0.0}

            black = req.scenario.get('black_swan') if req.scenario else None
            sim = monte_carlo(fc.get('mean', 0.0), variability=0.25, runs=2000, black_swan=black)

            current_rows = df[df['product_name'] == p]
            cur_stock = int(current_rows['stock_quantity'].iloc[-1]) if not current_rows.empty else 0

            p_stockout = stockout_prob(sim, cur_stock)

            out[p] = {'forecast_conf': fc, 'simulation': sim, 'current_stock': cur_stock, 'p_stockout': p_stockout}

        return {'status': 'ok', 'metrics': metrics, 'results': out}

    except KeyError as e:
        # This will catch the 'day_of_week' error if it persists
        raise HTTPException(status_code=500, detail=f"Missing column in DataFrame: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/risk')
def risk(req: ForecastRequest):
    """
    (Internal) Calculates risk, sentiment, and reasons.
    This is called by other endpoints.
    """
    try:
        # 1. Get simulation results
        sim_resp = simulate(req)
        results = sim_resp['results']

        # 2. Process review data for sentiment
        reviews_df = _build_reviews_df(req.reviews)
        sentiment_model = Sentiment()

        products = []
        for p, info in results.items():
            prod_insight = {
                'product_name': p,
                'forecast_mean': info['forecast_conf'].get('mean', 0.0),
                'simulation_mean': info['simulation'].get('mean', 0.0),
                'std': info['simulation'].get('std', 0.0),
                'p_stockout': info.get('p_stockout', 0.0),
                'current_stock': info.get('current_stock', 0)
            }

            # Calculate Risk Index
            prod_insight['risk_index'] = max(0.0,
                                             (prod_insight['simulation_mean'] - prod_insight['current_stock']) / max(
                                                 1.0, prod_insight['simulation_mean']))

            # Calculate Sentiment
            prod_reviews = reviews_df[reviews_df['product_name'] == p][
                'review_text'] if not reviews_df.empty else pd.Series([])
            if not prod_reviews.empty:
                prod_insight['sentiment'] = float(prod_reviews.apply(sentiment_model.score).mean())
            else:
                prod_insight['sentiment'] = 0.0

            # Get Reasons
            prod_insight['reasons'] = explain_simple(prod_insight)  # Use explain module

            products.append(prod_insight)

        # 3. Setup RAG for evidence
        if not reviews_df.empty:
            docs = [{'id': str(i), 'text': row['review_text'], 'meta': {'product': row['product_name']}} for i, row in
                    reviews_df.iterrows()]
        else:
            docs = []

        rag = SimpleRAG(documents=docs);
        rag.index()
        evidence = []
        for prod in products:
            if prod['sentiment'] < -0.2:  # Find evidence for negative sentiment
                hits = rag.query(f"bad reviews for {prod['product_name']}", top_k=1)
                evidence += [h['text'] for h in hits]

        return {'status': 'ok', 'risk_products': products, 'evidence': evidence}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/recommend')
def recommend(req: ForecastRequest):
    """
    (Internal) Generates reorder recommendations.
    This is called by the /summary endpoint.
    """
    try:
        r = risk(req)  # Get risk analysis
        products = r['risk_products']

        recs = []
        for p in products:
            name = p['product_name']
            risk_idx = p.get('risk_index', 0.0)
            mean = p.get('simulation_mean', 0.0)
            cur_stock = p.get('current_stock', 0)

            if risk_idx >= 0.6:
                qty = max(1, int(round(mean - cur_stock)))
                recs.append({'product': name, 'action': 'reorder', 'quantity': qty, 'priority': 'high',
                             'reason': f'High Risk ({risk_idx:.0%})'})
            elif risk_idx >= 0.3:
                qty = max(0, int(round(mean - cur_stock)))
                recs.append({'product': name, 'action': 'reorder', 'quantity': qty, 'priority': 'medium',
                             'reason': f'Medium Risk ({risk_idx:.0%})'})
            else:
                recs.append({'product': name, 'action': 'no_action', 'quantity': 0, 'priority': 'low',
                             'reason': 'Stock sufficient'})

        return {'status': 'ok', 'recommendations': recs}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/summary')
def summary(req: ForecastRequest):
    """
    *** MAIN ENDPOINT ***
    Runs the full pipeline: risk, recommendations, and LLM summary.
    """
    try:
        risk_resp = risk(req)
        rec_resp = recommend(req)

        # Consolidate data for the LLM
        insights = {'products': risk_resp['risk_products']}
        evidence = risk_resp['evidence']

        summary_text = llm_summary(insights, evidence)

        return {
            'status': 'ok',
            'summary_text': summary_text,
            'recommendations': rec_resp['recommendations'],
            'risk_products': risk_resp['risk_products']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/query')
def query(req: QueryRequest):
    """
    Runs an ad-hoc RAG query against the review data.
    """
    try:
        reviews_df = _build_reviews_df(req.reviews)
        if reviews_df.empty:
            return {'status': 'ok', 'hits': ['No review data provided.']}

        docs = [{'id': str(i), 'text': row['review_text'], 'meta': {'product': row['product_name']}} for i, row in
                reviews_df.iterrows()]

        rag = SimpleRAG(documents=docs);
        rag.index()
        hits = rag.query(req.query, top_k=4)

        return {'status': 'ok', 'hits': hits}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

