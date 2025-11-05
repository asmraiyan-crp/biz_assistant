import uvicorn
import pandas as pd
import numpy as np
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from datetime import datetime, timedelta

# Import all AI modules
from ai_modules.forecast import ForecastModel
from ai_modules.simulation import monte_carlo_simulation
from ai_modules.risk import stockout_prob
from ai_modules.sentiment import Sentiment
from ai_modules.rag import SimpleRAG
# --- FIX: Import both LLM functions ---
from ai_modules.llm import llm_summary, llm_chat
# --- END FIX ---
from ai_modules.explain import explain_simple

# --- FastAPI App & Pydantic Models ---

app = FastAPI(title='Business AI Assistant - Prototype (Advanced)')


class SalesRow(BaseModel):
    date: str
    product_name: str
    quantity_sold: int
    stock_quantity: int
    avg_temp_c: float = 15.0
    rainfall_mm: float = 0.0


class ReviewRow(BaseModel):
    product_name: str
    review_text: str


class ForecastRequest(BaseModel):
    sales: List[SalesRow]
    reviews: List[ReviewRow]
    scenario: Dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    reviews: List[ReviewRow]
    query: str


# --- Helper Functions (no change) ---

def _build_sales_df(sales_rows: List[SalesRow]) -> pd.DataFrame:
    if not sales_rows:
        raise ValueError("Sales data is empty")
    df = pd.DataFrame([r.dict() for r in sales_rows])
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    return df


def _build_reviews_df(review_rows: List[ReviewRow]) -> pd.DataFrame:
    if not review_rows:
        return pd.DataFrame(columns=['product_name', 'review_text'])
    return pd.DataFrame([r.dict() for r in review_rows])


def _load_holidays() -> pd.DataFrame:
    try:
        holidays_df = pd.read_csv('data/holidays_bd.csv', parse_dates=['ds'])
        print("Successfully loaded 'data/holidays_bd.csv'")
        return holidays_df
    except FileNotFoundError:
        print("Warning: 'data/holidays_bd.csv' not found. Continuing without holiday features.")
        return None
    except Exception as e:
        print(f"Error loading holidays: {e}")
        return None


# --- Main Summary Endpoint (no change) ---

@app.post("/summary")
async def get_full_summary(req: ForecastRequest):
    """
    Main endpoint that runs the full pipeline with advanced features
    and returns a 7-day cumulative forecast.
    """
    try:
        # --- 1. Load & Prep Data ---
        sales_df = _build_sales_df(req.sales)
        reviews_df = _build_reviews_df(req.reviews)
        holidays_df = _load_holidays()
        products = sales_df['product_name'].unique().tolist()

        # --- 2. Train Models ---
        model = ForecastModel()

        if holidays_df is not None:
            holiday_dates = set(holidays_df['ds'].dt.date)
            sales_df['is_holiday'] = sales_df['date'].dt.date.isin(holiday_dates).astype(int)
        else:
            sales_df['is_holiday'] = 0

        model.train(sales_df, holidays_df=holidays_df)

        sentiment_model = Sentiment()

        rag_docs = [{'id': str(i), 'text': r['review_text'], 'meta': {'product': r['product_name']}}
                    for i, r in reviews_df.iterrows()]
        rag_model = SimpleRAG(documents=rag_docs)
        rag_model.index()

        # --- 3. Run 7-Day Risk Analysis ---
        current_date = sales_df['date'].max().date()
        print(f"Forecasting 7 days starting from: {current_date + timedelta(days=1)}")

        black_swan = req.scenario.get('black_swan')

        product_insights = []
        all_evidence = []

        stock_levels = {}
        last_weather = {}
        for p in products:
            current_rows = sales_df[sales_df['product_name'] == p].sort_values(by='date')
            if not current_rows.empty:
                stock_levels[p] = int(current_rows['stock_quantity'].iloc[-1])
                last_weather[p] = {
                    'avg_temp_c': current_rows['avg_temp_c'].iloc[-1],
                    'rainfall_mm': current_rows['rainfall_mm'].iloc[-1]
                }
            else:
                stock_levels[p] = 0
                last_weather[p] = {'avg_temp_c': 15.0, 'rainfall_mm': 0.0}

        for p in products:
            daily_risk_forecast = []
            product_cumulative_demand = 0.0
            product_initial_stock = stock_levels.get(p, 0)
            effective_stock = product_initial_stock

            for day in range(1, 8):
                date_to_forecast = current_date + timedelta(days=day)
                date_str = date_to_forecast.isoformat()

                future_external_factors = {
                    'avg_temp_c': last_weather[p]['avg_temp_c'],
                    'rainfall_mm': last_weather[p]['rainfall_mm'],
                    'is_holiday': 1 if (holidays_df is not None and date_to_forecast in holiday_dates) else 0
                }

                fc = model.predict_with_confidence(p, date_str, external_factors=future_external_factors)
                sim = monte_carlo_simulation(fc.get('mean', 0.0), variability=0.25, runs=2000, black_swan=black_swan)
                p_stockout = stockout_prob(sim, effective_stock)

                daily_demand = sim.get('mean', 0.0)
                projected_stock_end = max(0, effective_stock - daily_demand)

                daily_risk_forecast.append({
                    'product': p,
                    'day': day,
                    'date': date_str,
                    'forecast_demand': round(daily_demand, 2),
                    'projected_stock_end': round(projected_stock_end, 2),
                    'stockout_prob': p_stockout
                })

                effective_stock = projected_stock_end
                product_cumulative_demand += daily_demand

            final_p_stockout = max(d['stockout_prob'] for d in daily_risk_forecast)
            total_7_day_demand = product_cumulative_demand

            prod_reviews = reviews_df[reviews_df['product_name'] == p]['review_text']
            sentiment_score = float(prod_reviews.apply(sentiment_model.score).mean()) if not prod_reviews.empty else 0.0

            prod_insight = {
                'product_name': p,
                'forecast_mean': total_7_day_demand,
                'simulation_mean': total_7_day_demand,
                'p_stockout': final_p_stockout,
                'risk_index': final_p_stockout,
                'current_stock': product_initial_stock,
                'sentiment': sentiment_score,
                'daily_risk_forecast': daily_risk_forecast
            }

            prod_insight['reasons'] = explain_simple(prod_insight)
            product_insights.append(prod_insight)

            if sentiment_score < -0.3:
                hits = rag_model.query(f"bad reviews for {p}", top_k=1)
                all_evidence.extend([h['text'] for h in hits])

        # --- 4. Get Recommendations (no change) ---
        recs = []
        for p in product_insights:
            name = p['product_name']
            risk_idx = p['risk_index']
            mean_demand = p['simulation_mean']
            cur_stock = p['current_stock']
            qty_needed = max(0, int(round(mean_demand - cur_stock)))

            if risk_idx >= 0.60:
                recs.append({'product': name, 'action': 'reorder', 'quantity': qty_needed, 'priority': 'high',
                             'reason': f'High Risk ({risk_idx:.0%})'})
            elif risk_idx >= 0.20:
                recs.append({'product': name, 'action': 'reorder', 'quantity': qty_needed, 'priority': 'medium',
                             'reason': f'Medium Risk ({risk_idx:.0%})'})
            else:
                recs.append({'product': name, 'action': 'no_action', 'quantity': 0, 'priority': 'low',
                             'reason': f'Low Risk ({risk_idx:.0%})'})

        # --- 5. Generate LLM Summary (no change) ---
        summary = llm_summary({'products': product_insights}, all_evidence)

        # --- 6. Return Combined Response (no change) ---
        return {
            'status': 'ok',
            'summary': summary,
            'recommendations': recs,
            'risk_analysis': product_insights,
            'model_accuracy': model.accuracy
        }

    except Exception as e:
        print("--- ERROR IN /summary ---")
        traceback.print_exc()
        print("-------------------------")
        raise HTTPException(status_code=500, detail=f"Error in /summary: {str(e)}")


# --- *** UPGRADED CHATBOT ENDPOINT *** ---
@app.post('/query')
def query(req: QueryRequest):
    """
    Endpoint to query the review documents using RAG.
    It now finds context and passes it to the LLM for a natural answer.
    """
    try:
        reviews_df = _build_reviews_df(req.reviews)
        if reviews_df.empty:
            return {'status': 'ok', 'answer': "Please upload a reviews file to chat about it."}

        # 1. Build the RAG index
        rag_docs = [{'id': str(i), 'text': r['review_text'], 'meta': {'product': r['product_name']}}
                    for i, r in reviews_df.iterrows()]

        rag_model = SimpleRAG(documents=rag_docs)
        rag_model.index()

        # 2. Retrieve relevant documents (context)
        hits = rag_model.query(req.query, top_k=5)  # Get top 5 snippets
        context_snippets = [f"Review for {h['meta'].get('product', 'N/A')}: \"{h['text']}\"" for h in hits]

        # 3. Generate an answer using the LLM
        answer = llm_chat(req.query, context_snippets)

        # 4. Return the final answer
        return {'status': 'ok', 'answer': answer, 'context': context_snippets}

    except Exception as e:
        print(f"--- ERROR IN /query ---")
        traceback.print_exc()
        print("-------------------------")
        raise HTTPException(status_code=500, detail=f"Error in /query: {str(e)}")


# --- Main execution (no change) ---
if __name__ == "__main__":
    print("Starting AI Service on http://127.0.0.1:5000")
    uvicorn.run("ai_service:app", host="127.0.0.1", port=5000, reload=True)