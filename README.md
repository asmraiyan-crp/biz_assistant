BusinessAI_RiskWise_Prototype_With_UI

This package includes:
- ai_service.py : FastAPI microservice with endpoints for forecast, simulate, risk, recommend, summary, query
- streamlit_app.py : Streamlit dashboard to upload CSVs, run AI endpoints locally, and visualize results
- ai_modules/* : forecasting, simulation, risk, sentiment, rag, llm, explain helpers (safe fallbacks)
- data/* : sample CSVs

Run backend:
    uvicorn ai_service:app --reload --port 5000

Run UI (in another terminal):
    streamlit run streamlit_app.py --server.port 8501

If you want LLM summaries, set OPENAI_API_KEY environment variable.
