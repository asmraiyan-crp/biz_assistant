import streamlit as st
import pandas as pd
import requests
import json
import traceback
import numpy as np

# --- Page Config ---
st.set_page_config(page_title='AI Business Assistant', layout='wide')
st.title('🤖 AI Business Assistant - Prototype')

# --- Sidebar (Inputs) ---
with st.sidebar:
    st.header('Upload Data')
    sales_file = st.file_uploader('Upload Sales CSV', type=['csv'])
    reviews_file = st.file_uploader('Upload Reviews CSV', type=['csv'])

    api_url = st.text_input('AI Service URL', 'http://localhost:5000')

    run_button = st.button('Run 7-Day Forecast', type="primary", use_container_width=True)

# --- Main Page (Outputs) ---

if not sales_file or not reviews_file:
    st.info('Please upload both Sales and Reviews CSV files to begin.')
    # Show data tables on upload (before running)
    if sales_file:
        st.subheader("Uploaded Sales Data (Sample)")
        st.dataframe(pd.read_csv(sales_file, parse_dates=['date']).head(), use_container_width=True)
    if reviews_file:
        st.subheader("Uploaded Reviews Data (Sample)")
        st.dataframe(pd.read_csv(reviews_file).head(), use_container_width=True)

if run_button and sales_file and reviews_file:
    with st.spinner('Running full 7-day analysis... This may take a moment.'):
        try:
            # --- 1. Read and Prepare Data ---
            sales_file.seek(0)
            reviews_file.seek(0)

            sales_df = pd.read_csv(sales_file)
            reviews_df = pd.read_csv(reviews_file)

            sales_df['date'] = pd.to_datetime(sales_df['date'], errors='coerce')

            sales_df = sales_df.replace({np.nan: None})
            reviews_df = reviews_df.replace({np.nan: None})

            if 'date' in sales_df.columns:
                sales_df['date'] = sales_df['date'].astype(str)

            sales_payload = sales_df.to_dict(orient='records')
            reviews_payload = reviews_df.to_dict(orient='records')

            payload = {
                "sales": sales_payload,
                "reviews": reviews_payload,
                "scenario": {}
            }

            # --- 2. Call AI Service ---
            resp = requests.post(
                f"{api_url}/summary",
                json=payload,
                timeout=180  # 3 minute timeout
            )

            if resp.status_code == 200:
                data = resp.json()
                st.success('Analysis Complete!')

                # --- 3. Display Results ---

                # AI Summary
                st.subheader('📈 AI-Generated 7-Day Summary')
                st.markdown(data.get('summary', 'No summary generated.'))

                # Reorder Recommendations
                st.subheader('📦 7-Day Reorder Recommendations')
                recs_df = pd.DataFrame(data.get('recommendations', []))
                if not recs_df.empty:
                    st.dataframe(recs_df, use_container_width=True)
                else:
                    st.info("No reorder recommendations at this time.")

                # Full Risk Analysis
                st.subheader('⚠️ Product 7-Day Risk & Sentiment')
                risk_df = pd.DataFrame(data.get('risk_analysis', []))

                if not risk_df.empty:
                    risk_df_display = risk_df.copy()
                    risk_df_display['risk_index'] = risk_df_display['risk_index'].map('{:.1%}'.format)
                    risk_df_display['sentiment'] = risk_df_display['sentiment'].map('{:.2f}'.format)
                    risk_df_display['forecast_mean'] = risk_df_display['forecast_mean'].map('{:.0f}'.format)
                    risk_df_display['current_stock'] = risk_df_display['current_stock'].map('{:.0f}'.format)

                    st.dataframe(risk_df_display[[
                        'product_name',
                        'risk_index',  # Max 7-day risk
                        'sentiment',
                        'forecast_mean',  # 7-day total demand
                        'current_stock',
                        'reasons'
                    ]], use_container_width=True)

                    # --- 7-DAY FORECAST SECTION ---
                    st.subheader("🗓️ 7-Day Daily Stockout Forecast")

                    product_list = risk_df['product_name'].tolist()
                    selected_product = st.selectbox("Select product to see daily forecast:", product_list)

                    if selected_product:
                        product_data = next((item for item in data.get('risk_analysis', []) if
                                             item["product_name"] == selected_product), None)

                        if product_data and 'daily_risk_forecast' in product_data:
                            daily_df = pd.DataFrame(product_data['daily_risk_forecast'])

                            daily_df_display = daily_df.copy()
                            daily_df_display['stockout_prob_pct'] = (daily_df['stockout_prob'] * 100).round(1).astype(
                                str) + '%'
                            daily_df_display['forecast_demand'] = daily_df_display['forecast_demand'].map(
                                '{:.1f}'.format)
                            daily_df_display['projected_stock_end'] = daily_df_display['projected_stock_end'].map(
                                '{:.1f}'.format)

                            st.dataframe(daily_df_display[[
                                'day',
                                'date',
                                'forecast_demand',
                                'projected_stock_end',
                                'stockout_prob_pct'
                            ]], use_container_width=True)

                # Raw JSON
                with st.expander("Show Raw JSON Response"):
                    st.json(data)

            else:
                st.error(f"Error from AI Service (HTTP {resp.status_code}):")
                st.json(resp.json())

        except requests.exceptions.ConnectionError:
            st.error(f"Connection Error: Could not connect to the AI service at {api_url}. Is the backend running?")
        except Exception as e:
            st.error(f"An error occurred in the Streamlit app: {e}")
            traceback.print_exc()

# --- *** UPGRADED CHATBOT UI *** ---
st.markdown("---")
st.subheader("💬 AI Co-pilot (Chat about Reviews)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask me anything about your customer reviews!"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What are the main complaints about headphones?"):
    if not reviews_file:
        st.error("Please upload a Reviews CSV file first to chat about it.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Re-read file for a fresh state
                    reviews_file.seek(0)
                    reviews_df = pd.read_csv(reviews_file)
                    reviews_df = reviews_df.replace({np.nan: None})
                    reviews_payload = reviews_df.to_dict(orient='records')

                    payload = {
                        "query": prompt,
                        "reviews": reviews_payload
                    }

                    resp = requests.post(
                        f"{api_url}/query",
                        json=payload,
                        timeout=60
                    )

                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data.get("answer", "I'm sorry, I couldn't get a valid answer.")
                        st.markdown(answer)
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": answer})

                        # Optionally show the context used
                        with st.expander("Show context used"):
                            st.json(data.get("context", []))
                    else:
                        st.error(f"Error from AI Service (HTTP {resp.status_code}):")
                        st.json(resp.json())

                except Exception as e:
                    st.error(f"An error occurred during query: {e}")
                    traceback.print_exc()