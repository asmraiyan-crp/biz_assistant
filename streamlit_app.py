import streamlit as st
import pandas as pd
import requests
import json
import io

st.set_page_config(page_title='AI Business Assistant (Prototype)', layout='wide')
st.title('🤖 AI Business Assistant - Prototype (Risk-style)')

# --- Sidebar for Inputs ---
st.sidebar.header('Upload Data')
sales_file = st.sidebar.file_uploader('Upload Sales CSV', type=['csv'])
reviews_file = st.sidebar.file_uploader('Upload Reviews CSV', type=['csv'])

api_url = st.sidebar.text_input('AI Service URL', 'http://localhost:5000')

st.sidebar.markdown("---")
# Use a more descriptive button label
run_button = st.sidebar.button('Run Full Analysis', type="primary")

# --- Main Page Layout ---
if sales_file is None or reviews_file is None:
    st.info('Please upload both Sales and Reviews CSV files to begin.')
else:
    # Load data for display and payload
    try:
        # Read files into memory
        sales_data_bytes = sales_file.getvalue()
        reviews_data_bytes = reviews_file.getvalue()

        # Read into DataFrames
        sales_df = pd.read_csv(io.BytesIO(sales_data_bytes), parse_dates=['date'])
        reviews_df = pd.read_csv(io.BytesIO(reviews_data_bytes))

        st.subheader('Uploaded Sales Data (Sample)')
        st.dataframe(sales_df.head())

        st.subheader('Uploaded Reviews Data (Sample)')
        st.dataframe(reviews_df.head())

        if run_button:
            # Build the payload
            products = sales_df['product_name'].unique().tolist()

            # --- THIS IS THE FIX ---
            # Create a copy to avoid changing the displayed dataframe
            sales_df_payload = sales_df.copy()
            # Convert Timestamp objects to standard strings BEFORE sending
            sales_df_payload['date'] = sales_df_payload['date'].astype(str)
            # --- END FIX ---

            sales_payload = sales_df_payload.to_dict(orient='records')
            reviews_payload = reviews_df.to_dict(orient='records')

            payload = {
                'sales': sales_payload,
                'reviews': reviews_payload,
                'products': products
            }

            with st.spinner('Calling AI Service... This may take a moment (training models)...'):
                try:
                    # Call the /summary endpoint to get the full report
                    resp = requests.post(api_url + '/summary', json=payload, timeout=120)

                    if resp.status_code == 200:
                        data = resp.json()
                        st.success('Analysis Complete!')

                        # --- Display Results ---
                        st.subheader('📈 AI-Generated Summary')
                        st.markdown(data.get('summary_text', 'No summary available.'))

                        st.subheader('📦 Reorder Recommendations')
                        rec_df = pd.DataFrame(data.get('recommendations', []))
                        st.dataframe(rec_df, use_container_width=True)

                        st.subheader('⚠️ Product Risk & Sentiment Analysis')
                        risk_df = pd.DataFrame(data.get('risk_products', []))

                        # Format columns for better display
                        if not risk_df.empty:
                            risk_df_display = risk_df.copy()
                            risk_df_display['p_stockout'] = risk_df_display['p_stockout'].map('{:.1%}'.format)
                            risk_df_display['risk_index'] = risk_df_display['risk_index'].map('{:.1%}'.format)
                            risk_df_display['sentiment'] = risk_df_display['sentiment'].map('{:.2f}'.format)
                            risk_df_display['forecast_mean'] = risk_df_display['forecast_mean'].map('{:.1f}'.format)
                            risk_df_display['simulation_mean'] = risk_df_display['simulation_mean'].map('{:.1f}'.format)

                            st.dataframe(risk_df_display[[
                                'product_name',
                                'risk_index',
                                'p_stockout',
                                'sentiment',
                                'forecast_mean',
                                'current_stock',
                                'reasons'
                            ]], use_container_width=True)

                        with st.expander("Show Raw JSON Response"):
                            st.json(data)

                    else:
                        st.error(f"Error from AI Service (HTTP {resp.status_code}):")
                        try:
                            st.json(resp.json())
                        except:
                            st.text(resp.text)

                except requests.exceptions.RequestException as e:
                    st.error(f'AI service connection error: {e}')
                except Exception as e:
                    st.error(f'An unexpected error occurred: {e}')

    except Exception as e:
        # This will catch the 'Timestamp' error if it happens during file read
        st.error(f"Error reading or processing files: {e}")

st.markdown('---')
st.subheader('Ad-hoc Query')
query_text = st.text_input('Ask about product reviews:')

if st.button('Run Query'):
    if reviews_file is None:
        st.error('Upload reviews CSV first')
    else:
        # We don't need to convert dates for the query payload
        reviews_payload = reviews_df.to_dict(orient='records')
        payload = {'query': query_text, 'reviews': reviews_payload}

        with st.spinner('Querying reviews...'):
            try:
                resp = requests.post(api_url + '/query', json=payload, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success('Query complete')
                    st.write(data.get('hits', []))
                else:
                    st.error(f"Error from AI Service (HTTP {resp.status_code}):")
                    st.json(resp.json())
            except Exception as e:
                st.error(f'AI service error: {e}')

