import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score

# Suppress noisy warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from prophet import Prophet
    PROPHET = True
except Exception:
    PROPHET = False
    print("Warning: Prophet library not found. Falling back to LGBM-only mode.")
    print("To install Prophet, run: pip install prophet")


class ForecastModel:
    """
    ForecastModel using Prophet (if available) + LightGBM for engineered features.
    Improvements:
      - Safe feature engineering
      - Does not mutate self.features permanently
      - Stores last-feature-row per product for LGBM fallback predictions
    """

    def __init__(self):
        self.model = lgb.LGBMRegressor(
            random_state=42,
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            n_jobs=-1
        )
        self.prophet_models = {}            # product -> Prophet model
        self.trained_products = set()
        self.holidays_df = None
        # Base features (do not mutate this list)
        self.features_base = [
            'day_of_week', 'month', 'is_holiday',
            'avg_temp_c', 'rainfall_mm',
            'sales_lag_7', 'sales_lag_14', 'sales_lag_30',
            'sales_rolling_mean_7', 'sales_rolling_mean_30'
        ]
        self.target = 'quantity_sold'
        self.accuracy = {'mae': None, 'r2': None, 'model_type': 'LGBM', 'features_used': list(self.features_base)}
        # Saved last feature row per product (used for LGBM fallback prediction)
        self._last_feature_row = {}

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Ensure date
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])

        # Ensure no duplicate aggregation collisions; take last if duplicates
        df = df.groupby(['date', 'product_name']).last().reset_index()

        # Build full date scaffolds per product
        scaffold = []
        for product in df['product_name'].unique():
            prod = df[df['product_name'] == product]
            min_date, max_date = prod['date'].min(), prod['date'].max()
            all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
            scaffold.append(pd.DataFrame({'date': all_dates, 'product_name': product}))

        if not scaffold:
            return pd.DataFrame(columns=self.features_base + [self.target, 'product_name', 'date'])

        full_range_df = pd.concat(scaffold, ignore_index=True)
        df = pd.merge(full_range_df, df, on=['date', 'product_name'], how='left')

        # Fill/Interpolate external features safely
        for col, fill_val in [('is_holiday', 0), ('avg_temp_c', 15.0), ('rainfall_mm', 0.0)]:
            if col not in df.columns:
                df[col] = fill_val

        df['is_holiday'] = df['is_holiday'].ffill(limit=3).fillna(0)
        df['avg_temp_c'] = df['avg_temp_c'].interpolate(method='linear').fillna(method='bfill').fillna(15.0)
        df['rainfall_mm'] = df['rainfall_mm'].interpolate(method='linear').fillna(method='bfill').fillna(0.0)

        # Fill missing sales with 0
        if 'quantity_sold' not in df.columns:
            df['quantity_sold'] = 0
        df['quantity_sold'] = df['quantity_sold'].fillna(0)

        # Create grouped shifts and rolling means
        grouped = df.groupby('product_name')
        df['sales_lag_7'] = grouped['quantity_sold'].shift(7).fillna(0)
        df['sales_lag_14'] = grouped['quantity_sold'].shift(14).fillna(0)
        df['sales_lag_30'] = grouped['quantity_sold'].shift(30).fillna(0)
        df['sales_rolling_mean_7'] = grouped['quantity_sold'].shift(1).rolling(window=7, min_periods=1).mean().fillna(0)
        df['sales_rolling_mean_30'] = grouped['quantity_sold'].shift(1).rolling(window=30, min_periods=1).mean().fillna(0)

        df = df.reset_index(drop=True)

        # Date features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month

        # Ensure we return rows that have the required features (drop rows that still have NaNs in required features)
        df_out = df.dropna(subset=self.features_base).copy()
        # Save last known feature vector per product to support fallback prediction
        for product in df_out['product_name'].unique():
            prod_rows = df_out[df_out['product_name'] == product].sort_values('date')
            last_row = prod_rows.iloc[-1]
            feat_row = {f: last_row.get(f) for f in self.features_base}
            # Also include product_name for categorical encoding if needed
            feat_row['product_name'] = product
            self._last_feature_row[product] = feat_row

        return df_out

    def train(self, df: pd.DataFrame, holidays_df: pd.DataFrame = None):
        """
        Train Prophet models (if available) per product and prepare LGBM features.
        """
        if holidays_df is not None:
            self.holidays_df = holidays_df

        if df is None or df.empty:
            raise ValueError("Sales dataframe is empty")

        all_features_df = []
        # Per-product training prep (Prophet models)
        for product in df['product_name'].unique():
            prod_df = df[df['product_name'] == product].copy()
            if len(prod_df) < 30:
                print(f"Skipping {product}: not enough data (need 30+ days for lag features).")
                continue

            # Prophet training if available
            if PROPHET:
                try:
                    prophet_df = prod_df[['date', 'quantity_sold', 'avg_temp_c', 'rainfall_mm', 'is_holiday']].rename(
                        columns={'date': 'ds', 'quantity_sold': 'y'}
                    )
                    # Ensure regressor columns exist
                    for c in ['avg_temp_c', 'rainfall_mm', 'is_holiday']:
                        if c not in prophet_df.columns:
                            prophet_df[c] = 0.0

                    m = Prophet(holidays=self.holidays_df)
                    # Only add regressors if present in data
                    m.add_regressor('avg_temp_c')
                    m.add_regressor('rainfall_mm')
                    m.add_regressor('is_holiday')

                    # Fit prophet on available columns
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m.fit(prophet_df)
                    self.prophet_models[product] = m
                except Exception as e:
                    print(f"Warning: Prophet model failed for '{product}': {e}")

            # LGBM feature engineering (for global LGBM)
            feats = self._engineer_features(prod_df)
            if not feats.empty:
                feats['product_name'] = product  # ensure product_name present
                all_features_df.append(feats)
                self.trained_products.add(product)

        if not all_features_df:
            raise ValueError("No products had sufficient data to train the model.")

        full_feature_df = pd.concat(all_features_df, ignore_index=True)

        # Make a local copy of features list (don't mutate base)
        features = list(self.features_base) + ['product_name']

        # Prepare X and y
        X = full_feature_df[features]
        y = full_feature_df[self.target]

        # Train LGBM using time-series CV
        ts_cv = TimeSeriesSplit(n_splits=5)
        mae_scores = []
        r2_scores = []
        for train_idx, test_idx in ts_cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            if len(X_test) == 0:
                continue
            # Convert categorical product_name to category dtype for LightGBM
            X_train = X_train.copy()
            X_test = X_test.copy()
            X_train['product_name'] = X_train['product_name'].astype('category')
            X_test['product_name'] = X_test['product_name'].astype('category')

            self.model.fit(X_train, y_train, categorical_feature=['product_name'])
            preds = self.model.predict(X_test)
            preds = np.maximum(0, preds)
            mae_scores.append(mean_absolute_error(y_test, preds))
            r2_scores.append(r2_score(y_test, preds))

        if mae_scores:
            self.accuracy['mae'] = float(np.mean(mae_scores))
            self.accuracy['r2'] = float(np.mean(r2_scores))
            self.accuracy['features_used'] = features
        else:
            # fallback: train on all data
            X['product_name'] = X['product_name'].astype('category')
            self.model.fit(X, y, categorical_feature=['product_name'])

        print(f"LGBM Model trained. Accuracy: MAE={self.accuracy.get('mae')} R2={self.accuracy.get('r2')}")
        return {'aggregate': self.accuracy}

    def predict_with_confidence(self, product_name: str, date, external_factors: dict = None):
        """
        Predict using Prophet if available for the product, otherwise fallback to LGBM estimate
        using the last saved feature row for that product.
        Returns dict with 'mean' and 'std' at minimum.
        """
        external_factors = external_factors or {}

        # If Prophet model exists for this product, use it
        if PROPHET and product_name in self.prophet_models:
            try:
                future_df = pd.DataFrame([{
                    'ds': pd.to_datetime(date),
                    'avg_temp_c': external_factors.get('avg_temp_c', 15.0),
                    'rainfall_mm': external_factors.get('rainfall_mm', 0.0),
                    'is_holiday': external_factors.get('is_holiday', 0)
                }])
                fc = self.prophet_models[product_name].predict(future_df)
                mean_pred = float(fc['yhat'].iloc[0])
                # derive std from CI width if available
                try:
                    std_dev = (float(fc['yhat_upper'].iloc[0]) - float(fc['yhat_lower'].iloc[0])) / 3.92
                except Exception:
                    std_dev = max(0.1, 0.1 * abs(mean_pred))
                mean_pred = max(0.0, mean_pred)
                std_dev = max(0.0, std_dev)
                return {'mean': mean_pred, 'std': std_dev, 'model': 'Prophet'}
            except Exception as e:
                print(f"Warning: Prophet prediction failed for {product_name}: {e}")

        # Otherwise fallback to LGBM-based estimation using the last known feature row
        if product_name in self._last_feature_row:
            # Build input vector from last row and override with any external_factors
            feat = self._last_feature_row[product_name].copy()
            # Merge external factors into the feature vector where appropriate
            for k in ['avg_temp_c', 'rainfall_mm', 'is_holiday']:
                if k in external_factors:
                    feat[k] = external_factors[k]

            # Assemble DataFrame row
            features = list(self.features_base) + ['product_name']
            X_new = pd.DataFrame([ {f: feat.get(f, 0) for f in features} ])
            # Ensure categorical dtype
            X_new['product_name'] = X_new['product_name'].astype('category')
            try:
                pred = float(self.model.predict(X_new)[0])
                # Estimate std from MAE if present
                std_est = (self.accuracy['mae'] * 1.96) if self.accuracy.get('mae') is not None else max(1.0, 0.15*abs(pred))
                pred = max(0.0, pred)
                std_est = max(0.0, float(std_est))
                return {'mean': pred, 'std': std_est, 'model': 'LGBM_fallback'}
            except Exception as e:
                print(f"Warning: LGBM fallback prediction failed for {product_name}: {e}")

        # Final last-resort fallback
        return {'mean': 0.0, 'std': 1.0, 'model': 'fallback'}
