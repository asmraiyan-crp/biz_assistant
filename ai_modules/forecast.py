import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

try:
    from prophet import Prophet

    PROPHET = True
except Exception:
    PROPHET = False


class ForecastModel:
    def __init__(self):
        self.rf = RandomForestRegressor(n_estimators=100, random_state=42)
        self.le = LabelEncoder()
        self.prophet_models = {}
        self.trained = False
        self.features = ['day_of_week', 'month', 'product_encoded']

    def train(self, df):
        df = df.copy().sort_values(by='date')
        df.reset_index(drop=True, inplace=True)

        if PROPHET:
            for p in df['product_name'].unique():
                prod_df = df[df['product_name'] == p][['date', 'quantity_sold']].rename(
                    columns={'date': 'ds', 'quantity_sold': 'y'})
                if len(prod_df) >= 3:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            m = Prophet()
                            m.fit(prod_df)
                        self.prophet_models[p] = m
                    except Exception as e:
                        print(f"Warning: Prophet model training failed for '{p}': {e}")
                        pass

        df['product_encoded'] = self.le.fit_transform(df['product_name'])
        X = df[self.features]  # X now has feature names
        y = df['quantity_sold']

        if len(y) < 10:
            raise ValueError('Not enough data to train model (need at least 10 records)')

        split_point = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError('Not enough data to create a train/test split.')

        self.rf.fit(X_train, y_train)  # Train with feature names
        self.trained = True
        preds = self.rf.predict(X_test)

        return {
            'mae': float(mean_absolute_error(y_test, preds)),
            'r2': float(r2_score(y_test, preds)),
            'prophet_models': len(self.prophet_models),
            'rf_features': self.features
        }

    def _get_features_from_date(self, date_str, product_name) -> pd.DataFrame:
        """
        Creates a DataFrame with correct feature names to silence sklearn warning.
        """
        d = pd.to_datetime(date_str)
        try:
            p_enc = int(self.le.transform([product_name])[0])
        except ValueError:
            raise ValueError(f"Product '{product_name}' was not seen during training.")

        # Create DataFrame with feature names
        X_new = pd.DataFrame(
            [[d.dayofweek, d.month, p_enc]],
            columns=self.features
        )
        return X_new

    def predict(self, product, date_str):
        if PROPHET and product in self.prophet_models:
            try:
                future = pd.DataFrame({'ds': [pd.to_datetime(date_str)]})
                fc = self.prophet_models[product].predict(future)
                return max(0, int(round(float(fc['yhat'].iloc[0]))))
            except Exception as e:
                print(f"Warning: Prophet prediction failed for '{product}', falling back to RF: {e}")

        if not self.trained:
            raise ValueError('Model not trained')

        try:
            X_new = self._get_features_from_date(date_str, product)
            # Suppress the warning, though it's already fixed by using a DataFrame
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                pred = self.rf.predict(X_new)[0]
            return max(0, int(round(pred)))
        except ValueError as e:
            print(f"Warning: Cannot predict for unknown product: {e}")
            return 0

    def predict_with_confidence(self, product, date_str, n=500, variability=0.25):
        if PROPHET and product in self.prophet_models:
            try:
                future = pd.DataFrame({'ds': [pd.to_datetime(date_str)]})
                fc = self.prophet_models[product].predict(future)

                y = float(fc['yhat'].iloc[0])

                # --- FIX FOR FutureWarning ---
                # Use .iloc[0] to get the scalar value from the pandas Series
                lower_val = fc.get('yhat_lower', y * 0.85)
                upper_val = fc.get('yhat_upper', y * 1.15)

                lower = float(lower_val.iloc[0]) if isinstance(lower_val, pd.Series) else float(lower_val)
                upper = float(upper_val.iloc[0]) if isinstance(upper_val, pd.Series) else float(upper_val)
                # --- END FIX ---

                y = max(0, y)
                lower = max(0, lower)
                upper = max(0, upper)
                return {'mean': y, 'lower': lower, 'upper': upper, 'std': (upper - lower) / 3.92}
            except Exception as e:
                print(f"Warning: Prophet confidence prediction failed, falling back to RF: {e}")

        point = self.predict(product, date_str)
        samples = np.random.normal(loc=point, scale=max(1.0, point * variability), size=n)
        samples = samples[samples >= 0]
        if len(samples) == 0:
            return {'mean': 0.0, 'std': 0.0, 'p10': 0.0, 'p90': 0.0}

        return {
            'mean': float(samples.mean()),
            'std': float(samples.std()),
            'p10': float(np.percentile(samples, 10)),
            'p90': float(np.percentile(samples, 90))
        }
