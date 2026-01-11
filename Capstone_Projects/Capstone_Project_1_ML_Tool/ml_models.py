import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import io

class MLTool:
    def __init__(self):
        self.model = None
        self.model_type = None

    def train(self, model_type, data_csv, target_column, params):
        try:
            # Load Data
            df = pd.read_csv(io.StringIO(data_csv))
            
            if target_column not in df.columns:
                return {"error": f"Target column '{target_column}' not found in dataset."}

            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Simple preprocessing: Handle non-numeric 
            X = pd.get_dummies(X, drop_first=True)
            
            # Split
            test_size = float(params.get('test_size', 0.2))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Select Model
            if model_type == 'linear_regression':
                self.model = LinearRegression()
                self.model.fit(X_train, y_train)
                predictions = self.model.predict(X_test)
                metric = mean_squared_error(y_test, predictions)
                metric_name = "Mean Squared Error"
                self.model_type = 'linear'
                
            elif model_type == 'logistic_regression':
                self.model = LogisticRegression(max_iter=1000)
                self.model.fit(X_train, y_train)
                predictions = self.model.predict(X_test)
                metric = accuracy_score(y_test, predictions)
                metric_name = "Accuracy"
                self.model_type = 'logistic'
                
            else:
                return {"error": "Invalid model type"}

            return {
                "success": True,
                "metric_name": metric_name,
                "metric_value": metric,
                "coef": self.model.coef_.tolist(),
                "intercept": float(self.model.intercept_)
            }

        except Exception as e:
            return {"error": str(e)}

    def predict(self, input_features):
        if not self.model:
            return {"error": "Model not trained yet."}
        
        try:
            # Expecting input_features to be a list of values matching the trained features
            # This is a basic implementation; in a full tool, you'd match columns precisely
            prediction = self.model.predict([input_features])
            return {"prediction": prediction.tolist()[0]}
        except Exception as e:
            return {"error": str(e)}
