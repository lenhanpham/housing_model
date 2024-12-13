from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np

# Define custom functions and classes used in preprocessing
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ['ratio']

# Load your trained model
model = joblib.load("california_housing_random_forest_regression_model.pkl")
cols = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
        'total_bedrooms', 'population', 'households', 'median_income',
        'ocean_proximity']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        int_features = [x for x in request.form.values()]
        print(f"Received form data: {int_features}")

        input_df = pd.DataFrame([int_features], columns=cols)
        print(f"Data for prediction: {input_df}")

        prediction = model.predict(input_df)
        print(f"Prediction result: {prediction}")

        return render_template('home.html', pred='Expected House Price: ${:,.2f}'.format(prediction[0]))
    except Exception as e:
        print(f"Error occurred: {e}")
        return render_template('home.html', pred='An error occurred while processing your request.')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json(force=True)
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': 'An error occurred while processing your request.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
