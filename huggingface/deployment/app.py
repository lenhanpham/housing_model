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
import gradio as gr

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

# Define the prediction function
def predict(longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity):
    """
    Predicts the house price based on the given input features.

    Args:
        Each input feature as a separate argument.

    Returns:
        A string representing the predicted house price.
    """

    try:
        # Create a DataFrame from the input values
        input_df = pd.DataFrame([[
            longitude, latitude, housing_median_age, total_rooms,
            total_bedrooms, population, households, median_income, ocean_proximity
        ]], columns=cols)

        # Make the prediction
        prediction = model.predict(input_df)

        return f"Expected House Price: ${prediction[0]:,.2f}"
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred. Please check your input values."

# Define Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Longitude"),
        gr.Textbox(label="Latitude"),
        gr.Textbox(label="Housing Median Age"),
        gr.Textbox(label="Total Rooms"),
        gr.Textbox(label="Total Bedrooms"),
        gr.Textbox(label="Population"),
        gr.Textbox(label="Households"),
        gr.Textbox(label="Median Income"),
        gr.Textbox(label="Ocean Proximity"),
    ],
    outputs="text",
    title="California Housing Price Prediction",
    description="Predict house prices based on various features.",
    flagging_options=["Spam", "Incorrect Prediction", "Other"] # Custom flagging options
)

if __name__ == "__main__":
    interface.launch(debug=True, server_name="0.0.0.0")
