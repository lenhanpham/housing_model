### The whole code for training: data preprocessing, traning using only RandomizedSearchCV
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import joblib


def load_housing_data():
  tarball_path = Path("datasets/housing.tgz")
  if not tarball_path.is_file():
    Path("datasets").mkdir(parents=True, exist_ok=True)
    url = "https://github.com/ageron/data/raw/main/housing.tgz"
    urllib.request.urlretrieve(url, tarball_path)
    with tarfile.open(tarball_path) as housing_tarball:
      housing_tarball.extractall(path="datasets")
  return pd.read_csv(Path("datasets/housing/housing.csv"))
housing = load_housing_data()



train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing = train_set.drop('median_house_value', axis=1)
housing_labels = train_set['median_house_value'].copy()

### create a ClusterSimilarity class that uses KMean and rbf_kernal to check simularity

class ClusterSimilarity(BaseEstimator, TransformerMixin):
  def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
    self.n_clusters = n_clusters
    self.gamma = gamma
    self.random_state = random_state
  def fit(self, X, y=None, sample_weight=None):
    self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
    self.kmeans_.fit(X, sample_weight=sample_weight)
    return self # always return self!
  def transform(self, X):
    return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
  def get_feature_names_out(self, names=None):
    return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

#### full data tranformation using ColumTransformer or make_columnTransformer


def column_ratio(X):
  return X[:,[0]]/X[:,[1]]

def ratio_name(function_transformer, feature_names_in):
  return ['ratio']  # feature names out

def ratio_pipeline():
  return make_pipeline(
      SimpleImputer(strategy='median'),
      FunctionTransformer(column_ratio, feature_names_out=ratio_name), ## note that a callable function ratio_name here without (), not calling the function immediately like ratio_name()
      StandardScaler()
  )

log_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    FunctionTransformer(np.log, feature_names_out='one-to-one'),
    StandardScaler()
    )

cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)

default_num_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler()
)

categorical_pipeline = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore'))

full_preprocessing = ColumnTransformer([
    ('bedrooms_ratio', ratio_pipeline(), ['total_bedrooms', 'total_rooms']),
    ('room_per_house', ratio_pipeline(), ['total_rooms', 'households']),
    ('people_per_house', ratio_pipeline(), ['population', 'households']),
    ('log', log_pipeline, ['total_bedrooms', 'total_rooms', 'population', 'households', 'median_income']),
    ('geo', cluster_simil, ['latitude', 'longitude']),
    ('cat', categorical_pipeline, make_column_selector(dtype_include=object))
],
remainder=default_num_pipeline
)

housing_prepared = full_preprocessing.fit_transform(housing)

param_distribution = {'preprocessing__geo__n_clusters': randint(low=3, high=50),
                      'random_forest__max_features': randint(low=2, high=20)}

full_pipeline = Pipeline([
    ('preprocessing', full_preprocessing),
    ('random_forest', RandomForestRegressor(random_state=42))
    ])

rnd_Search = RandomizedSearchCV(
    full_pipeline, param_distribution,
    scoring='neg_root_mean_squared_error',
    n_iter=10, cv=3,
    random_state=42)

rnd_Search.fit(housing, housing_labels)
rnd_search_model = rnd_Search.best_estimator_

## evaluate model
X_test = test_set.drop('median_house_value', axis=1)
y_test = test_set['median_house_value'].copy()
y_predictions = rnd_search_model.predict(X_test)

final_rmse = root_mean_squared_error(y_test, y_predictions)
print(final_rmse)

## Save the trained model with joblib

joblib.dump(rnd_search_model, "california_housing_random_forest_regression_model.pkl")
