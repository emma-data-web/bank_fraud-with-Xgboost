import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


# THe import dataset from EDA
df = pd.read_pickle('data.plk')

preprocessing = ColumnTransformer(transformers =[],
      remainder= 'passthrough'                           
)

