import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV


# THe import dataset from EDA
df = pd.read_pickle('data.plk')

preprocessing = ColumnTransformer(transformers =[],
      remainder= 'passthrough'                           
)
model = XGBClassifier()

final_pipeline = Pipeline(steps=[
  ('processing',preprocessing),
  ('model',XGBClassifier())
])

# spliting the data

x_train , x_test, y_train, y_test = train_test_split(df.drop('Class', axis=1), df['Class'],
test_size=0.2, random_state=101)

param_grid = {
  'model__n_estimators': [100,300,500],
  'model__learning_rate': [0.01,0.1,0.3,0.5],
  'model__max_depth': [3,5,7],
  'model__gamma': [0.1,1,3,5],
}
grid = RandomizedSearchCV(
  estimator=final_pipeline,
  param_distributions =param_grid,
  n_iter=4,
  scoring='accuracy',
  cv=3,
  n_jobs=-1
)

grid.fit(x_train,y_train)


print(grid.predict(x_test))

