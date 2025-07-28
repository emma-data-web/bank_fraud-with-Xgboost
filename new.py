import pymysql
from sqlalchemy import create_engine
import pandas as pd

db_url = 'mysql+pymysql://root:Emmanuel-3102@localhost:3306/practice' 

engine = create_engine(db_url)

data = pd.read_sql('select * from titanic_train', con=engine)

print(data)