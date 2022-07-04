import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

Data = pd.read_csv('toy_dataset.csv')
'''For checking null Values'''
Data.isnull().sum()
Data = Data.drop(['Number'], axis=1)

''' There is not null values available'''
Data['City'].value_counts()
x = Data.iloc[:, :-1].values
y = Data.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

y = LabelEncoder().fit_transform(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

step_1 = ColumnTransformer(transformers=[('onehotencoder', OneHotEncoder(sparse=False, drop='first'), [0, 1])],
                           remainder='passthrough')
step_2 = KNeighborsClassifier()

pipe = Pipeline([('step_1', step_1), ('step_2', step_2)])
pipe.fit(x_train, y_train)

# confusion_metrics
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred = pipe.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print('Accuracy_score =', accuracy_score(y_test, y_pred))

# save the model to disk
filename = 'week 4.sav'
pickle.dump(pipe, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_test, y_test)
print(result)

from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/", methods=['POST'])
def bFlipper(output=None):
    Input = []
    City = request.form.get('city_name')
    Gender = request.form.get('Gender')
    Age = request.form.get('Age')
    Income = request.form.get('Income')

    Input.append(City)
    Input.append(Gender)
    Input.append(Age)
    Input.append(Income)

    y_prediction = loaded_model.predict([Input])
    print(y_prediction)

    return 'Illness is available: {}'.format(bool(y_prediction))


if __name__ == "__main__":
    app.run()
