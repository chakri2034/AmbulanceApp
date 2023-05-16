import flask
from flask import *

app = Flask(__name__,static_url_path = '/static')

@app.route('/')
def Home():
   return render_template('new 1.html')
   

@app.route("/add",methods = ['POST'])

def add():
   if request.method == 'POST':
       a = int(request.form['cn1'])
       b = int(request.form['cn2'])
       c = int(request.form['ln1'])
       d = int(request.form['ln2'])
 
 
       import pandas as pd
       url = "https://raw.githubusercontent.com/chakri2034/AmbulanceApp/main/Ambulance_dataset.csv"
       data = pd.read_csv(url)
       
       import numpy as np
       features = data[['condition1', 'condition2','location1','location2']]
       targets = data['Hospital']
       
       from sklearn.preprocessing import LabelEncoder
       le = LabelEncoder()
       features.condition1 = le.fit_transform(data['condition1'])
       features.condition2 = le.fit_transform(data['condition2'])
       features.location1 = le.fit_transform(data['location1'])
       features.location2 = le.fit_transform(data['location2'])
       
       features = np.array(features[['condition1', 'condition2','location1','location2']], dtype = 'object')
       targets = np.array(data['Hospital'], dtype = 'object')
       
       from sklearn.model_selection import train_test_split
       xTrain, xTest, yTrain, yTest = train_test_split(features, targets, test_size = 0.3, random_state = 42)
       
       param_grid_nb = {'var_smoothing': np.logspace(0,-9, num=100)}
       
       from sklearn.naive_bayes import GaussianNB
       from sklearn.model_selection import GridSearchCV
       nbModel_grid = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid_nb, verbose=1, cv=10, n_jobs=-1)
       nbModel_grid.fit(xTrain, yTrain)
       
       y_pred = nbModel_grid.predict([[a,b,c,d]])
       
       return render_template('new 1.html',results = "Hospital is : {}".format(y_pred))
   else:
       return render_template('new 1.html')

if __name__ == "__main__":
    app.run(debug = True)