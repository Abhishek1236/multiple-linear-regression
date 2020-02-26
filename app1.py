import flask
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
with open(f'model2.pkl', 'rb') as f:
    model = pickle.load(f)


app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return (flask.render_template('home.html'))

    if flask.request.method == 'POST':
        RD_Spend = flask.request.form['RD_Spend']
        Administration = flask.request.form['Administration']
        Marketing_Spend = flask.request.form['Marketing_Spend']
        State = flask.request.form['State']
        
        input_variables = pd.DataFrame([[RD_Spend, Administration,  Marketing_Spend, State]],
                                       columns=['RD_Spend', 'Administration', 'Marketing_Spend', 'State'],
                                       index=['input'])
        d = {'New York':0,'California':1,'Florida':2}
        input_variables['State'] = input_variables['State'].map(d)
        #input_variables = np.array(input_variables)
        predictions = model.predict(input_variables)
        #print(predictions)

        return flask.render_template('home.html', original_input={'RD_Spend': RD_Spend,'Administration':Administration,'Marketing_Spend':Marketing_Spend,'State':State},
                                     result=predictions[0])


if __name__ == '__main__':
    app.run(debug=False)
