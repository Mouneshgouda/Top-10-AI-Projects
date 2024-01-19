from flask import Flask, render_template, session, redirect, url_for, session
import pandas as pd
from flask_wtf import FlaskForm

from wtforms import StringField
from wtforms import SubmitField
from wtforms.validators import NumberRange

import numpy as np  
import joblib


def return_prediction(model,scaler,sample_json):
    
    ft_a = sample_json['distance_from_home']
    ft_b = sample_json['distance_from_last_transaction']
    ft_c = sample_json['ratio_to_median_purchase_price']
    ft_d = sample_json['repeat_retailer']
    ft_e = sample_json['used_chip']
    ft_f = sample_json['used_pin_number']
    ft_g = sample_json['online_order']
    
    columns = ['distance_from_home',
                'distance_from_last_transaction',
                'ratio_to_median_purchase_price',
                'repeat_retailer',
                'used_chip',
                'used_pin_number',
                'online_order']
    
    transaction = [[ft_a,ft_b,ft_c,ft_d,ft_e,ft_f,ft_g]]
    
    transaction = pd.DataFrame(transaction, columns = columns)
    
    transaction = scaler.transform(transaction)
    
    transaction = pd.DataFrame(transaction, columns = columns)
    
    classes = np.array(['not fraudulent', 'fraudulent'])
    
    class_ind = model.predict(transaction)

    class_ind = class_ind[0]
    
    return classes[int(class_ind)] 

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

# LOAD THE MODEL AND THE SCALER!
model = joblib.load("agada_credit_card_fraud_prediction.sav")
scaler = joblib.load("agada_credit_card_fraud_prediction_scaler.pkl")

class FlowerForm(FlaskForm):

    ft_1 = StringField('Distance from home')
    ft_2 = StringField('Distance from location of last transaction')
    ft_3 = StringField('Ration to median purchase price')
    ft_4 = StringField('Repeat retailer? Enter 1 if answer is yes and 0 otherwise')
    ft_5 = StringField('Used cheap? Enter 1 if answer is yes and 0 otherwise')
    ft_6 = StringField('Used pin number? Enter 1 if answer is yes and 0 otherwise')
    ft_7 = StringField('Online transaction? Enter 1 if answer is yes and 0 otherwise')

    submit = SubmitField('Predict')

@app.route('/', methods=['GET','POST'])
def index():

    # Create instance of the form.
    form = FlowerForm()
    # If the form is valid on submission (we'll talk about validation next)
    if form.validate_on_submit():
        # Grab the data from the breed on the form.

        session['ft_1'] = form.ft_1.data
        session['ft_2'] = form.ft_2.data
        session['ft_3'] = form.ft_3.data
        session['ft_4'] = form.ft_4.data
        session['ft_5'] = form.ft_5.data
        session['ft_6'] = form.ft_6.data
        session['ft_7'] = form.ft_7.data

        return redirect(url_for("prediction"))

    return render_template('home.html', form=form)

@app.route('/prediction')
def prediction():

    content = {}

    content['distance_from_home'] = float(session['ft_1'])
    content['distance_from_last_transaction'] = float(session['ft_2'])
    content['ratio_to_median_purchase_price'] = float(session['ft_3'])
    content['repeat_retailer'] = float(session['ft_4'])
    content['used_chip'] = float(session['ft_5'])
    content['used_pin_number'] = float(session['ft_6'])
    content['online_order'] = float(session['ft_7'])

    results = return_prediction(model=model,scaler=scaler,sample_json=content)

    return render_template('prediction.html',results=results)


if __name__ == '__main__':
    app.run(debug=True)