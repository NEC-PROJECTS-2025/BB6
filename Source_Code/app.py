from flask_bootstrap import Bootstrap
import sklearn
from flask import Flask,redirect,render_template,url_for,request,url_for
from flask import jsonify
import pandas as pd
import numpy as np
from joblib import load # type: ignore


app = Flask(__name__)

best_model=load('model.joblib')
scaler = load('scaler.joblib')
lda = load('lda.joblib')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['POST','GET'])
def prediction():
    return render_template('prediction.html')
@app.route('/cost',methods=['POST','GET'])
def cost():
    try:
        d_ata = [float(request.form['battery_power']),str(request.form['bluetooth']),str(request.form['dual_sim']),str(request.form['has_4g']),float(request.form['int_memory']),float(request.form['mobile_wt']),float(request.form['px_height']),float(request.form['px_width']),float(request.form['ram']),str(request.form['touch_screen']),str(request.form['wifi'])]

        # Convert "Yes"/"No" to binary values
        for i in range(len(d_ata)):
            if d_ata[i] == "Yes":
                d_ata[i] = 1
            elif d_ata[i] == "No":
                d_ata[i] = 0

        # Reshape data for scaler and model
        d_ata = np.array(d_ata).reshape(1, -1)

        # Normalize and transform
        norm = scaler.transform(d_ata)
        l_da = lda.transform(norm)

        # Predict the price range
        predict = best_model.predict(l_da)
        price_range_map = {0: 'Low Cost', 1: 'Medium Cost', 2: 'High Cost', 3: 'Very High Cost'}
        output = price_range_map.get(predict[0], "Unknown")
        return render_template("result.html", predict=output)
    except Exception as e:
             return jsonify({"error":f"the error is {e}"}), 500
@app.route('/about',methods=['POST',"GET"])
def about():
    try:
        return render_template("about.html")
    except Exception as e:
        return jsonify({"error":f"the error is {e}"}),5001
@app.route('/contact',methods=["POST","GET"])
def contact():
    try:
      return render_template("contact.html")
    except Exception as e:
        return jsonify({"error":f"the error is {e}"}),5002

           
@app.route('/home',methods=["POST","GET"])
def home():
    try:
      return render_template("home.html")
    except Exception as e:
        return jsonify({"error":f"the error is {e}"}),5002


if __name__ == '__main__':
    app.run(debug=True)
