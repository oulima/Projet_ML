from flask import Flask, render_template, request
import json
import pickle
from sklearn.linear_model import LogisticRegression
from fonctions import prediction, entrainement, nettoyage



app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", title='Home')

""" @app.route("/prediction",methods=['POST'])
def text():
    user_text = request.form.get('input_text')
    print(user_text)
    if len(user_text) > 10:
        user_text = "Positif"
    else :
        user_text = "Négatif"
    return json.dumps({'text_user':user_text})
     """
@app.route("/prediction",methods=['POST'])
def text():
    user_text = request.form.get('input_text')
    retour=prediction(user_text)
    #return json.dumps({'text_user':retour})
    return render_template("interface.html", input_text=user_text,prediction=retour)


@app.route("/result",methods=['POST'])
def retour():
    user_text = request.form.get('input_text')
    print(user_text)
    return json.dumps({'text_user':user_text})

@app.route("/entrainement",methods=['GET'])
def entr(usertexte=None):
    retour = entrainement()
    #return json.dumps({'text_user':retour})
    return render_template("entrainement.html",entrainement=retour)


if __name__ == "__main__":
    app.run(debug=True)

