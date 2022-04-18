# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 15:28:05 2022

@author: Alessandro
"""
from flask import Flask

app = Flask(__name__)


@app.route('/')
def index():
    return 'Web App with Python Flask!'


app.run(host='0.0.0.0', port=1000)


# =============================================================================
# from flask import Flask
#
# app = Flask(__name__)
#
# @app.route('/user/<username>')
# def show_user(username):
#     # Greet the user
#     return f'Hello {username} !'
#
# # Pass the required route to the decorator.
# @app.route("/hello")
# def hello():
#     return "Hello, Welcome to GeeksForGeeks"
#
# @app.route("/")
# def index():
#     return "Homepage of GeeksForGeeks"
#
# if __name__ == "__main__":
#     app.run(debug=True)
#
# =============================================================================
