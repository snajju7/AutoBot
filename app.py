
from flask import Flask, redirect, render_template, request, json
from response import autobot_response

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get')
def get_response():
    user_input = request.args.get('chat_area')
    res = autobot_response()
    return res


if __name__ == "__main__":
    app.run(debug=True)

