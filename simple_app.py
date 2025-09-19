from flask import Flask, render_template_string

app = Flask(__name__)

@app.route('/')
def hello():
    return "StudyMate is working!"

if __name__ == '__main__':
    app.run(debug=True)
