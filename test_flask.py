from flask import Flask, jsonify
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def home():
    return "StudyMate is running!"

@app.route('/api/test')
def test():
    return jsonify({
        'status': 'success',
        'message': 'Flask test endpoint is working!',
        'time': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("Starting test Flask server...")
    print("Visit http://localhost:5000 in your browser")
    print("Or test the API at http://localhost:5000/api/test")
    app.run(debug=True, port=5000)
