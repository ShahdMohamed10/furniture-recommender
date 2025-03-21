from flask import Flask, jsonify
import os

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({
        'status': 'success',
        'message': 'Furniture Recommendation API is running',
        'endpoints': {
            'test': '/test'
        }
    })

@app.route('/test')
def test():
    return jsonify({
        'status': 'success',
        'message': 'Test endpoint is working'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

