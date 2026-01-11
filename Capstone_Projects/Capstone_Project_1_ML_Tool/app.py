from flask import Flask, render_template, request, jsonify
from ml_models import MLTool

app = Flask(__name__)
tool = MLTool()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    try:
        data = request.json
        csv_content = data.get('csv_data')
        model_type = data.get('model_type')
        target_col = data.get('target_col')
        test_size = data.get('test_size', 0.2)
        
        result = tool.train(model_type, csv_content, target_col, {'test_size': test_size})
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = data.get('features') # list of floats
        result = tool.predict(features)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
