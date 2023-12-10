from flask import Flask, render_template, request, jsonify, make_response
from flask_cors import CORS
import joblib
#from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)

# Load the saved model
model = joblib.load('iris_model.pkl')
#le = LabelEncoder()
@app.route('/')
def home():
    # Render the predict.html page
    return render_template('predict.html')

class_to_label = {0: 'Iris-Setosa', 1: 'Iris-versicolor', 2: 'Iris-vriginca'}
@app.route('/predict', methods=['GET', 'POST', 'OPTIONS'])
def predict():
    if request.method == 'GET':
        return "This is a GET request. Use POST with JSON data to make predictions."

    # Handle pre-flight request
    if request.method == 'OPTIONS':
        response = make_response()
    else:
        try:
            # Get input data from the frontend
            input_data = request.json['data']

            # Debugging: Print received data
            print('Received data:', input_data)

            # Ensure correct order of features
            features_order = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            input_data_ordered = [input_data[feature] for feature in features_order]

            # Perform predictions using the loaded model
            predictions = model.predict([input_data_ordered])
            predicted_labels = [class_to_label[class_num-1] for class_num in predictions]
            
            print('prediction_label',predicted_labels)
            print('predictions',predictions)
      #      predicted_class = le.inverse_transform(predictions)
         #   print('predicted_class',predicted_class)

            # Return predictions as JSON
            response_data = {'predictions': predicted_labels}
            response = make_response(jsonify(response_data))
        except Exception as e:
            response_data = {'error': f'400 Bad Request: {str(e)}'}
            response = make_response(jsonify(response_data))
            response.status_code = 400
            print('Exception:', str(e))

    # Set CORS headers for the response
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'

    return response


if __name__ == '__main__':
    app.run(port=5000)
