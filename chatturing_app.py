import requests
from flask import Flask, request, jsonify
from chatturing_inference import run_inference

app = Flask(__name__)

@app.route('/api/messages', methods=['POST'])
def receive_message():
    data = request.get_json()
    print('data', data)
    message = data.get('message')
    saved = data.get('saved')
    user_id = data.get('user_id')
    print(f'Received message: {message} from user: {user_id}')
    # Process the message as needed
    model_message, saved = run_inference(message, saved)

    print('model_message!', model_message)
    print('saved!', saved)

    # Respond back to the Elixir app
    return jsonify({'status': 'success', 'message': model_message[:-10], 'saved': saved}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# url = "http://localhost:4000/api/messages"

# # Define the payload to send
# payload = {"message": "Hello from Python"}

# # Send a POST request
# response = requests.post(url, json=payload)

# # Print the response
# if response.status_code == 200:
#     print("Success:", response.json())
# else:
#     print("Error:", response.status_code, response.text)