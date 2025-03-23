from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/api/table', methods=['POST'])
def generate_table():
    data = request.get_json()
    number = data['number']
    print(f"Server processing a request for number {number}")
    table = {i: number * i for i in range(1,11)}
    return jsonify(table=table)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)

