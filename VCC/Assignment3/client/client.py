import requests
import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('number', type=int, help="Input Number")

    args = parser.parse_args()
    if args.number:
        print(f"Requesting Server to generate a table of {args.number}")
        data = {'number': args.number}
        headers = {'Content-Type': 'application/json'}
        response = requests.post('http://192.168.56.101:5050/api/table', data=json.dumps(data), headers=headers)
        if response.status_code == 200:
            print("Server response: ", response.json()['message'])
        else:
            print("Failed to reach the server!")
