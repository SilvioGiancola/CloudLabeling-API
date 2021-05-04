from flask import Flask, request, jsonify, send_from_directory       
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from cloudlabeling import server as cloudserver


app = Flask(__name__)
import boto3
import botocore

# Main index for visualization
@app.route('/', methods=['GET'])
def index():
    return server.get_index()

# Call to download files from training
@app.route('/training/<path:filename>', methods=['GET', 'POST'])
def download(filename):   
    return sever.download(filename)

# API to run prediction on specifc projects
@app.route("/api/predict", methods=['POST'])
def predict():
    return server.predict(request)

if __name__ == "__main__":

    parser = ArgumentParser(
        description='Run server to perform inference for object detection.',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--HOST", type=str, default="localhost",
                        help="host IP")
                        
    parser.add_argument("--PORT", type=int, default=4000,
                        help="commmunication port")

    args = parser.parse_args()

    server = cloudserver.FlaskServer(args.HOST, args.PORT)
    app.run(host=server.HOST, port=server.PORT)

    print("hell")