
# import flast module
from flask import Flask,jsonify, send_from_directory
from flask import request
from Scrape import scrapper
from GenderDetection.Detect import read_images_from_folder
from RankingAlgorithms import RankingAlgos
from flask_cors import CORS
import os
import shutil
import urllib.parse
# instance of flask application
app = Flask(__name__)
CORS(app)
 
# home route that returns below text when root url is accessed
@app.route("/")
def hello_world():
    delete_all_files()
    return "<p>Hello, World!</p>"

@app.route('/reranked_images/<path:filename>')
def serve_image(filename):

    return send_from_directory('reranked_images', filename)

@app.route('/before_images/<path:filename>')
def serve2_image(filename):
    encoded_filename = urllib.parse.quote(filename)
    print(send_from_directory('GenderDetection/ScrapedImages', encoded_filename))
    return send_from_directory('GenderDetection/ScrapedImages', encoded_filename)


@app.route('/api/get_search_images', methods=['GET'])
def get_before_images():
    images_folder = 'GenderDetection/ScrapedImages'  # Change this to the path of your local images folder
    image_files = os.listdir(images_folder)
    print("images_folder: ",images_folder,image_files)
    images = []
    for filename in image_files:
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            src = f"{filename}"  # Assuming the images are served from a static route named 'images'
            images.append({"src": "http://127.0.0.1:5000/before_images/"+src})
    
    return jsonify(images)

@app.route('/api/get_reranked_images', methods=['GET'])
def get_images():
    images_folder = 'reranked_images'  # Change this to the path of your local images folder
    image_files = os.listdir(images_folder)
    
    images = []
    for filename in image_files:
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            src = f"{filename}"  # Assuming the images are served from a static route named 'images'
            images.append({"src": "http://127.0.0.1:5000/reranked_images/"+src})
    
    return jsonify(images)

@app.route('/handle_post', methods=['POST'])
def handle_post():
    delete_all_files()
    request_data = request.get_json()
    print(request_data)
    # try:
    if(request_data["search_key"]!="" and int(request_data["no_of_images"])>0 ):
            search_key=request_data["search_key"]
            no_of_images=int(request_data["no_of_images"])
            response=scrapper(search_key,no_of_images)
            response=response+"\n"+read_images_from_folder(search_key)
            response=response+"\n"+RankingAlgos("GenderDetection/ScrapedImages","relevance_aware_swapping")
            return(response)
    else:
            print(request_data)
            return "Wrong values"
    # except BaseException as e:
        # return f"error: {e} is not valid."


def delete_all_files():
    try:
        shutil.rmtree(os.getcwd()+"/GenderDetection/ScrapedImages")
        # shutil.rmtree('mydir')
        shutil.rmtree(os.getcwd()+"/Male")
        shutil.rmtree(os.getcwd()+"/Female")
        shutil.rmtree(os.getcwd()+"/reranked_images")
        print("% s removed successfully")
    except OSError as error:
        print(error)
        print("File path can not be removed")


if __name__ == '__main__':  
   app.run()

