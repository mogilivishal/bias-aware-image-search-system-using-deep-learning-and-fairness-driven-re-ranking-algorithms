import os
# from Detect import read_images_from_folder
import shutil
from google_images_search import GoogleImagesSearch

# class choose_search_eng:
#     def __init__(self, search_eng):
#         self.search_eng 

def scrapper(search_key:str,No_of_images:int):
   
    # Set up your Google API credentials
    API_KEY = 'Enter your API Key here'
    CX = '8116e1240cca141cf'

    # Create a GoogleImagesSearch object
    gis = GoogleImagesSearch(API_KEY, CX)

    # Define your search query and parameters


    search_params = {
        'q': search_key,
        'num': No_of_images,  # Number of images to fetch
        'safe': 'high',  # Safe search level: high, medium, off
        'fileType': 'jpg',  # File type: jpg, png, gif, bmp, svg, webp
        'imgSize': 'medium',  # Image size: large, medium, icon
    }



    # Perform the image search
    gis.search(search_params=search_params)

    # Get URLs of fetched images
    urls = [image.url for image in gis.results()]
    file_key=search_params['q']
    file_path=os.getcwd()+'/GenderDetection/ScrapedImages'
    # Download images
    for i, url in enumerate(urls):
        # Replace 'path_to_save_image' with the path where you want to save the images
        # file_path = os.path.join(file_path,file_key)
        gis.download(url, file_path)

    return "Images downloaded successfully!"

# def search_eng_1():

# def search_eng_2():

