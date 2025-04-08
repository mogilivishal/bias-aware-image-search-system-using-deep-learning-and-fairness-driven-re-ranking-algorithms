import os
import glob
import shutil
# from age_gender_predictor import AgeGenderPredictor
# age_gender_predictor = AgeGenderPredictor()
from GenderDetection.age_gender_predictor import AgeGenderPredictor


def read_images_from_folder(folder_path):
    # try:
    #     shutil.rmtree(os.getcwd()+"/GenderDetection/ScrapedImages")
    #     # shutil.rmtree('mydir')
    #     shutil.rmtree(os.getcwd()+"/Male")
    #     shutil.rmtree(os.getcwd()+"/Female")
    #     shutil.rmtree(os.getcwd()+"/reranked_images")
    #     print("% s removed successfully")
    # except OSError as error:
    #     print(error)
    #     print("File path can not be removed")
    
    folder_path=os.getcwd()+"/GenderDetection/ScrapedImages"
    male=0
    female=0
    images = []
    age_instance=AgeGenderPredictor()
    for img_path in glob.glob(folder_path + '/*.*'):
        j=age_instance.process_image_file(img_path)  # For image file detection
        if(type(j)==tuple):
            male=male+j[0]
            female=female+j[1]
    return ("Male: "+str(male)+", Female: "+str(female))

# read_images_from_folder('Doctor')

