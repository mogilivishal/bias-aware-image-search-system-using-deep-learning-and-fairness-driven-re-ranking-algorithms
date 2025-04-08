from Scrape import scrapper
from GenderDetection.Detect import read_images_from_folder
from RankingAlgorithms import RankingAlgos

search_key=input("Please Enter the Search key:")
no_of_images=int(input("Please Enter the number of images to be retrived:"))

scrapper(search_key,no_of_images)
read_images_from_folder(search_key)
RankingAlgos("GenderDetection/ScrapedImages","relevance_aware_swapping")

# RankingAlgos("GenderDetection/ScrapedImages/"+search_key,"relevance_aware_swapping")
