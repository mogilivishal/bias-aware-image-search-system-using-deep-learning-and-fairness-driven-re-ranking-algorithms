from Scrape import scrapper
# from Detect import read_images_from_folder

search_key=input("Please Enter the Search key:")
no_of_images=int(input("Please Enter the number of images to be retrived:"))

scrapper(search_key,no_of_images)
# read_images_from_folder(search_key)
