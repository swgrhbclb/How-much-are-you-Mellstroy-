from model import OpenCVMellstroy
from functions import formats
import os

def main():
    mellstroy_photo = "mellstroy_photo"
    user_photo_folder = "user_photo"
    analys = OpenCVMellstroy(mellstroy_photo)
    user_photos = []
    for filename in os.listdir(user_photo_folder):
        if filename.lower().endswith(formats):
            full_path = os.path.join(user_photo_folder, filename)
            user_photos.append(full_path)
    for photo_path in user_photos:
        result = analys.analysis(photo_path)
        print(result)

if __name__ == "__main__":
    main()

