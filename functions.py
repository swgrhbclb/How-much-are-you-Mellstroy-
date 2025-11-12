import cv2
import os

VERDICTS = [
    (0, 15, "Сходство: {similarity:.1f}%. ты больше мне не друн, а ч тебе не ч(())"),
    (15, 30, "Сходство: {similarity:.1f}%. тебе еще далеко до мурино"),
    (30, 45, "Сходство: {similarity:.1f}%. имеются небольшие позывы лудомании"),
    (45, 60, "Сходство: {similarity:.1f}%. я начинаю чувствовать запах fog'a "),
    (60, 75, "Сходство: {similarity:.1f}%. ты изириджыыыыыджиджи..... изибрибрибриджыджы"),
    (75, 90, "Сходство: {similarity:.1f}%. я уже красный........ по культурному не получится"),
    (90, 100, "Сходство: {similarity:.1f}%. БЭМ БЭМ БЭМ БЭМ БЭМ ТЫ МЭСТРОЙ Ч ТИКТОКЕР ТЫ МОЙ ДРУН")
]

formats = (".jpg", ".png", ".jpeg")

def loader(img_path):
    img = cv2.imread(img_path)
    return img

def grey_convert(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return img

def get_photo_file(f_path):
    img_file = []
    for filename in os.listdir(f_path):
        if filename.lower().endswith(formats):
            img_file.append(filename)
    return img_file

def img_resize(face_image, size=(100,100)):
    return cv2.resize(face_image, size)

def check_folder(f_path):
    if not os.path.exists(f_path):
        print("Папки не существует")
    img_file = get_photo_file(f_path)
    return img_file

def calculate_similarity_percentage(confidence):
    return max(0, 100 - confidence)