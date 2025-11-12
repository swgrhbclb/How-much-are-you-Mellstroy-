import cv2
import numpy as np
import os
from functions import loader, grey_convert, get_photo_file, img_resize, check_folder, calculate_similarity_percentage, VERDICTS

class OpenCVMellstroy:
    def __init__(self, mellstroy_photo):
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_samples= []
        self.ids = []
        self.mellstroy = os.path.basename(mellstroy_photo)
        self.mellstroy_id = 1
        self.is_trained = False
        self.detected_folder = "detected_faces"
        self.train_model(mellstroy_photo)

    def process_train(self, img_path):
        try:
            image = loader(img_path)
            gray_image = grey_convert(image)
            faces = self.face_detector.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors = 10,minSize= (30,30))
            for i, (x, y, w, h) in enumerate(faces):
                face_roi = gray_image[y:y+h, x:x+w]
                self.face_samples.append(face_roi)
                self.ids.append(self.mellstroy_id)
                self.save_detected_face(face_roi, i)
        except Exception as e:
            print(f"ОШибка короче лица не найдены {img_path}: {e}")

    def save_detected_face(self, face_image, face_index):
        filename = f"face_{face_index + 1}.jpg"
        save_path = os.path.join(self.detected_folder, filename)
        cv2.imwrite(save_path, face_image)
                
    def train_model(self, f_path):
        try:
            img_file = check_folder(f_path)
            for filename in img_file:
                full_path = os.path.join(f_path, filename)
                self.process_train(full_path)
            if len(self.face_samples) > 0:
                self.face_recognizer.train(self.face_samples, np.array(self.ids))
                self.is_trained = True
            else:
                print("не найдено лиц для обучения")
        except Exception as e:
            print(f"ошибка при обучении модели: {e}")
    def analysis(self, user_path):
        try:
            user_image = loader(user_path)
            gray_user = grey_convert(user_image)
            user_faces = self.face_detector.detectMultiScale(gray_user, scaleFactor=1.1, minNeighbors= 10,minSize= (30,30))
            if len(user_faces) == 0:
                print("Лица не найдены")
            for i, (x,y,w,h) in enumerate(user_faces):
                user_face_roi = gray_user[y:y+h, x:x+w]
                self.save_detected_face(user_face_roi, i)
            similarity = self.calculate_face_similarity(gray_user, user_faces[0])
            return self.generate_verdict(similarity)
        except Exception as e:
            return f"Ошибка анализа"
    
    def calculate_face_similarity(self, gray_image, face_coordinates):
        x, y, w, h = face_coordinates
        user_face = gray_image[y:y+h, x:x+w]
        
        if self.is_trained:
            label, confidence = self.face_recognizer.predict(user_face)
            return calculate_similarity_percentage(confidence)
        else:
            print("Ничего не получилось")
    

    def generate_verdict(self, similarity):
        for min_sim, max_sim, verdict_template in VERDICTS:
            if min_sim <= similarity < max_sim:
                return verdict_template.format(similarity=similarity)
        return f"Сходство: {similarity:.1f}%. Результаты могут быть тревожными."