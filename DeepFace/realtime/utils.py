import os
from os import listdir
import numpy as np
import cv2


from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.keras.preprocessing import image

def l2_normalization(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))

def Euclidean_Distance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def Cosine_Distance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def detect_face(img_path, target_size=(152, 152)):
    # initializing th classifiers
    cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    img = cv2.imread(img_path)
    faces = faceCascade.detectMultiScale(img, 1.3, 5)
    
    if len(faces) > 0:
        x,y,w,h = faces[0]
        
        margin = 0
        x_margin = w * margin / 100
        y_margin = h * margin / 100
        
        if y - y_margin > 0 and y+h+y_margin < img.shape[1] and x-x_margin > 0 and x+w+x_margin < img.shape[0]:
            detected_face = img[int(y-y_margin):int(y+h+y_margin), int(x-x_margin):int(x+w+x_margin)]
        else:
            detected_face = img[int(y):int(y+h), int(x):int(x+w)]
        
        detected_face = cv2.resize(detected_face, target_size)
        
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        
        #normalize in [0, 1]
        img_pixels /= 255 
        
        return img_pixels
    else:
        raise ValueError("Face could not be detected in ", img_path,". Please confirm that the picture is a face photo.")
        
def capture():
    names =  ["SurajKarki"]
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Create your image database.")
    img_counter = 0
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("Image", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            if (img_counter == len(names)):
                break
            img_name = "database/{}.jpg".format(names[img_counter])
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
    cam.release()
    cv2.destroyAllWindows()


def get_embedding(deepface):
    employee_pictures = "database/"
    employees = dict()

    for file in listdir(employee_pictures):
        employee, extension = file.split(".")
        img_path = 'database/%s.jpg' % (employee)
        img = detect_face(img_path)

        representation = deepface.predict(img)[0]
        employees[employee] = representation
    print("employee representations retrieved successfully")
    
    return employees
