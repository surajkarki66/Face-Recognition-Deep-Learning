import cv2
import os
import numpy as np



from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.keras.preprocessing import image

from model import build_model
from utils import l2_normalization, Euclidean_Distance

def verification(target_size=(152, 152), deepface=None, employees=None):
    # initializing th classifiers
    cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    cap = cv2.VideoCapture(0)

    while(True):
        ret, img = cap.read()
        faces = faceCascade .detectMultiScale(img, 1.3, 5)
        
        for (x,y,w,h) in faces:
            if w > 130: #discard small detected faces
                #draw rectangle to main image
                cv2.rectangle(img, (x,y), (x+w,y+h), (67, 67, 67), 3) 
                #crop detected face
                detected_face = img[int(y):int(y+h), int(x):int(x+w)] 
                #resize to 152x152
                detected_face = cv2.resize(detected_face, target_size) 

                img_pixels = image.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis = 0)
                img_pixels /= 255

                captured_representation = deepface.predict(img_pixels)[0]

                distances = []
                # loop over employee database
                for i in employees:
                    employee_name = i
                    source_representation = employees[i]

                    distance = Euclidean_Distance(l2_normalization(captured_representation), l2_normalization(source_representation))
                    distances.append(distance)
                is_found = False
                index = 0
                for i in employees:
                    employee_name = i
                    if index == np.argmin(distances):
                        if distances[index] <= 0.70:

                            print("Detected: ",employee_name, "(",distances[index],")")
                            employee_name = employee_name.replace("_", "")
                            similarity = distances[index]
                            is_found = True
                            break
                    index = index + 1
                
                if is_found:
                    display_img = cv2.imread("database/%s.jpg" % employee_name)
                    pivot_img_size = 112
                    display_img = cv2.resize(display_img, (pivot_img_size, pivot_img_size))
                                    
                    try:
                        resolution_x = img.shape[1]; resolution_y = img.shape[0]
                        
                        label = employee_name+" ("+"{0:.2f}".format(similarity)+")"
                        
                        if y - pivot_img_size > 0 and x + w + pivot_img_size < resolution_x:
                            #top right
                            img[y - pivot_img_size:y, x+w:x+w+pivot_img_size] = display_img
                            cv2.putText(img, label, (x+w, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (223,21,41), 1)					
                            
                            #connect face and text
                            cv2.line(img,(x+int(w/2), y), (x+3*int(w/4), y-int(pivot_img_size/2)),(67,67,67),2)
                            cv2.line(img, (x+3*int(w/4), y-int(pivot_img_size/2)), (x+w, y - int(pivot_img_size/2)), (67,67,67),2)
                        elif y + h + pivot_img_size < resolution_y and x - pivot_img_size > 0:
                            #bottom left
                            img[y+h:y+h+pivot_img_size, x-pivot_img_size:x] = display_img
                            cv2.putText(img, label, (x - pivot_img_size, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (223,21,41), 1)
                            
                            #connect face and text
                            cv2.line(img,(x+int(w/2), y+h), (x+int(w/2)-int(w/4), y+h+int(pivot_img_size/2)),(67,67,67),2)
                            cv2.line(img, (x+int(w/2)-int(w/4), y+h+int(pivot_img_size/2)), (x, y+h+int(pivot_img_size/2)), (67,67,67),2)
                            
                        elif y - pivot_img_size > 0 and x - pivot_img_size > 0:
                            #top left
                            img[y-pivot_img_size:y, x-pivot_img_size:x] = display_img
                            cv2.putText(img, label, (x - pivot_img_size, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (223,21,41), 1)
                            
                            #connect face and text
                            cv2.line(img,(x+int(w/2), y), (x+int(w/2)-int(w/4), y-int(pivot_img_size/2)),(67,67,67),2)
                            cv2.line(img, (x+int(w/2)-int(w/4), y-int(pivot_img_size/2)), (x, y - int(pivot_img_size/2)), (67,67,67),2)
                            
                        elif x+w+pivot_img_size < resolution_x and y + h + pivot_img_size < resolution_y:
                            #bottom righ
                            img[y+h:y+h+pivot_img_size, x+w:x+w+pivot_img_size] = display_img
                            cv2.putText(img, label, (x+w, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (223,21,41), 1)
                            
                            #connect face and text
                            cv2.line(img,(x+int(w/2), y+h), (x+int(w/2)+int(w/4), y+h+int(pivot_img_size/2)),(67,67,67),2)
                            cv2.line(img, (x+int(w/2)+int(w/4), y+h+int(pivot_img_size/2)), (x+w, y+h+int(pivot_img_size/2)), (67,67,67),2)
                        
                    except Exception as e:
                        print("exception occured: ", str(e))
        cv2.imshow('img',img)

        if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
            break

    #kill open cv things
    cap.release()
    cv2.destroyAllWindows()