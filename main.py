import cv2
import os
import time
import pyttsx4
import json
import random
import threading
import logging
import pyfirmata
import numpy as np

from collections import Counter
from datetime import datetime
from kivy.app import App
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.textinput import TextInput
from kivy.utils import get_color_from_hex
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

Logger = logging.getLogger(__name__)  # Replace __name__ with the appropriate logger name
Logger.setLevel(logging.DEBUG)

try:
    board = pyfirmata.Arduino('COM7')
    pin13 = board.get_pin('d:13:o')  # Set pin 13 as output
except:
    pass


class CamApp(App):
    def build(self):
        Window.clearcolor = get_color_from_hex('#191970')
        self.title = "YOLONAS Person Counter"
        self.model = YOLO("yolov8s.pt")
        # results = self.model.val(data='coco8.yaml')
        self.size = (1080, 720)
        self.class_colors = {}
        self.weight = 65
        self.timer = 5000
        
        #self.url = "http://192.000.000.00:8080/video" use the url given by the IP camera
        self.url = 0  # use the webcam
        self.capture = cv2.VideoCapture(self.url)

        self.update_flag = True  # Flag variable to control the update process

        

        self.Led = Button(text="LED", background_color=get_color_from_hex('#000000'), size_hint=(1, 0.1)) 
        self.button = Button(text="Verify", on_press=self.verify, background_color=get_color_from_hex('#00D0D0'), size_hint=(1, 0.1))  # Green color
        self.verification_text = TextInput(text="", size_hint=(1, 0.5))
        layout = BoxLayout(orientation='vertical')
        horizontal_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.5))  # Half of the height
        
        # Create the Camera widget
        self.camera = Camera()
        
        other_widgets = BoxLayout(orientation='vertical')
        other_widgets.padding = dp(40)  # Left and right padding
        other_widgets.spacing = dp(40)  # Gap between buttons
        other_widgets.add_widget(self.Led)
        other_widgets.add_widget(self.button)
        other_widgets.add_widget(self.verification_text)
        
        horizontal_layout.add_widget(self.camera,)
        horizontal_layout.add_widget(other_widgets)
        
        layout.add_widget(horizontal_layout)
        
        # Schedule the update function to run every frame
        Clock.schedule_interval(self.update, 1.0 / 120.0)  # 120 FPS
        
        return layout

    def update(self, dt):
        if not self.update_flag:
            return  # Stop the update process
        
        # Read frame from OpenCV
        ret, frame = self.capture.read()

        # Flip horizontally and convert image to texture
        if ret:
            buf = cv2.flip(frame, 0).tobytes()
            img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.camera.texture = img_texture
        else:
            Logger.warning("Failed to capture frame from the camera.")

    def img_resize(self, img_path):
        img = cv2.imread(img_path)
        self.org_shape = img.shape
        print(f"Orginal Shape: {img.shape}")
        img = cv2.resize(img, self.size)
        print(f"Resized shape : {img.shape}")
        self.resize_shape = img.shape
        return img
    
    def pred(self, img_path):
        img = self.img_resize(img_path)
        result = self.model.predict(img)[0]
        class_names = result.names
        predicted_classes = [class_names[int(label[-1].item())] for label in result.boxes.data]
        class_counts = json.dumps(Counter(predicted_classes), indent=4)
        return result, class_counts, img, class_names

    def verify(self, instance):
        try:
            start = time.time()
            self.date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            # Capture input image from the webcam 
            self.img_path = f"Img_{self.date}.jpg"
            ret, frame = self.capture.read()
            
            # Save captured frame to a file
            self.saved_path = os.path.join('Image Try', self.img_path)
            cv2.imwrite(self.saved_path, frame)
            
            # Perform inference using TensorFlow Lite model
            result, class_counts, img, class_names = self.pred(self.saved_path)
            self.results = result, class_counts, img, class_names
            annotator = Annotator(img)
            print(annotator)
            try:
                for label in result.boxes.data:
                    index = int(label[-1].item())
                    if index not in self.class_colors:
                        # Generate a random color
                        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                        self.class_colors[index] = color
                    else:
                        color = self.class_colors[index]
                        
                    if index < len(class_names):
                        annotator.box_label(
                            label[0:],
                            f"{class_names[index]} {round(label[-2].item(), 2)}",
                            color)
                    else:
                        annotator.box_label(
                            label[0:4],
                            f"{class_names[index]} {round(label[-2].item(), 2)}",
                            (0, 0, 0))  # Default color if index is out of range
                # Save captured frame to a file
                self.ann_saved_path = os.path.join('Image Pred', self.img_path)
                cv2.imwrite(self.ann_saved_path, annotator.im)
                self.update_flag = False  # Stop the`update` method from running
                Clock.schedule_once(self.pred_display)  # Reset camera texture after 5 seconds
            except Exception as e:
                print(f"Error while drawing bounding: {e}")
        except Exception as e:
            print(f"Error while verifing image: {e}")

    def pred_display(self, dt):
        try:
            start = time.time()
            # Display the annotated image or "new_im.jpg" in the camera widget for 2 seconds
            img_1 = cv2.imread(self.ann_saved_path)
            img = cv2.flip(img_1, 0)  # Flip the image vertically
            img_texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
            img_texture.blit_buffer(img.tobytes(), colorfmt='bgr', bufferfmt='ubyte')
            self.camera.texture = img_texture
            self.verification_text.text = f"{self.results[1]}"
            try:
                self.overload = False
                self.person = json.loads(self.results[1])["person"]
                if self.person > 2:
                    self.overload = True
                    threading.Thread(target=self.alarm, args=(f"Overload alert : {self.person} persons Found",)).start()
        
            except Exception as e:
                print(f"Error while sending alarm: {e}")
                self.person = 0
            gray_img = cv2.cvtColor(cv2.resize(img_1, (224, 224)), cv2.COLOR_BGR2GRAY)
            normalized_img = np.array(gray_img.tolist(), dtype=np.float32) / 255.0

            self.data = [
                {"Person": self.person,
                "Weight": f"{self.person * self.weight * 1.2} kg",
                "Overload": self.overload,
                "Orginal shape": self.org_shape,
                "Resized shape": self.resize_shape,
                "Date" : self.date,
                "Processing Time": f"{(time.time() - start):.4f} sec"
                },
                normalized_img.tolist()
            ]

            self.verification_text.text = json.dumps(self.data[0], indent=4)
            self.data[0]["Img_path"] = self.ann_saved_path
            threading.Thread(target=self.write_json).start()

            Clock.schedule_once(self.reset_camera_texture, 6)  # Reset camera texture after 5 seconds
        except Exception as e:
            print(f"Error while displaying the annotated image: {e}")
        

    def reset_camera_texture(self, dt):
        try:
            # Reset the camera texture to the live video feed
            self.camera.texture = None
            self.verification_text.text = ""
            self.Led.background_color = get_color_from_hex('#000000')  # Green color
            self.update_flag = True  # Resume the update process
            pin13.write(0)
        except Exception as e:
            print(f"Error while resetting the camera texture: {e}")

    def alarm(self, sentence):
        try:
            self.Led.background_color = get_color_from_hex('#FF0000')
            engine = pyttsx4.init()
            engine.say(sentence)
            engine.runAndWait()
            engine.stop()
            pin13.write(1)
        except Exception as e:
            print(f"Error while sending alarm: {e}")    

    def write_json(self):
        try:
            with open("data.json", 'r+') as file:
                dt = json.load(file)
                dt["data"].insert(0, self.data[0])
                file.seek(0)
                json.dump(dt, file, indent=2)
                file.truncate()
        except Exception as e:
            print(f"Error while writing json: {e}")
    def cont_run(self, instance):
        Clock.schedule_interval(self.verify, 5)  # Schedule the verification process to run every 5 seconds

if __name__ == '__main__':
    CamApp().run()