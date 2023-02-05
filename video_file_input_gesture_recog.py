#!/usr/bin/env python
# coding: utf-8


# In[ ]:


#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



#@markdown We implemented some functions to visualize the gesture recognition results. <br/> Run the following cell to activate the functions.
from matplotlib import pyplot as plt
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.bottom': False,
    'xtick.labelbottom': False,
    'xtick.bottom': False,
    'ytick.labelleft': False,
    'ytick.left': False,
    'xtick.labeltop': False,
    'xtick.top': False,
    'ytick.labelright': False,
    'ytick.right': False
})

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def display_one_image(image, title, subplot, titlesize=16):
    """Displays one image along with the predicted category name and score."""
    plt.subplot(*subplot)
    plt.imshow(image)
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize), color='black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)


def display_batch_of_images_with_gestures_and_hand_landmarks(images, results):
    """Displays a batch of images with the gesture category and its score along with the hand landmarks."""
    # Images and labels.
    images = [image.numpy_view() for image in images]
    gestures = [top_gesture for (top_gesture, _) in results]
    multi_hand_landmarks_list = [multi_hand_landmarks for (_, multi_hand_landmarks) in results]

    # Auto-squaring: this will drop data that does not fit into square or square-ish rectangle.
    rows = int(math.sqrt(len(images)))
    cols = len(images) // rows

    # Size and spacing.
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot=(rows,cols, 1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))
    else:
        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))

    # Display gestures and hand landmarks.
    for i, (image, gestures) in enumerate(zip(images[:rows*cols], gestures[:rows*cols])):
        title = f"{gestures.category_name} ({gestures.score:.2f})"
        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols) * 40 + 3
        annotated_image = image.copy()

        for hand_landmarks in multi_hand_landmarks_list[i]:
          hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
          hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
          ])

          mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        subplot = display_one_image(annotated_image, title, subplot, titlesize=dynamic_titlesize)

    # Layout.
    plt.tight_layout()
    plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()


# ## Download test images
# 
# Let's grab some test images that we'll use later. The images ([1](https://pixabay.com/photos/idea-pointing-raise-hand-raise-3082824/), [2](https://pixabay.com/photos/thumbs-up-happy-positive-woman-2649310/), [3](https://pixabay.com/photos/epidemic-disease-coronavirus-5082474/), [4](https://pixabay.com/photos/thumbs-down-disapprove-gesture-6744094/)) are from Pixabay.

# In[6]:


import urllib

IMAGE_FILENAMES = ['thumbs_down.jpg', 'victory.jpg', 'thumbs_up.jpg', 'pointing_up.jpg']

for name in IMAGE_FILENAMES:
  url = f'https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/{name}'
  urllib.request.urlretrieve(url, name)


# Optionally, you can upload your own image. If you want to do so, uncomment and run the cell below.

# In[7]:


# from google.colab import files
# uploaded = files.upload()

# for filename in uploaded:
#   content = uploaded[filename]
#   with open(filename, 'wb') as f:
#     f.write(content)
# IMAGE_FILENAMES = list(uploaded.keys())

# print('Uploaded files:', IMAGE_FILENAMES)


# Then let's check out the images.

# ## Running inference and visualizing the results
# 
# Here are the steps to run gesture recognizer using MediaPipe.
# 
# Check out the [MediaPipe documentation](https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer/python) to learn more about configuration options that this solution supports.
# 
# *Note: Gesture Recognizer also returns the hand landmark it detects from the image, together with other useful information such as whether the hand(s) detected are left hand or right hand.*

# In[ ]:


import cv2
def check_again2():
  
  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_hands = mp.solutions.hands

  cap = cv2.VideoCapture('/Users/s/dacon/open/train/TRAIN_302.mp4')
  with mp_hands.Hands(
      model_complexity=0,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = hands.process(image)

      # Draw the hand annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
          mp_drawing.draw_landmarks(
              image,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())
      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
      if cv2.waitKey(5) & 0xFF == 27:
        break
  cap.release()

def check_again(video_file_path):
  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_hands = mp.solutions.hands
  cap = cv2.VideoCapture(video_file_path)
  
  with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.") 
        break;
 
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = hands.process(image)

      # Draw the hand annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      print(results.multi_hand_landmarks)

      if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
          mp_drawing.draw_landmarks(
              image,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())
      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
      if cv2.waitKey(5) & 0xFF == 27:
        break

# In[34]:
def do_process(video_file_path):
  print(video_file_path)

  import math
  import cv2

  import mediapipe as mp
  from mediapipe.tasks import python
  from mediapipe.tasks.python import vision
  from mediapipe.python import packet_creator
  from mediapipe.python import packet_getter

  from mediapipe.python._framework_bindings import calculator_graph
  from mediapipe.python._framework_bindings import image
  from mediapipe.python._framework_bindings import image_frame
  from mediapipe.python._framework_bindings import packet

  CalculatorGraph = calculator_graph.CalculatorGraph
  Image = image.Image
  ImageFormat = image_frame.ImageFormat
  ImageFrame = image_frame.ImageFrame

  # STEP 2: Create an GestureRecognizer object.
  VisionRunningMode = mp.tasks.vision.RunningMode
  base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
  options = vision.GestureRecognizerOptions(base_options=base_options, running_mode=VisionRunningMode.VIDEO )
  recognizer = vision.GestureRecognizer.create_from_options(options)

  results = []
  cap = cv2.VideoCapture(video_file_path)
  fps = cv2.CAP_PROP_FPS
  calc_timestamps = [0.0]

  res_landmark = []
  res_not_found_count = 0
  
  ret = True
  while ret:
    ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
    if ret:      
      ts = cap.get(cv2.CAP_PROP_POS_MSEC)      
      cts = calc_timestamps[-1] + 1000/fps
      #print(abs(ts-cts))
      recognition_result =recognizer.recognize_for_video( mp.Image(image_format=ImageFormat.SRGB, data=img), int(ts))
      
      try : 
        top_gesture = recognition_result.gestures[0][0]
        
        hand_landmarks = top_gesture.category_name
        #print(hand_landmarks)
        results.append(top_gesture)
        res_landmark.append(hand_landmarks)
      except IndexError: 
        res_landmark.append('-')
        res_not_found_count += 1
        continue 
        
      
  cap.release()
  cv2.destroyAllWindows
    
  print("====RESULT ===== ")
  check_again(video_file_path)
  if res_landmark.count('-') == len(res_landmark) :
    print("None!")
    check_again(video_file_path)
  else: 
    #print(results)
    print(res_landmark)
    #print(res_not_found_count)
      #print(res_landmark.count('Open_Palm'))
      

def main():
    file_index = ''
    for i in range(302, 303): 
        if i < 10 : file_index = '00'+str(i)
        elif i> 10 and i<100: file_index= '0'+str(i)
        else: file_index = str(i)
        video_file_path = './open/train/TRAIN_'+file_index + '.mp4'
        
        do_process(video_file_path)
if __name__ == "__main__":
  main()
 
    #print(res_not_found_count)
    #print(res_landmark.count('Open_Palm'))
    #print(results)
    
#  display_batch_of_images_with_gestures_and_hand_landmarks(images, results)

 
