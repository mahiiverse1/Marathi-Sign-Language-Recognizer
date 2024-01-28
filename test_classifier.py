import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

def pad_sequences(data, target_length):
    padded_data = []
    for sequence in data:
        if len(sequence) >= target_length:
            padded_data.append(sequence[:target_length])
        else:
            padded_data.append(np.pad(sequence, (0, target_length - len(sequence)), mode='constant', constant_values=0))
    return np.asarray(padded_data, dtype=np.float32)

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Check if the webcam is opened successfully
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Couldn't open the webcam.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0:'अ' , 1:'आ' , 2:'इ' , 3:'ई' , 4:'उ' , 5:'ऊ' , 6:'ए' , 7:'ऐ' , 8:'ओ' , 9:'औ' , 10:'क' , 11:'ख' , 12:'ग' , 13:'घ' , 14:'च' , 15:'छ' , 16:'ज' , 17:'झ' , 18:'ट' , 19:'ठ' , 20:'ड' , 21:'ढ' , 22:'ण' , 23:'त' , 24:'थ' , 25:'द' , 26:'ध' , 27:'न' , 28:'प' , 29:'फ' , 30:'ब' , 31:'भ' , 32:'म' , 33:'य' , 34:'र' , 35:'ल' , 36:'व' , 37:'श' , 38:'स' , 39:'ह' , 40:'ळ' , 41:'क्ष' , 42:'ज्ञ'}

# Create Tkinter window
window = tk.Tk()
window.title("Devanagari Hand Gesture Recognition")

font = ('Noto Sans Devanagari', 12)

# Create a canvas to display the video feed
canvas = tk.Canvas(window, width=640, height=480)
canvas.pack()

# Create a Label for displaying the predicted character
text_label = tk.Label(window, text="", font=font)
text_label.place(x=10, y=10)  # Adjust the position as needed

# Declare global variables
x1, y1, x2, y2 = 0, 0, 0, 0

def update_frame():
    global x1, y1, x2, y2  # Declare the variables as global
    try:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        if not ret:
            print("Error reading frame from the webcam.")
            return

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # Pad the input sequence to match the expected length
                padded_data = pad_sequences([data_aux], target_length=100)

                print("Padded Data Shape:", padded_data.shape)  # Add this line for debugging

                prediction = model.predict(padded_data)

                print("Prediction:", prediction)  # Add this line for debugging

                # Check if the prediction is successful
                if prediction is not None and len(prediction) > 0:
                    predicted_class = int(prediction[0])
                    predicted_character = labels_dict[predicted_class]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                                cv2.LINE_AA)

                    # Update the Label with the predicted character
                    text_label.config(text=predicted_character)

        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        canvas.img = img
        canvas.create_image(0, 0, anchor=tk.NW, image=img)

        # Call the update_frame function after a delay
        window.after(10, update_frame)

    except Exception as e:
        import traceback
        print("Error in update_frame:", e)
        traceback.print_exc()  # Print the full traceback

# Call the update_frame function to start displaying the video feed
update_frame()

# Start the Tkinter event loop
window.mainloop()
