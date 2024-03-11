import os
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from gtts import gTTS  # Import gTTS library for text-to-speech
import tempfile
import pygame

# Function to pad sequences
def pad_sequences(data, target_length):
    padded_data = []
    for sequence in data:
        if len(sequence) >= target_length:
            padded_data.append(sequence[:target_length])
        else:
            padded_data.append(np.pad(sequence, (0, target_length - len(sequence)), mode='constant', constant_values=0))
    return np.asarray(padded_data, dtype=np.float32)

# Function to play audio
def play_audio(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

def on_speak_click(text):
    print("Text to be spoken:", text)  # Check if the correct text is retrieved
    # Create a temporary file to store the generated speech
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        # Convert text to speech and save it to the temporary file
        tts = gTTS(text=text, lang='mr')  # 'mr' is the language code for Marathi
        tts.save(f"{fp.name}.mp3")
        print("Saved speech to:", f"{fp.name}.mp3")  # Check if the speech is saved
        # Play the generated speech using pygame
        play_audio(f"{fp.name}.mp3")

try:
    # Load the trained model
    model = load_model('./my_model.h5')

    # Check if the webcam is opened successfully
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Couldn't open the webcam.")
        exit()

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    # Labels dictionary
    labels_dict = {0: 'अ', 1: 'आ', 2: 'इ', 3: 'ई', 4: 'उ', 5: 'ऊ', 6: 'ए', 7: 'ऐ', 8: 'ओ', 9: 'औ', 10: 'क', 11: 'ख',
                   12: 'ग', 13: 'घ', 14: 'च', 15: 'छ', 16: 'ज', 17: 'झ', 18: 'ट', 19: 'ठ', 20: 'ड', 21: 'ढ', 22: 'ण',
                   23: 'त', 24: 'थ', 25: 'द', 26: 'ध', 27: 'न', 28: 'प', 29: 'फ', 30: 'ब', 31: 'भ', 32: 'म', 33: 'य',
                   34: 'र', 35: 'ल', 36: 'व', 37: 'श', 38: 'स', 39: 'ह', 40: 'ळ', 41: 'क्ष', 42: 'ज्ञ', 43: 'नमस्कार',
                   44: 'ठीक आहे', 45: 'धन्यवाद', 46: 'तुझ्यावर प्रेम आहे', 47: 'नमस्ते', 48: 'सर्व उत्तम', 49: 'नापसंत',
                   50: 'थांबा', 51: 'पराभूत', 52: 'स्मित करणे', 53: 'मला कॉल करा', 54: 'काळजी', 55: 'मैत्री', 56: 'उभे रहा',
                   57: 'लग्न'}

    # Expected characters range
    expected_characters = list(labels_dict.values())

    # Create Tkinter window
    window = tk.Tk()
    window.title("Devanagari Hand Gesture Recognition")
    window.configure(bg='#f0f0f0')  # Set background color

    # Define fonts
    title_font = ('Noto Sans Devanagari', 30, 'bold')
    label_font = ('Noto Sans Devanagari', 25)  # Increase font size

    # Create a frame for the video feed and predicted character
    frame = tk.Frame(window, bg='#51829B', width=1600, height=900)  # Set width and height to match canvas
    frame.pack(padx=20, pady=20)

    # Create a Label for the title
    title_label = tk.Label(frame, text="Real Time Marathi Sign Language Recognizer", font=title_font, fg="blue", bg='#f0f0f0')
    title_label.pack(pady=10)

    # Create a canvas to display the video feed
    canvas_width = int(window.winfo_screenwidth() * 0.7)  # Set canvas width to 70% of screen width
    canvas = tk.Canvas(frame, width=canvas_width, height=550, bg='white')  # Set initial height to 550
    canvas.pack(expand=True, fill=tk.X)  # Use expand and fill to stretch canvas horizontally

    # Create a Label for displaying the predicted character with increased font size
    text_label = tk.Label(frame, text="", font=label_font, fg="blue", bg='#f0f0f0')
    text_label.pack(pady=20)  # Increase pady for spacing

    # Create a button for converting text to speech
    speak_button = tk.Button(frame, text="Text-to-Speech", command=lambda: on_speak_click(text_label.cget("text")), font=('Noto Sans Devanagari', 20), bg="#DBCC95", fg="black", relief=tk.RAISED, padx=10, pady=5)
    speak_button.pack(pady=(0, 10))  # Add margin after the button


    # Function to maintain aspect ratio of the webcam feed
    def maintain_aspect_ratio(image, width):
        h, w = image.shape[:2]
        aspect_ratio = w / h
        return cv2.resize(image, (width, int(width / aspect_ratio)))

    # Function to update the frame
    def update_frame():
        try:
            data_aux = []
            x_ = []
            y_ = []

            ret, frame = cap.read()

            if not ret:
                print("Error reading frame from the webcam.")
                return

            frame = maintain_aspect_ratio(frame, canvas_width)  # Maintain aspect ratio while resizing

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

                    # Make prediction
                    prediction = model.predict(padded_data)

                    # Check if the prediction is successful
                    if prediction is not None and len(prediction) > 0:
                        predicted_class = np.argmax(prediction)  # Get the index of the highest probability
                        confidence = np.max(prediction)  # Get the confidence score
                        print("Predicted class:", predicted_class, "Confidence:", confidence)
                        
                        # Update the Label with the predicted character or error message
                        if confidence < 0.5:
                            text_label.config(text="Hand gesture not predicted", fg="red")
                        elif predicted_class in labels_dict:
                            predicted_character = labels_dict[predicted_class]
                            text_label.config(text=predicted_character, fg="blue")
                        else:
                            text_label.config(text="Error: Unrecognized character", fg="red")
                    else:
                        text_label.config(text="Error: Prediction unsuccessful", fg="red")
            else:
                text_label.config(text="Error: Hand landmarks not detected", fg="red")

            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            canvas.img = img
            # Center the image in canvas
            canvas.create_image((canvas_width - img.width()) // 2, 0, anchor=tk.NW, image=img)

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

except Exception as e:
    print("Error:", e)
