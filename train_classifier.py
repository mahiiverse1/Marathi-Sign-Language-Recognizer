import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

def pad_sequences(data, target_length):
    padded_data = []
    for sequence in data:
        if len(sequence) >= target_length:
            padded_data.append(sequence[:target_length])
        else:
            padded_data.append(np.pad(sequence, (0, target_length - len(sequence)), mode='constant'))
    return np.asarray(padded_data)

def train_and_save_model(data_path, model_save_path='my_model.h5', target_length=100):
    # Load data
    data_dict = pickle.load(open(data_path, 'rb'))
    
    # Ensure consistent shapes in the data
    data = pad_sequences(data_dict['data'], target_length=target_length)
    labels = np.asarray(data_dict['labels'])

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42)

    # Reshape data for CNN
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # CNN model architecture
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(target_length, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(len(np.unique(labels)), activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1, mode='max')
    model_checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', verbose=1, save_best_only=True)

    # Train the model with callbacks
    history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test), callbacks=[early_stopping, model_checkpoint])

    # Evaluate the model
    _, accuracy = model.evaluate(x_test, y_test)
    print('Final Accuracy: {}%'.format(accuracy * 100))

    return history

# Example usage
history = train_and_save_model('./data.pickle', 'my_model.h5', target_length=100)
