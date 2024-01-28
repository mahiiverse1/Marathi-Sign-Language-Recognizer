import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def pad_sequences(data, target_length):
    padded_data = []
    for sequence in data:
        if len(sequence) >= target_length:
            padded_data.append(sequence[:target_length])
        else:
            padded_data.append(np.pad(sequence, (0, target_length - len(sequence))))
    return np.asarray(padded_data)

def train_and_save_model(data_path, model_save_path='model.p', target_length=100):
    # Load data
    data_dict = pickle.load(open(data_path, 'rb'))
    
    # Ensure consistent shapes in the data
    data = pad_sequences(data_dict['data'], target_length=target_length)
    labels = np.asarray(data_dict['labels'])

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42)

    # Create and train the model
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    # Evaluate the model
    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)
    print('{}% of samples were classified correctly!'.format(score * 100))

    # Save the trained model
    with open(model_save_path, 'wb') as f:
        pickle.dump({'model': model}, f)

# Example usage
train_and_save_model('./data.pickle', 'model.p', target_length=100)
