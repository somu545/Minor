# import numpy as np
# import pandas as pd
# from keras.utils import to_categorical

# # Function to load FER2013 dataset
# def load_fer2013(csv_file='fer2013.csv'):
#     """
#     Load the FER2013 dataset from a CSV file.
#     Args:
#         csv_file (str): Path to the CSV file containing the dataset.
#     Returns:
#         faces (numpy array): The images in the dataset, normalized.
#         emotions (numpy array): The emotion labels in one-hot encoded format.
#     """
#     # Load the CSV file using pandas
#     data = pd.read_csv(csv_file)
    
#     faces = []
#     emotions = []

#     # Loop through each row in the dataset
#     for index, row in data.iterrows():
#         # Convert the 'pixels' string into a numpy array of integers
#         pixels = np.fromstring(row['pixels'], dtype=int, sep=' ')
        
#         # Reshape the pixels into a 48x48 image and append to faces list
#         faces.append(pixels.reshape(48, 48, 1))
        
#         # Append the emotion (label) to the emotions list
#         emotions.append(row['emotion'])
    
#     # Convert faces and emotions lists to numpy arrays
#     faces = np.array(faces, dtype='float32')
#     emotions = np.array(emotions)
    
#     # Normalize the pixel values to be between 0 and 1
#     faces = faces / 255.0
    
#     # One-hot encode the emotion labels (7 categories)
#     emotions = to_categorical(emotions, num_classes=7)
    
#     return faces, emotions

# # Function to preprocess input images (optional, for further processing like augmentation)
# def preprocess_input(faces):
#     """
#     Preprocess the input images (this can include normalization, resizing, etc.).
#     Args:
#         faces (numpy array): The images in the dataset.
#     Returns:
#         numpy array: The preprocessed images.
#     """
#     # In your case, normalization was already handled in load_fer2013, so this can remain empty
#     return faces.astype('float32')  # Ensure correct dtype for neural network compatibility
