#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 01:07:32 2020
Last Updated: Tue Jul 6 23:11:21 2021
Author: Tyler Pruitt
"""

# Import packages
import matplotlib.pyplot as plt
import os
import re
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# Examine structure of IMDB dataset
datasetDirectory = os.path.join('./aclImdb')

# Print out structure of dataset directory aclImdb
print('Dataset directory:')
print(os.listdir(datasetDirectory), end='\n\n')

trainDirectory = os.path.join(datasetDirectory, 'train')

# Print out structure of train directory
print('Train directory:')
print(os.listdir(trainDirectory), end='\n\n')

# Output a sample positive review
sampleReview = os.path.join(trainDirectory, 'pos/1181_9.txt')

with open(sampleReview) as review:
    print(review.read())

# Load the dataset

# For working with text_dataset_from_database in binary classification we will need 2
# folders in our directory: class_a and class_b

# Use tf.data.Dataset and make an 80:20 split of training data for validation set
# by using validation_split

# Create training dataset of 20,000 reviews
RawTrainDataset = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=32, 
    validation_split=0.2, 
    subset='training', 
    seed=42)

# Using 20,000 reviews for training, 5,000 reviews for validation

# We can also print out a few examples by iterating over the dataset in tf.data
for textBatch, labelBatch in RawTrainDataset.take(1):
    for i in range(3):
        print('Review', textBatch.numpy()[i])
        print('Label', labelBatch.numpy()[i], end='\n\n')

# Let's convert the labels 0,1 to pos,neg
print('Label 0 corresponds to', RawTrainDataset.class_names[0])
print('Label 1 corresponds to', RawTrainDataset.class_names[1], end='\n\n')

# Create validation set
RawValuationDataset = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=32,
    validation_split=0.2,
    subset='validation',
    seed=42)

# Create test set
RawTestDataset = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=32)

# Preprocess the dataset for training using preprocessing.TextVectorization layer
# (1) Standardize
# (2) Tokenize
# (3) Vectorize

# Write a custom standardization function that removes all HTML code in the reviews

def customStandardization(inputData):
    lowercase = tf.strings.lower(inputData)
    strippedHtml = tf.strings.regex_replace(lowercase, pattern='<br />', rewrite=' ')
    
    return tf.strings.regex_replace(strippedHtml, pattern='[%s]' % re.escape(string.punctuation), rewrite='')

# Create TextVectorization layer; make output int so each token has a unique int;
# Set sequence length for review

maxFeatures = 10000
sequenceLength= 250

vectorizeLayer = TextVectorization(
    standardize=customStandardization,
    max_tokens=maxFeatures,
    output_mode='int',
    output_sequence_length=sequenceLength)

# Call adapt to fit preprocessing to layer of dataset, makes model build index of strings to integers

# Make text-only dataset, then call adapt
TrainText = RawTrainDataset.map(lambda x, y: x)

vectorizeLayer.adapt(TrainText)

# Define a function to see result of layer to preprocess data
def vectorizeText(text, label):
    text = tf.expand_dims(text, axis=-1)
    return vectorizeLayer(text), label

# Output a batch of 32 reviews and labels from the dataset
textBranch, labelBranch = next(iter(RawTestDataset))
firstReview, firstLabel = textBatch[0], labelBatch[0]

print('')
print('Review', firstReview)
print('Label', RawTrainDataset.class_names[firstLabel])
print('Vectorized review', vectorizeText(firstReview, firstLabel), end='\n\n')

# Here we see that each token has been replaced by an integer
# to see the relationship between integers and words let's print out a few using .get_vocabulary()
print('1287 --->', vectorizeLayer.get_vocabulary()[1287])
print('313 --->', vectorizeLayer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorizeLayer.get_vocabulary())), end='\n\n')

# Now apply TextVectorization layer to train set, validation set, and test set
TrainDataset = RawTrainDataset.map(vectorizeText)
ValidationDataset = RawValuationDataset.map(vectorizeText)
TestDataset = RawTestDataset.map(vectorizeText)

# Configure the preprocessed dataset for performance metrics

# Use .cache() to prevent bottlenecking of data while training
# Use .prefetch() to overlap data preprocessing and model execution in training

AUTOTUNE = tf.data.experimental.AUTOTUNE

TrainDataset = TrainDataset.cache().prefetch(buffer_size=AUTOTUNE)
ValidationDataset = ValidationDataset.cache().prefetch(buffer_size=AUTOTUNE)
TestDataset = TestDataset.cache().prefetch(buffer_size=AUTOTUNE)

# Create the model (neural network)

model = tf.keras.Sequential([
    layers.Embedding(maxFeatures+1, 16),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1)])

# Print out summary of the model
model.summary()
print('\n')

# Choose loss function and optimizer

# For binary classification we will use losses.BinaryCrossentropy function

# Compile the model

model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

# Train the model for 10 epochs
# Save it as history for later analysis

history = model.fit(TrainDataset, validation_data=ValidationDataset, epochs=10)

# Evaluate the model on the test set

loss, accuracy = model.evaluate(TestDataset)

print('Loss:', loss)
print('Accuracy:', accuracy, end='\n\n')

# Create a plot of the model's accuracy and loss over time
historyDict = history.history

print(historyDict.keys(), end='\n\n')

binaryAccuracy = historyDict['binary_accuracy']
validationAccuracy = historyDict['val_binary_accuracy']
loss = historyDict['loss']
validationLoss = historyDict['val_loss']

epochs = range(1, len(binaryAccuracy)+1)

# Plot loss for training and validation
plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, validationLoss, 'b', label='Validation loss')

plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Plot accuracy for training and validation
plt.plot(epochs, binaryAccuracy, 'bo', label='Training accuracy')

plt.plot(epochs, validationAccuracy, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Adapt the model to be able to predict using raw text

model = tf.keras.Sequential([
    vectorizeLayer,
    model,
    layers.Activation('sigmoid')])

# Compile export model

model.compile(loss=losses.BinaryCrossentropy(from_logits=False),
                     optimizer='adam',
                     metrics=['accuracy'])

# Test export model with test dataset

testLoss, testAccuracy = model.evaluate(RawTestDataset)

print('Test loss:', testLoss)
print('Test accuracy:', testAccuracy, end='\n\n')

# Have the export model make predictions about new reviews
reviews = ['The movie was great!', 'The movie was fine.', 'The movie was awful.']

print(model.predict(reviews))

# Output the export model by saving it
model.save("model")

# Save the model in the h5 file format for the webapp
model.save("model.h5")
