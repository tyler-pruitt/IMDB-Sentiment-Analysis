#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 01:07:32 2020

@author: tylerpruitt
"""

#import packages
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

#download and check structure of IMDB dataset
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file(fname="aclImdb_v1.tar.gz", origin=url, untar=True, cache_dir='.', cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

#print out structure of dataset directory aclImdb
print('Dataset directory:')
print(os.listdir(dataset_dir))
print('')

train_dir = os.path.join(dataset_dir, 'train')

#print out structure of train directory
print('Train directory:')
print(os.listdir(train_dir))
print('')

#output a sample positive review
sample_review = os.path.join(train_dir, 'pos/1181_9.txt')

with open(sample_review) as review:
    print(review.read())

#load the dataset

#for working with text_dataset_from_database in binary classification we will need 2
#folders in our directory: class_a and class_b

#remove the remaining folders
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

#use tf.data.Dataset and make an 80:20 split of training data for validation set
#by using validation_split

#create training dataset of 20,000 reviews
raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=32, 
    validation_split=0.2, 
    subset='training', 
    seed=42)

#using 20,000 reviews for training, 5,000 reviews for validation

#we can also print out a few examples by iterating over the dataset in tf.data
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(3):
        print('Review', text_batch.numpy()[i])
        print('Label', label_batch.numpy()[i])
        print('')

#let's convert the labels 0,1 to pos,neg
print('Label 0 corresponds to', raw_train_ds.class_names[0])
print('Label 1 corresponds to', raw_train_ds.class_names[1])
print('')

#create validation set
raw_valuation_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=32,
    validation_split=0.2,
    subset='validation',
    seed=42)

#create test set
raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=32)

#preprocess the dataset for training using preprocessing.TextVectorization layer
#(1) standardize
#(2) tokenize
#(3) vectorize

#write a custom standardization function that removes all HTML code in the reviews

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, pattern='<br />', rewrite=' ')
    
    return tf.strings.regex_replace(stripped_html, pattern='[%s]' % re.escape(string.punctuation), rewrite='')

#create TextVectorization layer; make output int so each token has a unique int;
#set sequence length for review

max_features = 10000
sequence_length= 250

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

#call adapt to fit preprocessing to layer of dataset, makes model build index of strings to integers

#make text-only dataset, then call adapt
train_text = raw_train_ds.map(lambda x, y: x)

vectorize_layer.adapt(train_text)

#define a function to see result of layer to preprocess data
def vectorize_text(text, label):
    text = tf.expand_dims(text, axis=-1)
    return vectorize_layer(text), label

#output a batch of 32 reviews and labels from the dataset
text_branch, label_branch = next(iter(raw_test_ds))
first_review, first_label = text_batch[0], label_batch[0]

print('')
print('Review', first_review)
print('Label', raw_train_ds.class_names[first_label])
print('Vectorized review', vectorize_text(first_review, first_label))
print('')

#here we see that each token has been replaced by an integer
#to see the relationship between integers and words let's print out a few using .get_vocabulary()
print('1287 --->', vectorize_layer.get_vocabulary()[1287])
print('313 --->', vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))
print('')

#now apply TextVectorization layer to train set, validation set, and test set
train_ds = raw_train_ds.map(vectorize_text)
validation_ds = raw_valuation_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

#configure the preprocessed dataset for performance metrics

#use .cache() to prevent bottlenecking of data while training
#use .prefetch() to overlap data preprocessing and model execution in training

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

#create the model (neural network)

model = tf.keras.Sequential([
    layers.Embedding(max_features+1, 16),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1)])

#print out summary of the model
model.summary()
print('')

#choose loss function and optimizer

#for binary classification we will use losses.BinaryCrossentropy function

#compile the model

model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

#train the model for 10 epochs
#save it as history for later analysis

history = model.fit(train_ds, validation_data=validation_ds, epochs=10)

#evaluate the model on the test set

loss, accuracy = model.evaluate(test_ds)

print('Loss:', loss)
print('Accuracy:', accuracy)
print('')

#create a plot of the model's accuracy and loss over time
history_dict = history.history

print(history_dict.keys())
print('')

bin_acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(bin_acc)+1)

#plot loss for training and validation
plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#plot accuracy for training and validation
plt.plot(epochs, bin_acc, 'bo', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

#export the model to be able to predict using raw text

export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')])

#compile export model

export_model.compile(loss=losses.BinaryCrossentropy(from_logits=False),
                     optimizer='adam',
                     metrics=['accuracy'])

#test export model with test dataset

test_loss, test_accuracy = export_model.evaluate(raw_test_ds)

print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)
print('')

#have the export model make predictions about new reviews
reviews = ['The movie was great!', 'The movie was fine.', 'The movie was awful.']

print(export_model.predict(reviews))






