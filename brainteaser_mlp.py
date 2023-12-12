"""

    Created By: Ashutosh Mishra, Pauline Korukundo and Rohan Boorugu
    
    Date: 11 December 2023

    Final Project Source Code for Multi Layer Perceptron

    This script allows to train a neural network on the brainteaser train 
    dataset and predict the right answers for Brain Teaser Questions

     
"""


import numpy as np
import pandas as pd
import gensim.downloader as api
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

#Function to tokenize data
def preprocess_function(examples):
    first_sentences = [[context] * 4 for context in examples["question"]]
    second_sentences = [context for context in examples["choice_list"]]
    
    first_sentences = sum(first_sentences,[])
    second_sentences = sum(second_sentences,[])
    df = pd.DataFrame({"context": first_sentences, "choice": second_sentences})
    df['merged'] = df['context'].astype(str) + ' ' + df['choice']
    return df

#Function to train and test the MLP Classifier
def mlp(train_file):
    # Load the Word2Vec embeddings
    word_vectors = api.load('word2vec-google-news-300') #load or download word vectors using gensim downloader
    print(f'Loading Vector Embeddings: {word_vectors}')
    print('')

    # Compute the dimension of the vector embeddings
    max_len = len(word_vectors['the'])

    read_file = np.load(train_file, allow_pickle = True)                  #Read input file

    df = pd.DataFrame(read_file.tolist())                                 #Convert Numpy to DataFrame
        
    # Get train, test and validation set
    train_df, test_df = train_test_split(df, test_size=0.2, random_state= 42)   #Split train to test split

    x = preprocess_function(train_df)


    df_X = x['merged']
    df_Y = train_df['choice_order']

    z = df_Y.tolist()
    flattened_list = [item for sublist in z for item in sublist]
    df_y = pd.DataFrame(flattened_list)
    df_labels = df_y.replace(0,'answer')
    df_labels = df_labels.replace(1,'distractor')
    df_labels = df_labels.replace(2,'distractor')
    df_Y = df_labels.replace(3,'distractor')

    print("Computing Word Embeddings for Train Data...")

    X_train = np.array([np.mean([word_vectors[word] for word in text.split() if word in word_vectors], axis=0) for text in x['merged']])
    # Get the class labels 
    Y_train = df_Y[0].to_list()      

    print("Training the MLP Average Model...")
    # Train the MLP Model Using ReLU activation
    MLP_model = MLPClassifier(
        hidden_layer_sizes=(150,50,40,40),
        max_iter=500, 
        activation='relu',
        verbose=True,
        random_state=1
        )

    #fit the model
    MLP_model.fit(X_train, Y_train)

    x_test = preprocess_function(test_df)
    df_X_test = x_test['merged']  
    y_test = test_df['choice_order']

    z = y_test.tolist()
    flattened_list = [item for sublist in z for item in sublist]
    df_y = pd.DataFrame(flattened_list)
    df_labels = df_y.replace(0,'answer')
    df_labels = df_labels.replace(1,'distractor')
    df_labels = df_labels.replace(2,'distractor')
    Y_test = df_labels.replace(3,'distractor')

    #TEST
    print("")
    print("Computing Word Embeddings for Test Data...")
    X_test = np.array([np.mean([word_vectors[word] for word in text.split() if word in word_vectors], axis=0) for text in x_test['merged']])
    

    print(f'Running MLP Classifier on Test Embeddings ')
    Y_pred = MLP_model.predict(X_test)     

    #Y_pred_prob = MLP_model.predict_proba(X_test) * 100
    # Compute the confusion matrix 
    conf_mat = confusion_matrix(Y_test, Y_pred)
    print(f'\nConfusion Matrix:\n {conf_mat}\n')

    target_names = ['class 0', 'class 1']
    print(classification_report(Y_test, Y_pred, target_names=target_names, digits=2))


#Enter the file location
train_file =  "SP-train.npy"    #Path of Train File

#invoke the mlp function
mlp(train_file)