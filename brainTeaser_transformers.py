"""

    Created By: Ashutosh Mishra, Pauline Korukundo and Rohan Boorugu
    
    Date: 11 December 2023

    Final Project Source Code for Transformers

    This script allows to finetune multiple transformer models on the train dataset and predict the right answers for Brain Teaser Questions

    You can use the following models as it is a generalized one for transformers approach. Please note that for complex models like the 
     
"""
import os                                                                          #For os related functions like Greeting user
import pandas as pd                                                                #For using the datasets
import numpy as np                                                                 #For using numpy operations
from datasets import Dataset, DatasetDict                                          #For spliting the datasets and creating vector objects
from sklearn.model_selection import train_test_split                               #For splitting the datasets
from transformers import AutoTokenizer, DebertaTokenizer                           #For Tokenizing the data           
from transformers import AutoModelForMultipleChoice, TFDebertaV2ForMultipleChoice  #Importing AutomodelforMultipleChoice to load the models for MCQ
from transformers import TFAutoModelForMultipleChoice, TFDebertaV2ForMultipleChoice, TFAlbertForMultipleChoice
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import create_optimizer
from typing import Optional, Union
import tensorflow as tf
import evaluate
from transformers.keras_callbacks import KerasMetricCallback
from sklearn.metrics import classification_report

accuracy = evaluate.load("accuracy")


#Please change model name based on use case
model_name = "bert-base-uncased"
#model_name = "roberta-base"
#model_name = "albert-base-v2"

model = TFAutoModelForMultipleChoice.from_pretrained(model_name)    
#Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)    
#Change tokenizer based on use-case

#enter the file location for the dataset
fileInput = "SP-train.npy"      #Path of Train File


#Custom Data Collector for MCQ Dataset  #Using from the transformers training module
@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="tf",
        )

        batch = {k: tf.reshape(v, (batch_size, num_choices, -1)) for k, v in batch.items()}
        batch["labels"] = tf.convert_to_tensor(labels, dtype=tf.int64)
        return batch

#Function to tokenize the dataset. It appends a question to each of the 4 answer choices 
def preprocess_function(examples):
    first_sentences = [[context] * 4 for context in examples["question"]]
    second_sentences = [context for context in examples["choice_list"]]
    
    first_sentences = sum(first_sentences,[])
    second_sentences = sum(second_sentences,[])
    
    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation = True)
    return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

#Function to compute the validation accuracy 
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

#Function to predicting the values in a batched format on the test set
def Batch_Test(examples, batch_size=4):
    first_sentences = [[context] * batch_size for context in examples["question"]]
    second_sentences = [context for context in examples["choice_list"]]
    
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])
    print(len(first_sentences))
    # Find the longest sequence for padding
    max_length = max(len(sentence) for sentence in first_sentences + second_sentences)
    
    #predict answer for each question in a batched manner from the logits of the model output
    answers = []
    for i in range(0, len(first_sentences), batch_size):
        print(i)
        batch_first = first_sentences[i:i + batch_size]
        batch_second = second_sentences[i:i + batch_size]
        
        batch_tokenized = tokenizer(
            batch_first,
            batch_second,
            truncation=True,
            padding="max_length",  # Pad to the longest sequence
            return_tensors="np"    # Return numpy arrays
        )
        tokenized = {key: np.expand_dims(array, 0) for key, array in batch_tokenized.items()}
        
        #tokenized_examples.append(batch_tokenized)
        outputs = model(tokenized).logits
        answer = np.argmax(outputs)
        answers.append(answer)
    return answers



def transformer_model(fileInput):
    readFile = np.load(fileInput, allow_pickle = True)                              #load the file provided from SemEval Task

    dfInput = pd.DataFrame(readFile.tolist())                                                #convert Numpy to DataFrame
    dfInput['distractor1'] = dfInput['distractor1'].astype(str)                     #convert distractor1 to string type
    dfInput['distractor2'] = dfInput['distractor2'].astype(str)                     #convert distractor2 to string type
    dfInput['distractor(unsure)'] = dfInput['distractor(unsure)'].astype(str)       #convert distractor(unsure) to string type

    #Split dataset to Train, Validation and Test
    dfTrainVal, dfTest = train_test_split(dfInput, test_size=0.2, random_state= 42)   #Split input dataframe to train-val and test
    dfTrain, dfVal = train_test_split(dfTrainVal, test_size=0.2, random_state= 42)   #Split train-val to train and val

    train_dataset = Dataset.from_pandas(dfTrain)                                  #Create a dataset object for train
    val_dataset = Dataset.from_pandas(dfVal)                                      #Create a dataset object for validation

    dataset_dict = DatasetDict({'train': train_dataset, 'validation': val_dataset})  #Create a DatasetDictionary


    #Pre-process data by tokenizing with model's tokenizer
    tokenized_data = dataset_dict.map(preprocess_function, batched = True, remove_columns=['id', 'question', 'answer', 'distractor1', 'distractor2', 'distractor(unsure)', 'choice_list', 'choice_order', '__index_level_0__'])

    
    #parameters for training the model
    batch_size = 6
    num_train_epochs = 5
    total_train_steps = (len(tokenized_data["train"]) // batch_size) * num_train_epochs
    optimizer, schedule = create_optimizer(init_lr=5e-5, num_warmup_steps=0, num_train_steps=total_train_steps)


    data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
    tf_train_set = model.prepare_tf_dataset(
        tokenized_data["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    tf_validation_set = model.prepare_tf_dataset(
        tokenized_data["validation"],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    model.compile(optimizer=optimizer)  #Compile Model

    #Get metrics as the model training goes on
    metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
    callbacks = [metric_callback]

    #train the model using the train dataset
    model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=2, callbacks=callbacks)

    #Get the test dataset
    test_dataset = Dataset.from_pandas(dfTest)

    #Make predictions on the test set 
    answers = Batch_Test(test_dataset)

    #Check returned answers from the test set and print the metrics
    actual_labels = dfTest['label']
    correct_predictions = (answers == actual_labels).sum().item()
    total_examples = len(actual_labels)
    accuracy = correct_predictions / total_examples

    print("Correct Predictions:", correct_predictions)
    print("Total Examples:", total_examples)
    print("Accuracy:", accuracy)

    print(classification_report(answers, actual_labels))


#invoke the tranformer function to train the selected transformer
transformer_model(fileInput)