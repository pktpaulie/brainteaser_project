# Brain Teaser Project
The Brainteaser leverages BERT-based Transformers and a Multi-Layer Perceptron for Natural Language Inference in Answering Multiple Choice Brain Teaser questions. 

This project is part of the SemEval 2024 Shared Tasks https://brainteasersem.github.io/ 

1. The brainteaser_mlp file contains the multi-layer perceptron implementation with 4 hidden layers using TensorFlow. 

2. The brainteaser_transformers file contains the BERT, RoBERTa and AlBERT finetuned implementation using TensorFlow and 

3. The brainteaser_ensemble file contains an implementation that ensembles the weighted score from the 3 transformers
   
4. The bert_brain_teaser_proj_torch is a BERT implementation using the PyTorch library

The SP-train.npy file contains the dataset with 507 brainteaser questions, answer choices, choice order and correct answers.
