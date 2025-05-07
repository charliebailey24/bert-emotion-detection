import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from datasets import load_dataset
from transformers import (BertTokenizer, 
                          BertForSequenceClassification, 
                          DataCollatorWithPadding, 
                          TrainingArguments,
                          Trainer,
                          )

# The development of this code was heavily influenced by the
# Hugging Face LLM course Chapter 3 on model fine-tuning.
# code citation: https://huggingface.co/learn/llm-course/chapter3/

# style note: '' for arguments/parameters, "" for strings

class EmotionDetection():
    def __init__(self):
        """
        Load data. Initialize tokenizer and model.
        """
        self.dataset = load_dataset('dair-ai/emotion', 'split')
        self.checkpoint = "bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(self.checkpoint)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.tokenized_datasets = self.dataset.map(
            self.tokenize_function,
            batched=True
        )
        self.model = BertForSequenceClassification.from_pretrained(
            self.checkpoint,
            num_labels = 6,
        )
        self.trainer = None
        self.train = self.tokenized_datasets['train']
        self.validation = self.tokenized_datasets['validation']
        self.test = self.tokenized_datasets['test']

    def tokenize_function(self, data):
        """
        Tokenize the input data.
        """
        return self.tokenizer(
            data['text'],
            truncation=True, # cutoff text longer than max_length
        )

    def compute_metrics(self, eval_preds):
        """
        Compute the accuracy and weighted F1 metrics for the model.
        """
        # unpack the input predictions the logits and labels
        logits, labels = eval_preds
        # find the highest value logit among the 6 class options
        # this is the models prediction (e.g., 'joy')
        predictions = np.argmax(logits, axis=-1)

        # compute the accuracy and weighted F1 score for the predictions
        accuracy = accuracy_score(labels, predictions)
        # weighted F1 score is used to deal with the class imbalances
        f1_weighted = f1_score(labels, predictions, average='weighted')

        return { 'accuracy': accuracy, 'f1': f1_weighted }
    
    def get_training_args(self):
        """
        Get the training arguments for the model.
        """
        # set training arguments
        # `mps` device will be used by default if available
        training_args = TrainingArguments(
            output_dir='../trained_models', # output directory
            eval_strategy='epoch', # evaluate every epoch
        )
        return training_args
    
    def get_trainer(self, training_args):
        """
        Get the trainer for the model.
        """
        # create the trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train,
            eval_dataset=self.validation,
            processing_class=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )
        return trainer

    def train_model(self):
        """
        Train the model.
        """
        # get the training args
        training_args = self.get_training_args()
        # get the trainer
        trainer = self.get_trainer(training_args)
        # train the model
        trainer.train()

        # store trainer
        self.trainer = trainer

        # evaluate the model on validation data
        validation_results = trainer.evaluate()
        print(f"validation results::: {validation_results}")

    def display_confusion_matrix(self, labels, preds, class_names):
        """
        Display the confusion matrix for the model.
        """
        cm = ConfusionMatrixDisplay.from_predictions(labels,
                                                     preds,
                                                     display_labels=class_names)
        cm.ax_.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

    def evaluate(self):
        """
        Evaluate the model on the test data.
        """
        test_results = self.trainer.evaluate(self.test)

        print(f"test results::: {test_results}")

        # get the test predictions and labels
        predictions = self.trainer.predict(self.test)
        preds = np.argmax(predictions.predictions, axis=-1)
        labels = predictions.label_ids

        # get the class names
        class_names = self.dataset['train'].features['label'].names

        # load the dataset metrics
        eval_preds = (predictions.predictions, predictions.label_ids)
        test_score = self.compute_metrics(eval_preds)
        
        print(f"test score:::\n{test_score}\n")

        # display the confusion matrix
        self.display_confusion_matrix(labels, preds, class_names)


def main():
    # confirm torch access to local GPUs
    print(f"backend mps is available::: {torch.backends.mps.is_available()}\n")

    # parse the command-line arguments
    # code citation: ChatGPT o3
    parser  = argparse.ArgumentParser()
    parser.add_argument(
        '--eval_only',
        action='store_true', # default behavior is NOT to eval_only
        help='Skip training and only evaluate the saved model'
    )
    args = parser.parse_args()

    # create the emotion detector
    emotion_detector = EmotionDetection()

    # check the args to determine load or train new model
    # code citation: ChatGPT o3
    if args.eval_only and os.path.isdir('../trained_models'):
        # load the checkpoints from the directory
        checkpoints  = [
            os.path.join('../trained_models', d)
            for d in os.listdir('../trained_models')
            if d.startswith('checkpoint-')
        ]

        # pick the highest checkpoint
        if checkpoints:
            last_checkpoint = max(
                checkpoints,
                key=lambda x: int(x.split('-')[-1]) # split on the dash and grab the number
            )
            load_path = last_checkpoint
        else:
            load_path = '../trained_models'

        print(f"Loading model from {load_path}")

        # set the model to the last checkpoint if it exists
        emotion_detector.model = BertForSequenceClassification.from_pretrained(
            load_path
        )
        # get the training args
        training_args = emotion_detector.get_training_args()
        # get the trainer
        trainer = emotion_detector.get_trainer(training_args)
        # set the trainer to the loaded model
        emotion_detector.trainer = trainer
    else:
        # train the new model
        emotion_detector.train_model()
    
    # evaluate the either the new or loaded model
    emotion_detector.evaluate()

if __name__ == '__main__':
    main()
