# Speech Understanding Assignment 2

## Project Structure
```
|-- M23CSA507_1.py    # Question 1
|-- M23CSA507_2.py    # Question 2
```

## Question 1

- Speaker Verification: Implementation of pre-trained transformer models for speaker verification
- Fine-tuning: Efficient adaptation using LoRA and ArcFace loss
- Multi-speaker Dataset: Creation of overlapped speech datasets
- Speaker Enhancement: Integration of speaker ID with separation for targeted enhancement

### Steps to run
- Change the dataset path to point the actual location.
- Run `python M23CSA507_1.py`

## Question 2

- Extracts MFCC features from speech samples.
- Performs statistical analysis of MFCC coefficients.
- Trains a neural network to classify languages based on MFCC features.
- Achieves high classification accuracy (~98%).

### Steps to run
- Change the dataset path to point the actual location.
- Run `python M23CSA507_2.py`

### Results
The trained model achieves 98% accuracy in classifying Hindi, Marathi, and Gujarati speech data.

