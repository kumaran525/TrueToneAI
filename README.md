# TrueToneAI
Version: v1.0 | A research-driven POC to detect AI-generated deepfake voices using a Siamese network built with TensorFlow/Keras.

This project detects AI-generated deepfake voices using a Siamese network.

TrueToneAI is a deep learning pipeline designed to detect synthetic (AI-generated) voices from authentic human audio. This project uses a Siamese network that processes both:

MFCC sequences (temporal audio features)

Scalar features (formants, ZCR, pitch, etc.)

It learns similarity between real vs. fake audio pairs to make predictions.

# Model Architecture
Input A: MFCC + Scalar features from a real audio file

Input B: MFCC + Scalar features from a suspected fake audio file

Shared CNN for MFCCs & shared DNN for scalar inputs

Contrastive Loss and Euclidean Distance for similarity learning

# Dataset
We used audio data from:

/REAL/ — genuine human speech

/FAKE/ — synthetic voice clones using AI models (e.g., biden-to-trump, biden-to-obama)

# Features Extracted:
MFCCs (Mel-Frequency Cepstral Coefficients)

Spectral Centroid

Zero-Crossing Rate

Short-Time Energy

Formants

Pitch Contour

Speaker Metadata (1-hot)
