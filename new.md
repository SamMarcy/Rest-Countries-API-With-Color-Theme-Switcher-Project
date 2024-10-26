# CAVE-NET: CVIP-VISION-CHALLENGE Submission

## Overview

Welcome to our submission for the CVIP-VISION-CHALLENGE! Our solution, **CAVE-NET**, tackles the challenge of multi-class abnormality classification in medical imaging—a task complicated by the heavy class imbalances often found in medical datasets. To address this, we developed a unique pipeline combining data augmentation with a powerful ensemble of models to improve accuracy and generalization.

## Project Structure

Here’s a quick rundown of the project files and the structure:

1. **Data Augmentation** (`augmentation.ipynb`)  
   Medical imaging datasets can suffer from class imbalances, which can affect model performance. Our first step was to address this issue using data augmentation. Running this notebook will balance the dataset and create a strong foundation for model training. **Make sure to run this file first** to get everything set up.

2. **CAVE-NET Model** (`encoderdecoder.ipynb`)  
   This notebook brings together the core components of our final model, CAVE-NET. Here’s what’s inside:
   - **Encoder-Decoder Architecture**: We first run an encoder-decoder model on the augmented dataset, generating both **latent spaces** and **reconstructed images**.
   - **CBAM Model**: The Convolutional Block Attention Module (CBAM) model is trained on both the original and reconstructed images. CBAM focuses on refining feature extraction, allowing it to capture relevant details more effectively.
   - **Deep Neural Network (DNN)**: The DNN model takes in latent spaces from the encoder. This approach uses the high-level representations extracted from images, which enhances performance.
   - **Ensemble Model**: To bring in multiple perspectives, we created an ensemble of smaller models (SVM, KNN, XGB, and Random Forest), all trained on the latent space. This component adds diversity and stability.
   - **Final Ensemble**: The final output comes from combining predictions from the CBAM, DNN, and ensemble models using a soft-voting technique. This meta-ensemble boosts the overall accuracy of CAVE-NET.

## Instructions

1. **Run Data Augmentation**:  
   Open the augmentation notebook and run it to balance the dataset:
   ```bash
   jupyter notebook augmentation.ipynb
   ```

2. **Run CAVE-NET Model**:  
   Once you have the augmented dataset, open and run `encoderdecoder.ipynb` to perform latent space extraction, model training, and prediction.
   ```bash
   jupyter notebook encoderdecoder.ipynb
   ```

## Results

CAVE-NET performed exceptionally well, achieving **91.2% accuracy** on both the training and validation datasets. This result reflects CAVE-NET’s ability to handle class imbalance effectively and accurately classify images by integrating image features with high-level representations through its ensemble structure.

## Acknowledgements

We’d like to thank the organizers of the CVIP-VISION-CHALLENGE for this exciting opportunity. This challenge allowed us to push the boundaries of what’s possible in medical imaging, and we hope CAVE-NET can contribute meaningfully to this field.
