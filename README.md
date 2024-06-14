# DermalAI: An Intelligent System for Automated Skin Condition Classification

## Introduction

**DermalAI** is an innovative iOS application designed to assist in the classification of skin conditions using advanced machine learning techniques. Leveraging Core ML and Vision frameworks, this app aims to provide quick and accurate identification of various dermatological conditions from user-uploaded images or photos taken directly within the app.

## Objectives

- To develop a user-friendly mobile application for skin condition classification.
- To utilize a machine learning model trained on a diverse dataset of skin conditions.
- To provide users with confidence levels for each classification, enhancing the reliability of the results.
- To create a portable and accessible tool for preliminary skin condition assessment.

## Methodology

1. **Data Collection**: Gathered a comprehensive dataset of skin condition images, labeled with corresponding conditions.
2. **Model Training**: Trained a deep learning model using the collected dataset to recognize and classify various skin conditions.
3. **Model Conversion**: Converted the trained model to Core ML format for integration into the iOS application.
4. **App Development**: Developed the iOS application using Swift, integrating Core ML and Vision frameworks to enable image classification.
5. **User Interface**: Designed a user-friendly interface allowing users to upload images or take photos, view classification results, and see confidence levels.

## Features

- Upload multiple images from the photo library.
- Take photos using the camera within the app.
- Classify skin conditions using a trained machine learning model.
- Display confidence levels for each prediction.
- Provide a summary if multiple images are classified as the same condition.

## Requirements

- iOS 13.0+
- Xcode 12.0+
- Swift 5.0+
- Core ML
- Vision
- AVFoundation

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/DermalAI.git
