## Human Disease Detection

This repository contains the code for a web application that predicts human diseases based on symptoms. The application is built with Streamlit, and the core functionality is powered by several machine learning models.

Author: Vinisha Singh & Chitresh Mathur

## Features
This project have the following features:
* Interactive UI: A user-friendly interface built with Streamlit allows users to input symptoms.
* Machine Learning Models: The application utilizes multiple pre-trained machine learning models including SVM, Random Forest, Decision Tree, Naive Bayes, and KNN to predict potential diseases.
* Real-time Prediction: Provides instant predictions based on the user's input.
* Visual Graphs: The app now includes prediction comparison graphs to show model agreement at a glance.

## Repository Contents

The project is structured to organize data and models for easy access:
- streamlit_app.py: The main Python script that contains the Streamlit application logic and serves the web interface.
- models_training.ipynb: A Jupyter Notebook file that shows the process of training and evaluating the machine learning models.
- datasets/: This directory stores the data, including a CSV file of medical symptoms and diseases used for training the models.
- artifacts/: This directory contains the trained machine learning models, which are loaded by the app.py script to make predictions.

## Deployed Application

Link - https://humandiseasedetectionbyvinisha.streamlit.app/

## Contributing

Feel free to open issues or submit pull requests if you have suggestions for improvements.
