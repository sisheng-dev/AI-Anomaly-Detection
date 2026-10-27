# Blockchain Anomaly Detection with Machine Learning

An unsupervised machine learning system for detecting anomalous Ethereum smart contract transactions using behavioral analytics and Isolation Forest models.

## Overview

Blockchain ecosystems process millions of transactions daily, making manual identification of suspicious or abnormal activity nearly impossible at scale. Traditional rule-based approaches often fail to detect emerging attack patterns, fraudulent activity, or unusual transaction behaviors.

This project leverages unsupervised machine learning to automatically identify anomalous transaction patterns within Ethereum smart contract data, enabling data-driven blockchain risk analysis and fraud detection.

## Problem Statement

Ethereum transaction networks generate vast amounts of behavioral data, including transaction frequency, value transfers, wallet interactions, and temporal activity patterns.

## Challenge
Suspicious activity often represents only a small fraction of total transactions.
Labeled fraud datasets are limited or unavailable.
Traditional rule-based detection methods struggle to identify previously unseen behaviors.
Large blockchain datasets require scalable analytical approaches.

The objective was to develop a machine learning solution capable of identifying unusual transaction behavior without requiring labeled training data.

## Solution

Designed and implemented an anomaly detection pipeline using an Isolation Forest model to identify potentially suspicious Ethereum smart contract transactions.

The system collected blockchain transaction data through the Etherscan API, engineered behavioral and temporal features, and applied unsupervised learning techniques to detect outlier activity patterns.

## Technologies Used
Python
Pandas
NumPy
Scikit-learn
TensorFlow
Etherscan API
Matplotlib
Jupyter Notebook
Methodology
Data Collection

## Machine Learning Approach

An Isolation Forest anomaly detection model was trained to identify outliers within the feature space.

The model works by:

Randomly partitioning observations.
Isolating rare data points more quickly than normal observations.
Assigning anomaly scores based on isolation depth.
Flagging highly isolated transactions as potential anomalies.

Because the model is unsupervised, it can discover previously unseen suspicious behaviors without requiring labeled examples.
