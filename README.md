# Metaheuristic-Based Smart Parking Allocation Using GRU and TLBO

## 📌 Project Overview

This project presents a smart parking allocation framework that combines deep learning and metaheuristic optimization to improve parking space allocation.

The system uses **LASSO feature selection** to identify relevant features, a **Gated Recurrent Unit (GRU)** model to predict parking occupancy, and the **Teaching–Learning-Based Optimization (TLBO)** algorithm to allocate parking spaces efficiently.

The proposed TLBO-based approach is benchmarked against the **Enhanced Secretary Bird Optimization Algorithm (ESBOA)**.

---

## 🎯 Objectives

The main objectives of this project are:

- Predict future parking occupancy using GRU.
- Select important input features using LASSO.
- Allocate suitable parking spaces using TLBO.
- Reduce vehicle travel distance and waiting time.
- Improve parking space utilization.
- Consider traffic conditions and parking availability.
- Reduce carbon emissions caused by unnecessary vehicle movement.
- Compare TLBO performance with ESBOA.

---

## 🧠 Methodology

The proposed framework consists of the following stages:

### 1. Data Generation and Preprocessing

A synthetic parking occupancy dataset is generated to represent historical parking conditions.

The data is cleaned and prepared for model training.

### 2. LASSO Feature Selection

LASSO regression is used to identify the most relevant features from the parking dataset.

This helps reduce unnecessary features and improves the efficiency of the prediction model.

### 3. GRU-Based Occupancy Prediction

A Gated Recurrent Unit (GRU) neural network is trained using historical parking data.

The GRU model predicts future parking occupancy and provides information that can be used during parking allocation.

### 4. TLBO-Based Parking Allocation

The predicted parking occupancy is provided to the Teaching–Learning-Based Optimization (TLBO) algorithm.

TLBO searches for an efficient parking allocation considering multiple objectives such as:

- Travel distance
- Waiting time
- Parking availability
- Traffic conditions
- Carbon emissions
- Driver satisfaction

### 5. ESBOA Benchmark

The Enhanced Secretary Bird Optimization Algorithm (ESBOA) is implemented as a benchmark optimization method.

The performance of TLBO and ESBOA is compared using common optimization metrics.

---

## 🔄 System Workflow

```text
Historical Parking Data
          ↓
Data Preprocessing
          ↓
LASSO Feature Selection
          ↓
GRU Occupancy Prediction
          ↓
Predicted Parking Occupancy
          ↓
TLBO Parking Allocation
          ↓
Optimal Parking Assignment
          ↓
Performance Evaluation
          ↓
Comparison with ESBOA