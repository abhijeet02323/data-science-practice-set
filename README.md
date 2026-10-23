# Data Science Practice with Python

A hands-on collection of **Python Data Science practice sets, exercises,
and projects using real datasets**. This repository is designed to build
practical skills step by step---from data manipulation and cleaning to
exploratory analysis, machine learning, clustering, and a brief
introduction to deep learning.

The exercises emphasize writing code, interpreting results, visualizing
data, and documenting findings rather than only learning theory.

## Repository Goals

-   Practice core Python data science libraries on real datasets.
-   Learn to inspect, clean, transform, and validate data.
-   Use visualizations and descriptive statistics to understand
    datasets.
-   Implement supervised and unsupervised machine learning workflows.
-   Understand a basic deep learning workflow with a neural network.
-   Build reproducible scripts that generate evaluation metrics, plots,
    and output files.

## Technology Stack

-   **Language:** Python
-   **Data handling:** pandas, NumPy
-   **Visualization:** Matplotlib, Seaborn
-   **Machine learning:** scikit-learn
-   **Deep learning:** TensorFlow / Keras
-   **Datasets:** Titanic passenger data, Wine dataset, and MNIST
    handwritten digits


## Practice Sets

### Practice Set 1 --- Data Cleaning and Preprocessing

**Objective:** Prepare raw data for analysis and machine learning.

Practice questions: 1. Load a dataset using pandas and display its first
and last five rows. 2. Find the number of rows, columns, and data types
in the dataset. 3. Identify missing values in each column and calculate
their percentages. 4. Detect duplicate rows and decide whether they
should be removed. 5. Examine inconsistent values, invalid entries, and
unusual data types. 6. Impute missing numerical values using an
appropriate strategy. 7. Handle missing categorical values and explain
the selected method. 8. Identify potential outliers using summary
statistics or visualizations. 9. Convert columns to suitable data types
and encode categorical values where needed. 10. Export the cleaned
dataset and summarize the changes made.

**Skills:** pandas, NumPy, missing-value handling, duplicates, data
types, preprocessing, data quality.

### Practice Set 2 --- Exploratory Data Analysis (EDA)

**Dataset:** Titanic passenger dataset.

**Objective:** Explore the dataset and discover patterns using
statistics and visualizations.

Practice questions: 1. Load the dataset and inspect its shape, columns,
and data types. 2. Calculate descriptive statistics for numerical and
categorical columns. 3. Find the survival rate and visualize the target
distribution. 4. Compare survival rates by passenger sex and passenger
class. 5. Analyze the age distribution and investigate missing age
values. 6. Compare ticket fares across passenger classes. 7. Examine
family size and whether passengers travelled alone. 8. Create suitable
bar charts, histograms, box plots, and count plots. 9. Build a
correlation heatmap for appropriate numerical variables. 10. Write three
to five evidence-based observations from the analysis.

**Skills:** descriptive statistics, grouping and aggregation, Seaborn,
Matplotlib, correlation, data storytelling.

### Practice Set 3 --- Supervised Learning

**Dataset:** Titanic passenger dataset.

**Objective:** Train a classification model to predict whether a
passenger survived.

Practice questions: 1. Define the target variable and select relevant
input features. 2. Separate the data into training and testing sets
using a stratified split. 3. Build preprocessing steps for numerical and
categorical columns. 4. Handle missing values and encode categorical
features. 5. Train a Logistic Regression classification model. 6.
Evaluate the model using accuracy, precision, recall, and F1-score. 7.
Create and interpret a confusion matrix. 8. Calculate ROC-AUC and plot
the ROC curve where supported. 9. Use cross-validation to estimate
performance more reliably. 10. Explain limitations such as class
imbalance, missing information, and generalization.

**Skills:** scikit-learn pipelines, train/test split, classification,
model evaluation, cross-validation.

### Practice Set 4 --- Unsupervised Learning and Clustering

**Dataset:** Wine dataset from scikit-learn.

**Objective:** Discover groups in data without using class labels to
train the clustering model.

Practice questions: 1. Load the Wine dataset and inspect its features.
2. Standardize numerical features before distance-based clustering. 3.
Explain the difference between supervised and unsupervised learning. 4.
Apply K-Means clustering to the standardized features. 5. Use the elbow
method to explore a suitable number of clusters. 6. Calculate a
silhouette score to assess cluster separation. 7. Apply Agglomerative
Clustering and compare the results with K-Means. 8. Use Principal
Component Analysis (PCA) to project the data into two dimensions. 9.
Visualize the resulting clusters and discuss their separation. 10.
Describe why clusters may not correspond perfectly to known classes.

**Skills:** feature scaling, K-Means, hierarchical clustering,
silhouette score, elbow method, PCA.

### Practice Set 5 --- Deep Learning Basics

**Dataset:** MNIST handwritten digits.

**Objective:** Understand how a neural network can learn to classify
images.

Practice questions: 1. Load the MNIST dataset using TensorFlow/Keras. 2.
Inspect image dimensions, labels, and the number of classes. 3.
Normalize pixel values to a suitable numeric range. 4. Create training,
validation, and test data partitions. 5. Build a small Convolutional
Neural Network (CNN). 6. Explain the roles of convolution, pooling,
dense layers, and dropout. 7. Compile the model with an optimizer, loss
function, and evaluation metric. 8. Train the model and track training
and validation loss and accuracy. 9. Evaluate the trained model on the
test set. 10. Generate a confusion matrix and inspect examples of
incorrect predictions.

**Skills:** neural networks, CNNs, TensorFlow/Keras, epochs, batches,
loss, validation, classification metrics.

## Integrated Project --- Titanic Passenger Survival and Segmentation

This project combines data science techniques into one practical
workflow using the Titanic passenger dataset. It is an **integrated data
science project**, not a capstone project.

### Project Objectives

-   Inspect and assess the quality of passenger data.
-   Clean missing and inconsistent values for analysis and modelling.
-   Perform exploratory data analysis to identify patterns in survival.
-   Engineer useful features such as family size and whether a passenger
    travelled alone.
-   Build a supervised classification pipeline to predict passenger
    survival.
-   Apply K-Means clustering to explore passenger segments using
    selected passenger characteristics.
-   Evaluate model performance and interpret the discovered clusters.
-   Generate charts, evaluation summaries, cluster profiles, and
    passenger-level output files.

### Main Workflow

1.  **Data acquisition:** load the public Titanic dataset.
2.  **Data quality:** inspect missing values, data types, and
    duplicates.
3.  **Preprocessing:** impute missing values, encode categorical
    variables, and scale suitable features.
4.  **Feature engineering:** derive family-related features.
5.  **EDA:** visualize survival patterns by sex, class, age, and other
    available attributes.
6.  **Supervised learning:** train and evaluate a Logistic Regression
    classifier with a held-out test set and cross-validation.
7.  **Unsupervised learning:** use K-Means to group passengers based on
    selected features, without using the survival target to fit the
    clusters.
8.  **Interpretation:** compare cluster profiles and explain the
    findings and limitations.

### Example Outputs

The Python script is designed to create an output directory containing
files such as:

-   `data_quality_report.csv` --- summary of data quality checks.
-   `evaluation_report.txt` --- classification and clustering evaluation
    results.
-   `cluster_profiles.csv` --- summary statistics for each cluster.
-   `passengers_with_clusters.csv` --- passenger records with assigned
    cluster labels.
-   Charts for EDA, classification evaluation, and cluster
    visualization.

Run the script to generate the actual outputs. Report metrics should be
taken from the script's output rather than assumed in advance.

**Important:** The Titanic dataset is historical and incomplete. A
model's predictions do not establish why an individual passenger
survived, and the discovered clusters are exploratory groupings---not
definitive passenger categories.

## Dataset Sources

-   **Titanic:** [Seaborn Titanic dataset on
    GitHub](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv)
-   **Wine:** [scikit-learn Wine dataset
    documentation](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-dataset)
-   **MNIST:** [TensorFlow/Keras MNIST dataset
    documentation](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist)

Dataset availability, schemas, and download methods can change. Check
the source documentation if a download fails.

## Installation

Use Python 3.10 or a compatible version supported by the libraries
installed in your environment.

Create and activate a virtual environment:

``` bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the core data science dependencies:

``` bash
python -m pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn
```

For the MNIST deep learning exercise, install TensorFlow as well:

``` bash
pip install tensorflow
```

## Running the Exercises

Run the relevant Python script from the repository root. For example:

``` bash
python survival_pred_titanic.py
```

Replace the filename with the script you want to run. Some exercises
download datasets from public sources, so an internet connection may be
required on the first run. Review each script's comments for input paths
and output locations.

## Learning Outcomes

After completing these practice sets, you should be able to:

-   Load, inspect, clean, and transform tabular data.
-   Perform exploratory analysis and communicate findings with charts.
-   Build reproducible preprocessing and machine learning pipelines.
-   Evaluate classification models using multiple metrics.
-   Apply clustering and dimensionality reduction to explore data
    structure.
-   Explain the basic workflow of training and evaluating a CNN.
-   Organize a small end-to-end data science project with documented
    outputs.

## Contributing

Suggestions and improvements are welcome. To contribute:

1.  Create a branch for your change.
2.  Add a practice question, solution, dataset note, or visualization.
3.  Keep code readable and add comments where they clarify reasoning.
4.  Include instructions for running the script and identifying its
    outputs.
5.  Avoid committing large datasets, credentials, or generated files
    unless necessary.

