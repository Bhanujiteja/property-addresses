# Property Address Classifier

## Overview
This implements a Machine Learning classifier to categorize property addresses into 5 predefined classes (`flat`, `houseorplot`, `landparcel`, `commercial unit`, `others`). It uses a **Linear SVC** approach with **TF-IDF** vectorization, designed for high efficiency and interpretability.

## Directory Structure
* `p_c.ipynb`: The main notebook containing EDA, training logic, and evaluation.
* `predict.py`: A standalone script for running inference on new data.
* `best_model/`: Contains the serialized model pipeline (`.pkl`).
* `requirements.txt`: List of dependencies.
* `dataset.csv`: The training data used.

## Setup & Installation

1. **Clone the repository** (or unzip the folder).
2. **Create a virtual environment** (Recommended):
   
## python version 
python --version==3.13

also Implemented Stremlit web interface you can just run :
streamlit run app.py
or
you can directly test in P_C.ipynb notebook in the last cell with giving data
