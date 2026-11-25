# Insurance Customer Segmentation

This project uses **K-Means clustering** to group auto-insurance customers based on their premium, claims, customer value, and policy details.

The goal is to understand different customer groups and suggest actions for customer retention and cross-selling.

## Project Features

- Data cleaning and preparation
- Feature engineering and data scaling
- K-Means customer segmentation
- Cluster evaluation and validation
- Customer segment profiles
- Visual reports and Streamlit dashboard
- Prediction for new customers
- Automated tests

## Tools Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit
- Jupyter Notebook
- Pytest

## Project Structure

```text
insurance-customer-segmentation/
├── dashboard/       # Streamlit dashboard
├── data/            # Datasets
├── models/          # Saved models
├── notebooks/       # Jupyter notebook
├── reports/         # Charts and results
├── src/             # Project source code
├── tests/           # Automated tests
├── requirements.txt
├── run_pipeline.py
└── README.md
```

## How to Run

### 1. Create a virtual environment

**Windows:**

```bash
py -m venv .venv
.venv\\Scripts\\activate
```

### 2. Install the required packages

```bash
python -m pip install -r requirements.txt
```

### 3. Run the project

```bash
python run_pipeline.py
```

### 4. Run the dashboard

```bash
streamlit run dashboard/app.py
```

### 5. Run tests

```bash
pytest -q
```

## Dataset

A synthetic dataset is included so the project can run immediately.

If you want to use the public Auto Insurance Customer Lifetime Value dataset, obtain it from its original source and check its licence. Place the authorised CSV file in `data/raw/` with this name:

`auto_insurance_clv.csv`

The pipeline uses this file when it is available. Otherwise, it uses the synthetic dataset.



## Output

The project generates:

- Customer segments
- Segment profiles
- Model evaluation results
- Charts and reports
- Saved machine learning models
- Data for the Streamlit dashboard


