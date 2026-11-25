# Used Car Price Prediction

Predicts the selling price of a used car from its details (age, kilometers driven, specifications, brand, model, fuel type, transmission, and seller type), and serves the prediction in a Streamlit web app. Built as a Data Science course task (Mindrisers).

Dataset: CarDekho used car listings from Kaggle (Indian market, prices in rupees), 15,411 cars before cleaning.

Status: work in progress.

## Project structure

```
used-car-price-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                 original Kaggle CSV (never edited)
│   └── processed/           cleaned data and the selected feature list
├── notebooks/               numbered notebooks, one per step (see below)
├── models/                  the saved model and facts about it
├── app/                     Streamlit app
├── reports/figures/         saved charts
└── docs/                    project documentation
```

## Notebooks

Each step has its own notebook. Read and run them in order. Each one starts from the files saved by the step before it. They were written and run in Google Colab.

1. `01_data_cleaning.ipynb`: load, explore, and clean the data; create a new feature; set up preprocessing.
2. `02_eda.ipynb`: exploratory data analysis with charts (training data only).
3. `03_feature_selection.ipynb`: choose which features the model uses.
4. `04_model_development.ipynb`: compare Linear Regression, Random Forest, and Gradient Boosting on raw and log price.
5. `05_tuning_and_final_model.ipynb`: tune the best models, choose the final one, test it, and save it.

## Run the app

You need Python 3.14 (the versions in `requirements.txt` were checked on 64-bit Windows with Python 3.14). Open a terminal in the project folder and run:

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The app opens in your browser at http://localhost:8501. If Streamlit asks for an email address the first time, press Enter to skip it. To stop the app, press Ctrl+C in the terminal.

Notes:

- On Mac or Linux, activate the environment with `source .venv/bin/activate` instead.
- If PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope Process Bypass` first, then try again.
- The model was saved with scikit-learn 1.9.1, and the app checks that this exact version is installed. `requirements.txt` takes care of it.

## About the model

The final model is a tuned Gradient Boosting model trained on the raw price. In cross-validation on the training data, half of its predictions were within about 10% of the real price (RMSE about 2.1 lakh rupees, R2 0.93).

It is reliable for ordinary cars and unreliable for very expensive luxury cars. Tree-based models cannot predict prices above the highest price they were trained on (145 lakh), so cars like a Ferrari or a Rolls-Royce are badly underpredicted. The app shows a warning for very expensive predictions, for car models with few examples in the data, and for unusual inputs.
