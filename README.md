# Bank Marketing Classification

End-to-end binary-classification project developed for the DS52 mini-project. The objective is to predict whether a bank client will subscribe to a term deposit.

Prepared by **Iyad EL HAJJ CHEHADE**

## Dataset

The project uses the Kaggle Playground Series S5E8 dataset, [Binary Classification with a Bank Dataset](https://www.kaggle.com/competitions/playground-series-s5e8).

The competition data files are not stored in this repository. Download the following files from Kaggle and place them beside the notebook:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

## Project workflow

1. Data inspection and exploratory data analysis
2. Leakage prevention by excluding `duration`
3. Feature engineering for previous campaign contact history
4. Numerical imputation and categorical one-hot encoding
5. Stratified training and validation split
6. Baseline comparison of Logistic Regression, Random Forest and XGBoost
7. Hyperparameter tuning with cross-validation using PR-AUC
8. Out-of-fold classification-threshold selection
9. Feature importance, permutation importance and error analysis
10. Final model refitting and test prediction export

## Model comparison

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Tuned Logistic Regression | 0.8918 | 0.6835 | 0.1926 | 0.3005 | 0.7780 | 0.4400 |
| Tuned Random Forest | 0.8583 | 0.4363 | 0.5972 | 0.5042 | 0.8417 | 0.5242 |
| Tuned XGBoost, threshold 0.50 | 0.9007 | 0.6795 | 0.3353 | 0.4490 | 0.8527 | 0.5390 |
| Tuned XGBoost, threshold 0.2453 | 0.8828 | 0.5134 | 0.5415 | **0.5271** | **0.8527** | **0.5390** |

The final configuration is the tuned XGBoost pipeline with a classification threshold of `0.2453`. This threshold provides the strongest F1-score and a more balanced precision-recall trade-off for the project objective.

## Main findings

- The target is imbalanced, with approximately 12.07% positive observations.
- `duration` was removed because it is only known after the call and would introduce target leakage.
- Previous campaign outcome, contact month, contact method, balance and campaign timing were influential predictors.
- A previous successful campaign strongly increases the probability of a positive prediction.
- False positives often resemble true subscribers, while false negatives frequently resemble true non-subscribers.
- Feature importance describes predictive association and should not be interpreted as causality.

## Repository structure

```text
bank-marketing-classification/
├── bank-marketing-classification.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

The optional trained pipeline can be stored under `models/final_xgboost_pipeline.joblib`.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the project

1. Download the three competition CSV files from Kaggle.
2. Place them in the project directory beside the notebook.
3. Open `bank-marketing-classification.ipynb` in Jupyter or VS Code.
4. Restart the kernel and run all cells in order.

The notebook generates:

- `submission_xgboost_probabilities.csv`: probability predictions for competition submission
- `test_predictions_with_threshold.csv`: probabilities and threshold-based classes
- `final_xgboost_pipeline.joblib`: fitted preprocessing and XGBoost pipeline

## Reproducibility notes

- Random seeds are fixed at `42` where applicable.
- Cross-validation is stratified.
- Preprocessing is included inside scikit-learn pipelines.
- Hyperparameter tuning is CPU-based and can take several minutes depending on the machine.
- The saved Joblib model should be loaded with compatible scikit-learn and XGBoost versions.

## Limitations

- Validation results are specific to the available synthetic competition dataset.
- The optimized threshold depends on the selected F1 objective and may need adjustment for different operational costs.
- Observed relationships are predictive associations, not causal effects.
