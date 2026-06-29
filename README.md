# Apple Next-Day Direction Prediction with Logistic Regression

This project is a simple experiment in using Logistic Regression to predict whether Apple’s stock price will go up or down the next trading day. The point is not to present Logistic Regression as the best model for this problem, but to test a basic, interpretable classifier on financial time-series data and see what it can and cannot do.

## Project idea

The notebook uses historical Apple (`AAPL`) stock data downloaded from Yahoo Finance through `yfinance`, covering 2010-01-04 to 2026-06-26 in the current version of the project. The target is binary:

- `1` if the next day’s closing price is higher than today’s closing price
- `0` otherwise

This makes the task a classification problem rather than a price-forecasting problem.

## Why Logistic Regression?

Logistic Regression is not usually the first model people think of for stock prediction, especially for short-horizon direction forecasting. Still, it is a useful starting point because it is simple, fast, interpretable, and works well as a baseline before trying more complex models.

This project treats Logistic Regression as a learning tool. The goal is to understand the workflow of a time-series classification problem: data collection, feature engineering, chronological splitting, scaling, model fitting, evaluation, and a small amount of hyperparameter tuning.

## Dataset

The dataset contains daily Apple stock data with the following columns:

- `Adj Close`
- `Close`
- `High`
- `Low`
- `Open`
- `Volume`

In the current notebook, 4,145 daily records are downloaded, and after feature engineering and dropping rows with missing derived values, 4,096 rows remain for modelling.

## Features

The model uses a set of technical-indicator-style features built from historical price and volume data:

- 1-day, 2-day, 5-day, and 10-day returns
- Distance from 5-day, 20-day, and 50-day moving averages
- 1-day volume change
- 5-day, 10-day, and 20-day rolling volatility
- 14-day RSI

These features are intentionally simple. They are meant to test whether basic information from past prices and trading volume is enough to help a linear classifier predict next-day direction.

## Method

The modelling dataset is sorted in chronological order and split into training and testing sets using an 80/20 time-based split. In the current notebook:

- Training period: 2010-03-16 to 2023-03-20 
- Testing period: 2023-03-21 to 2026-06-26 
- Training shape: `(3276, 12)` 
- Testing shape: `(820, 12)` 

The baseline model is a Logistic Regression classifier trained on standardized features. A later section also explores hyperparameter tuning to see whether changing regularization improves performance.

## Baseline results

The baseline model produces the following test-set results:

| Metric | Value |
|--------|-------|
| Accuracy | 0.521951 |
| Balanced accuracy | 0.493900 |
| Precision | 0.533149 |
| Recall | 0.877273 |
| F1 score | 0.663230 |
| ROC-AUC | 0.458206 |

The confusion matrix from the baseline model is:

```text
[[ 42 338]
 [ 54 386]]
```

These results show that the model predicts upward days often and achieves high recall for class `1`, but performs poorly at separating upward and non-upward days overall. In particular, the ROC-AUC is below 0.5, which suggests that the model is weak as a ranking classifier.

## Tuned results

After hyperparameter tuning, the model changes only slightly:

| Metric | Baseline | Tuned |
|--------|----------|-------|
| Accuracy | 0.521951 | 0.523171 |
| Balanced accuracy | 0.493900 | 0.493959 |
| Precision | 0.533149 | 0.533243 |
| Recall | 0.877273 | 0.893182 |
| F1 score | 0.663230 | 0.667799 |
| ROC-AUC | 0.458206 | 0.460443 |

Tuning improves the model only marginally. The overall story stays the same: Logistic Regression can be used as a baseline for this task, but it does not produce a strong predictive signal here.

## Main takeaway

This repository is best understood as an experiment with a simple baseline model, not as a claim that Logistic Regression is well-suited to next-day stock prediction. The project is useful because it shows a complete machine learning workflow on financial data while also showing how limited the results can be when the underlying problem is difficult.

In that sense, the weak performance is still informative. It suggests that basic technical indicators and a linear model are not enough to produce a reliable next-day trading signal for Apple in this setup.

## Repository structure

```text
apple-next-day-direction/
├── README.md
├── requirements.txt
├── data/
│   └── AAPL_raw.csv
├── notebooks/
│   └── apple_next_day_direction.ipynb
└── figures/
```

## How to run

1. Clone the repository.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Open the notebook in Jupyter or Google Colab.
4. Run the notebook cells in order.

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- yfinance
- jupyter

## Limitations

- The model uses only historical price and volume information.
- It does not include news, fundamentals, macroeconomic data, or broader market indicators.
- Logistic Regression is a linear model, so it may miss more complex patterns in financial time series.
- Good notebook workflow does not automatically imply strong predictive performance.

## Final note

This project is mainly an attempt to try out Logistic Regression on a difficult financial classification task and document the result honestly. The model is useful as a baseline and as a learning exercise, but not as evidence of a robust trading strategy.
