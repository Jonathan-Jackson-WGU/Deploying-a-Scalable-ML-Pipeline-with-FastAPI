# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a Random Forest Classifier trained using scikit-learn's `RandomForestClassifier` with 100 estimators and a fixed random state of 42. It was developed as part of a scalable ML pipeline deployed via FastAPI. The model predicts whether an individual's annual income exceeds $50,000 based on demographic and employment features from the U.S. Census dataset.

## Intended Use

The model is intended to demonstrate a production-style ML pipeline, including data processing, model training, inference, and slice-based performance evaluation. Its primary use case is binary income classification for educational and research purposes. It should not be used to make real-world decisions about individuals' financial eligibility, hiring, lending, or any high-stakes outcome.

## Training Data

The model was trained on the UCI Adult Census Income dataset (`census.csv`), which contains approximately 32,561 records drawn from the 1994 U.S. Census database. Each record includes 14 features such as age, workclass, education, marital status, occupation, relationship, race, sex, hours-per-week, and native country. The target label is salary, binarized to indicate whether income is greater than $50K or at most $50K. An 80/20 train-test split was applied, yielding roughly 26,048 training samples. Categorical features were encoded using a one-hot encoder and the label was binarized using a label binarizer, both fit exclusively on the training set.

## Evaluation Data

The model was evaluated on the held-out 20% test split, consisting of approximately 6,513 samples drawn from the same `census.csv` dataset. The same one-hot encoder and label binarizer fit on the training data were applied to the test set in inference mode to avoid data leakage. Slice-based evaluation was also performed across all unique values of each of the eight categorical features to assess performance consistency across demographic subgroups.

## Metrics

Model performance is evaluated using three classification metrics: precision, recall, and F1 score (beta=1). Precision measures the fraction of positive predictions that are correct. Recall measures the fraction of actual positives that were correctly identified. F1 score is the harmonic mean of precision and recall, providing a balanced view of both.

On the test set, the model achieved the following results:

- **Precision:** 0.7419
- **Recall:** 0.6384
- **F1 Score:** 0.6863

## Ethical Considerations

The Census Income dataset contains sensitive demographic attributes including race, sex, and native country. These features are used as model inputs, which introduces the risk of the model learning and reinforcing historical societal biases present in the data. Slice-based evaluation revealed variation in model performance across demographic subgroups, meaning the model may perform meaningfully better or worse for certain groups. This model should not be used in any context where biased predictions could lead to discriminatory outcomes.

## Caveats and Recommendations

The dataset reflects U.S. Census data from 1994 and may not be representative of current income distributions or labor market conditions. The model has not been audited for fairness or disparate impact across protected classes. Before any real-world deployment, a thorough fairness audit should be conducted, and the model should be retrained on more recent and representative data. Hyperparameter tuning (e.g., via cross-validation) may improve performance beyond the baseline reported here.
