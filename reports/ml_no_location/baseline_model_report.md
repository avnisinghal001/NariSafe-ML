# NariSafe Baseline Model Report

## Important notes

- `risk_level` is a rule-based proxy label, not real incident-level ground truth.
- `risk_score` is not used as an input feature to avoid target leakage.
- Random split is useful for quick baseline checking.
- Group-city split is more realistic because it tests unseen cities.
- Macro F1 is more important than plain accuracy because the `high` risk class is small.

## Model summary

| Split | Model | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 | Test Weighted F1 |
|---|---|---:|---:|---:|---:|---:|
| group_city | hist_gradient_boosting | 0.8296 | 0.7502 | 0.8821 | 0.8051 | 0.8808 |
| group_city | random_forest | 0.7827 | 0.7125 | 0.8479 | 0.7920 | 0.8480 |
| group_city | logistic_regression | 0.6340 | 0.5709 | 0.6710 | 0.6481 | 0.6759 |
| group_city | dummy_most_frequent | 0.6709 | 0.2677 | 0.6383 | 0.2598 | 0.4974 |
| random | hist_gradient_boosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| random | random_forest | 0.9953 | 0.9860 | 0.9948 | 0.9847 | 0.9949 |
| random | logistic_regression | 0.9395 | 0.9132 | 0.9405 | 0.9154 | 0.9413 |
| random | dummy_most_frequent | 0.5876 | 0.2467 | 0.5876 | 0.2467 | 0.4350 |

## Detailed results

### group_city / hist_gradient_boosting

Validation confusion matrix:

Labels order: `low`, `medium`, `high`

```text
[[17605, 2905, 0], [7182, 50617, 189], [0, 4452, 3486]]
```

Validation classification report:

```text
              precision    recall  f1-score   support

         low       0.71      0.86      0.78     20510
      medium       0.87      0.87      0.87     57988
        high       0.95      0.44      0.60      7938

    accuracy                           0.83     86436
   macro avg       0.84      0.72      0.75     86436
weighted avg       0.84      0.83      0.83     86436

```

Test confusion matrix:

Labels order: `low`, `medium`, `high`

```text
[[30359, 1015, 0], [7826, 55895, 651], [0, 2401, 2695]]
```

Test classification report:

```text
              precision    recall  f1-score   support

         low       0.80      0.97      0.87     31374
      medium       0.94      0.87      0.90     64372
        high       0.81      0.53      0.64      5096

    accuracy                           0.88    100842
   macro avg       0.85      0.79      0.81    100842
weighted avg       0.89      0.88      0.88    100842

```

### group_city / random_forest

Validation confusion matrix:

Labels order: `low`, `medium`, `high`

```text
[[17829, 2681, 0], [11295, 46408, 285], [0, 4518, 3420]]
```

Validation classification report:

```text
              precision    recall  f1-score   support

         low       0.61      0.87      0.72     20510
      medium       0.87      0.80      0.83     57988
        high       0.92      0.43      0.59      7938

    accuracy                           0.78     86436
   macro avg       0.80      0.70      0.71     86436
weighted avg       0.81      0.78      0.78     86436

```

Test confusion matrix:

Labels order: `low`, `medium`, `high`

```text
[[28176, 3198, 0], [9520, 54493, 359], [0, 2263, 2833]]
```

Test classification report:

```text
              precision    recall  f1-score   support

         low       0.75      0.90      0.82     31374
      medium       0.91      0.85      0.88     64372
        high       0.89      0.56      0.68      5096

    accuracy                           0.85    100842
   macro avg       0.85      0.77      0.79    100842
weighted avg       0.86      0.85      0.85    100842

```

### group_city / logistic_regression

Validation confusion matrix:

Labels order: `low`, `medium`, `high`

```text
[[10185, 7140, 3185], [8470, 37597, 11921], [0, 917, 7021]]
```

Validation classification report:

```text
              precision    recall  f1-score   support

         low       0.55      0.50      0.52     20510
      medium       0.82      0.65      0.73     57988
        high       0.32      0.88      0.47      7938

    accuracy                           0.63     86436
   macro avg       0.56      0.68      0.57     86436
weighted avg       0.71      0.63      0.65     86436

```

Test confusion matrix:

Labels order: `low`, `medium`, `high`

```text
[[26537, 4837, 0], [22547, 36862, 4963], [231, 602, 4263]]
```

Test classification report:

```text
              precision    recall  f1-score   support

         low       0.54      0.85      0.66     31374
      medium       0.87      0.57      0.69     64372
        high       0.46      0.84      0.60      5096

    accuracy                           0.67    100842
   macro avg       0.62      0.75      0.65    100842
weighted avg       0.75      0.67      0.68    100842

```

### group_city / dummy_most_frequent

Validation confusion matrix:

Labels order: `low`, `medium`, `high`

```text
[[0, 20510, 0], [0, 57988, 0], [0, 7938, 0]]
```

Validation classification report:

```text
              precision    recall  f1-score   support

         low       0.00      0.00      0.00     20510
      medium       0.67      1.00      0.80     57988
        high       0.00      0.00      0.00      7938

    accuracy                           0.67     86436
   macro avg       0.22      0.33      0.27     86436
weighted avg       0.45      0.67      0.54     86436

```

Test confusion matrix:

Labels order: `low`, `medium`, `high`

```text
[[0, 31374, 0], [0, 64372, 0], [0, 5096, 0]]
```

Test classification report:

```text
              precision    recall  f1-score   support

         low       0.00      0.00      0.00     31374
      medium       0.64      1.00      0.78     64372
        high       0.00      0.00      0.00      5096

    accuracy                           0.64    100842
   macro avg       0.21      0.33      0.26    100842
weighted avg       0.41      0.64      0.50    100842

```

### random / hist_gradient_boosting

Validation confusion matrix:

Labels order: `low`, `medium`, `high`

```text
[[28433, 0, 0], [0, 46050, 0], [0, 0, 3886]]
```

Validation classification report:

```text
              precision    recall  f1-score   support

         low       1.00      1.00      1.00     28433
      medium       1.00      1.00      1.00     46050
        high       1.00      1.00      1.00      3886

    accuracy                           1.00     78369
   macro avg       1.00      1.00      1.00     78369
weighted avg       1.00      1.00      1.00     78369

```

Test confusion matrix:

Labels order: `low`, `medium`, `high`

```text
[[35540, 0, 0], [0, 57563, 0], [0, 0, 4858]]
```

Test classification report:

```text
              precision    recall  f1-score   support

         low       1.00      1.00      1.00     35540
      medium       1.00      1.00      1.00     57563
        high       1.00      1.00      1.00      4858

    accuracy                           1.00     97961
   macro avg       1.00      1.00      1.00     97961
weighted avg       1.00      1.00      1.00     97961

```

### random / random_forest

Validation confusion matrix:

Labels order: `low`, `medium`, `high`

```text
[[28433, 0, 0], [72, 45682, 296], [0, 0, 3886]]
```

Validation classification report:

```text
              precision    recall  f1-score   support

         low       1.00      1.00      1.00     28433
      medium       1.00      0.99      1.00     46050
        high       0.93      1.00      0.96      3886

    accuracy                           1.00     78369
   macro avg       0.98      1.00      0.99     78369
weighted avg       1.00      1.00      1.00     78369

```

Test confusion matrix:

Labels order: `low`, `medium`, `high`

```text
[[35540, 0, 0], [99, 57058, 406], [0, 0, 4858]]
```

Test classification report:

```text
              precision    recall  f1-score   support

         low       1.00      1.00      1.00     35540
      medium       1.00      0.99      1.00     57563
        high       0.92      1.00      0.96      4858

    accuracy                           0.99     97961
   macro avg       0.97      1.00      0.98     97961
weighted avg       1.00      0.99      0.99     97961

```

### random / logistic_regression

Validation confusion matrix:

Labels order: `low`, `medium`, `high`

```text
[[27368, 1065, 0], [2320, 42403, 1327], [0, 27, 3859]]
```

Validation classification report:

```text
              precision    recall  f1-score   support

         low       0.92      0.96      0.94     28433
      medium       0.97      0.92      0.95     46050
        high       0.74      0.99      0.85      3886

    accuracy                           0.94     78369
   macro avg       0.88      0.96      0.91     78369
weighted avg       0.94      0.94      0.94     78369

```

Test confusion matrix:

Labels order: `low`, `medium`, `high`

```text
[[34269, 1271, 0], [2939, 53054, 1570], [0, 47, 4811]]
```

Test classification report:

```text
              precision    recall  f1-score   support

         low       0.92      0.96      0.94     35540
      medium       0.98      0.92      0.95     57563
        high       0.75      0.99      0.86      4858

    accuracy                           0.94     97961
   macro avg       0.88      0.96      0.92     97961
weighted avg       0.94      0.94      0.94     97961

```

### random / dummy_most_frequent

Validation confusion matrix:

Labels order: `low`, `medium`, `high`

```text
[[0, 28433, 0], [0, 46050, 0], [0, 3886, 0]]
```

Validation classification report:

```text
              precision    recall  f1-score   support

         low       0.00      0.00      0.00     28433
      medium       0.59      1.00      0.74     46050
        high       0.00      0.00      0.00      3886

    accuracy                           0.59     78369
   macro avg       0.20      0.33      0.25     78369
weighted avg       0.35      0.59      0.43     78369

```

Test confusion matrix:

Labels order: `low`, `medium`, `high`

```text
[[0, 35540, 0], [0, 57563, 0], [0, 4858, 0]]
```

Test classification report:

```text
              precision    recall  f1-score   support

         low       0.00      0.00      0.00     35540
      medium       0.59      1.00      0.74     57563
        high       0.00      0.00      0.00      4858

    accuracy                           0.59     97961
   macro avg       0.20      0.33      0.25     97961
weighted avg       0.35      0.59      0.43     97961

```
