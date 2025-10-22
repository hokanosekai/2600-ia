# 2600 IA

## Démarrage

```sh
python gpt.py --train-file Training.parquet --cv 2 --model hgb --out student_model.skio
```

```sh
python gemini.py 
```

```sh
python mistral.py 
```


## Résultats

```
python gpt.py --train-file Training.parquet --cv 2 --model hgb --out student_model.skio

CV accuracy: 0.9867 ± 0.0006 | folds: [0.98737569 0.98610382]
Hold‑out accuracy: 0.9871
              precision    recall  f1-score   support

      Benign       0.98      1.00      0.99   1006067
      Botnet       1.00      0.99      0.99     20435
  Bruteforce       1.00      1.00      1.00     14454
        DDoS       1.00      0.99      1.00    172862
         DoS       0.99      0.98      0.99     55628
Infiltration       0.44      0.02      0.03     13280
    Portscan       0.73      0.78      0.76       316
   Webattack       0.77      0.81      0.79       419

    accuracy                           0.99   1283461
   macro avg       0.86      0.82      0.82   1283461
weighted avg       0.98      0.99      0.98   1283461
```

## Tests

```sh
python test_model.py --data Training.parquet --model gpt.skio
```