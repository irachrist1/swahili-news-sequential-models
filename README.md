# Swahili News Classification with Sequential Models

This repository classifies Swahili news articles into six topics with five models, from TF-IDF baselines to a fine-tuned AfriBERTa. The Colab notebook runs the full pipeline and produces every table and figure in the report.

The six topics are *uchumi* (economy), *kitaifa* (national), *michezo* (sports), *kimataifa* (international), *burudani* (entertainment) and *afya* (health). The data is the Swahili News Classification dataset (Davis, 2020), the corpus used in the Zindi AI4D Swahili News Classification Challenge.

## Results

| Model | Test macro-F1 | 95% CI | Accuracy | Log loss |
|---|---|---|---|---|
| Naive Bayes | 0.689 | 0.674–0.703 | 0.868 | 0.623 |
| Logistic Regression | 0.817 | 0.805–0.829 | 0.889 | 0.333 |
| BiLSTM + attention | 0.795 | 0.782–0.809 | 0.882 | 0.379 |
| TextCNN | 0.791 | 0.777–0.806 | 0.882 | 0.381 |
| AfriBERTa | **0.868** | 0.857–0.879 | 0.925 | 0.289 |

Test set: 7,303 articles. Full comparison, per-class scores and significance tests are in the [report](report/report.pdf).

Macro-F1 is the main metric because the classes are imbalanced 13 to 1. Every experiment, the change it made and the reason for it are listed in [`results/experiment_log.csv`](results/experiment_log.csv).

## Run on Google Colab

Open [`notebooks/run_on_colab.ipynb`](notebooks/run_on_colab.ipynb) in Colab, set the runtime to a T4 GPU and run all cells. The notebook clones this repository, installs the requirements, downloads the data and runs each step. The outputs of our own runs are already committed, so any training cell can be skipped.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd src
python eda.py                              # figures/eda_*.png, results/eda_summary.json
python baselines.py                        # Naive Bayes and Logistic Regression
python train_neural.py --model bilstm
python train_neural.py --model textcnn
python train_transformer.py                # AfriBERTa-small
python compare_models.py                   # comparison tables, ROC, calibration, significance
python error_analysis.py
```

Data (about 40 MB) and fastText vectors (about 230 MB, BiLSTM and TextCNN only) download on first use into `data/`.

## Repository layout

```
src/
  config.py            paths, labels, seed, device
  data.py              download, remove leaks and duplicates, stratified validation split
  preprocess.py        glued-sentence and ligature repair, cleaning modes
  eda.py               exploratory analysis
  metrics.py           metrics, experiment log, confusion matrix and learning-curve plots
  baselines.py         TF-IDF + Naive Bayes, TF-IDF + Logistic Regression
  sequence_data.py     vocabulary, length-bucketed batches, fastText loading
  neural_models.py     BiLSTM with attention, TextCNN
  train_neural.py      step-by-step experiment plans for BiLSTM and TextCNN
  train_transformer.py AfriBERTa-small fine-tuning
  compare_models.py    test-set comparison and McNemar tests
  error_analysis.py    errors by length, rare words, code-switching; hard examples
notebooks/             Colab notebook
results/               experiment log, metrics, test predictions, analysis tables
figures/               every figure in the report
report/                the research report (PDF)
```

## Team

| Member | Role |
|---|---|
| Gentil Tonny Christian Iradukunda | Data pipeline, EDA, AfriBERTa, report |
| Dan Paul Dushime | Baselines, evaluation metrics, model comparison |
| Rene Pierre Ntabana | BiLSTM, Colab notebook |
| Thierry Alain Tresor Ibyishaka | TextCNN, error analysis |

## References

Davis, D. (2020). *Swahili: News classification dataset* (Version 0.2) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5514203

Ogueji, K., Zhu, Y., & Lin, J. (2021). Small data? No problem! Exploring the viability of pretrained multilingual language models for low-resourced languages. *Proceedings of the 1st Workshop on Multilingual Representation Learning*, 116–126.
