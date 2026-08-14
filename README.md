# Aspect-Based Sentiment Classification: Improving Cross-Dataset Generalisation through Aspect Marking

Comparing SVM, BiLSTM, BERT, and RoBERTa on aspect-based sentiment classification, evaluating how well each architecture generalises from a standard benchmark (SemEval-2014) to a harder, multi-aspect dataset (MAMS). A simple input-level technique, aspect marking, is then introduced and tested as a way to close that generalisation gap.

## Overview

Aspect-Based Sentiment Classification (ABSC) identifies sentiment toward specific aspects within a sentence, rather than assigning one overall label (e.g. "the food was great but the service was slow" has different sentiment for food vs service). While transformer models like BERT and RoBERTa perform well on standard benchmarks, this project investigates whether that performance holds up on harder, multi-aspect sentences with conflicting sentiments, and whether a lightweight preprocessing technique, aspect marking, can improve generalisation without any architectural changes.

## Datasets

- **SemEval-2014** — available on Kaggle: https://www.kaggle.com/datasets/charitarth/semeval-2014-task-4-aspectbasedsentimentanalysis
- **MAMS** (Multi-Aspect Multi-Sentiment) — available on GitHub: https://github.com/siat-nlp/MAMS-for-ABSA/tree/master/data

SemEval was used for training and in-domain validation. MAMS was used for zero-shot cross-dataset testing, since it was specifically designed to contain sentences with conflicting sentiments across multiple aspects, making it a stronger test of genuine aspect-level reasoning.

## Models Compared

- SVM (TF-IDF features) — statistical baseline
- BiLSTM — sequential deep learning baseline
- BERT — transformer baseline
- RoBERTa — transformer baseline, then extended with aspect marking

## Method

- Established baseline performance for all four models on SemEval-2014 (Restaurant and Laptop domains)
- Tested each model zero-shot on MAMS to measure the in-domain-to-cross-dataset performance drop (the "generalisation gap")
- Applied aspect marking (wrapping the target aspect in `<aspect>` / `</aspect>` boundary tokens) to RoBERTa, the strongest baseline, to test whether explicit attention guidance improves generalisation
- Conducted detailed error analysis on the aspect-marked model to categorise remaining failure modes

## Key Results

- **Baseline progression:** SVM (0.6343 macro F1) → BiLSTM → BERT (0.7544) → RoBERTa (0.7791) on SemEval validation, with the largest single jump (+19.0% relative) coming from the shift to transformer architectures
- **Generalisation gap:** every model's performance dropped moving from SemEval to MAMS. BERT showed a smaller relative drop (25.5%) than RoBERTa (27.4%), despite RoBERTa's stronger in-domain score, showing that the best in-domain model isn't automatically the most robust one
- **Aspect marking impact:** improved average cross-dataset (MAMS) F1 by **10.6%**, and reduced the generalisation gap by **27.1%**, with no meaningful cost to in-domain accuracy
- **Domain-dependent effect ("rescue pattern"):** aspect marking helped far more where the baseline struggled (Laptop: +19.8%) than where it was already strong (Restaurant: +4.1%), suggesting it acts as a rescue technique for weaker deployment scenarios rather than a uniform booster
- **Remaining weakness:** error analysis showed 78.5% of remaining errors involved misclassifying neutral sentiment, indicating aspect marking solves the "where to look" problem but not the "how to interpret it" problem

## Tech Stack

Python, PyTorch/TensorFlow, scikit-learn, Pandas, NumPy, Hugging Face Transformers, Keras

## How to Run

1. Download SemEval-2014 and MAMS from the links above
2. Open `[notebook name].ipynb` in Jupyter or Google Colab
3. If running in Colab, mount your Google Drive and update the dataset paths to point to where you've saved SemEval and MAMS locally (e.g. `/content/drive/MyDrive/your-folder/`)
4. Run all cells to preprocess, train, and evaluate each model, including the aspect-marked RoBERTa variant

## Notes

All reported results are from single training runs using a fixed random seed (42), due to GPU constraints. See the full dissertation report for detailed methodology, per-class breakdowns, and qualitative error analysis.
