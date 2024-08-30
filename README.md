# Progetto di Statistica

## Dataset
- **Nome:** Obesity Prediction
- **Fonte:** [Kaggle](https://www.kaggle.com/datasets/mrsimple07/obesity-prediction)

## Pulizia dei Dati
1. **Caricamento e Pulizia:** Rimozione di valori mancanti e duplicati.
2. **Valori Fuori Scala:** Filtraggio di valori anomali basati su deviazione standard.
3. **Variabili Categoriali:** Conversione di `Gender` e `ObesityCategory` in numeriche.

## Analisi Esplorativa dei Dati (EDA)
- **Correlazioni:** Forte correlazione tra altezza e BMI, e BMI e peso.
- **Distribuzioni:** Maggiore incidenza di disordini alimentari tra i 30-60 anni. Equilibrio di genere.

## Modelli e Tuning
1. **Regressione Lineare:** Correlazione tra Peso e BMI.
2. **Classificazione:** 
   - **Logistica:** Migliori parametri trovati tramite Grid Search.
   - **SVM:** Testato con diversi kernel e parametri.

## Risultati
- **Miglior Modello:** Regressione Logistica.
- **Valutazione:** Analisi di performance con matrice di confusione e report di classificazione.
- **Statistiche sui Risultati:** Media, mediana, e deviazione standard del Misclassification Rate (MR).

## Codice
- **File Principale:** `progetto_statistica.py`
- **Dipendenze:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`
