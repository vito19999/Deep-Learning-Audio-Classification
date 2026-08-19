# Deep-Learning-Audio-Classification
Riconoscimento di versi animali tramite Deep Learning (PyTorch). Trasformazione audio in Spettrogrammi Mel e classificazione tramite CNN.    Animal sound recognition via Deep Learning (PyTorch). Audio to Mel Spectrograms conversion and CNN classification


**Nota sui Dati (File troppo pesanti per l'upload)**
Il dataset custom originale (composto da 50.000 *mixture* audio) e le matrici degli spettrogrammi convertiti in formato NumPy (`.npz`) sono file molto pesanti. A causa dei limiti strutturali di caricamento di GitHub, non possono essere inseriti nel repository. Per rispettare le best practice, le cartelle dei dati grezzi e processati sono ignorate tramite il file `.gitignore`. I file presenti in questa repository includono unicamente il codice sorgente, i notebook di addestramento, i metadati e la presentazione del progetto.


**Versione Italiana**

Questo progetto implementa una pipeline di **Machine Learning e Deep Learning** in PyTorch per il riconoscimento automatico dei versi di 12 diverse specie animali. Modelli di Deep Learning complessi non lavorano direttamente sulle onde sonore, ma su rappresentazioni visive: l'obiettivo del progetto è trasformare i segnali in **Spettrogrammi Mel** e classificarli tramite Reti Neurali Convoluzionali (CNN).

###  Pipeline dei Dati e Architettura

**1. Creazione del Dataset (Audio Mixtures):**
Il progetto non usa campioni puliti, ma risolve un problema reale creando 50.000 *Mixture Audio* personalizzate, unendo dataset pubblici come ESC-50, UrbanSound8k, Normaltargz e Soundscapes. Ogni mix fonde un rumore di background (es. vento, traffico) e un evento target (animale). Il set è bilanciato: 80% Training (30.000 mix), 10% Test (10.000 mix) e 10% Validation. 

**2. Feature Engineering (Generazione Spettrogrammi):**
L'audio è stato elaborato in formato Mono con un *Sample Rate* di 16 kHz. Tramite la libreria `librosa`, le frequenze sono state estratte e convertite in Decibel con i seguenti parametri tecnici:
* **Mel Bins**: 128
* **FFT Size**: 2048
* **Hop Length**: 512

Il risultato per ogni clip da 10 secondi è un tensore normalizzato di dimensione `(1, 128, 313)`.

**3. Architettura CNN (Convolutional Neural Network):**
Sono state sviluppate e testate 4 varianti di CNN in PyTorch:
* Incremento progressivo dei filtri nei blocchi convoluzionali: 32 -> 64 -> 128 -> 256.
* Classificazione con layer *Fully Connected* (12 classi), ottimizzatore Adam e Cross-Entropy Loss.
* Il modello vincitore (**Modello 2**) ha rimosso la BatchNorm in favore di `MaxPool2d` e `Dropout(0.10)`, allenato per 100 Epoche con un Learning Rate di 0.0001.

###  Risultati
Il **Modello 2** ha raggiunto un'**Accuratezza sul Test Set dell'80.5%**. 
Dall'analisi della *Confusion Matrix* si nota un'eccellente capacità di distinguere suoni complessi (es. gallo 97.3%, cane 96.1%), con fisiologiche sovrapposizioni su suoni simili (es. uccellini e insetti).

###  Struttura del Repository e File
* `metadata/`: Contiene i file CSV ed Excel originali (ESC-50) con la mappatura delle etichette e le metriche di accuratezza della *baseline* umana.
* `src/`: Contiene gli script Python puri, come `spectogram_creation.py` (per generare a blocchi gli spettrogrammi salvati in `.npz`), `models.py` e `train.py`.
* `notebooks/`: Notebook Jupyter interattivi contenenti DataLoaders, loop di training con *Early Stopping* (`testing.ipynb`) e test/plot esplorativi.

---
 **English Version**

This project implements an end-to-end **Deep Learning pipeline** in PyTorch for the automatic recognition of 12 different animal species. Since complex models perform better on visual representations rather than raw waveforms, the audio signals are transformed into **Mel Spectrograms** and classified using custom Convolutional Neural Networks (CNNs).

###  Data Pipeline & Architecture

**1. Dataset Creation (Audio Mixtures):**
We created 50,000 custom *Audio Mixtures* by combining public datasets such as ESC-50, UrbanSound8k, Normaltargz, and Soundscapes. Each 10-second mix blends random background noise (e.g., wind, traffic) with the target animal event. The dataset is split into 80% Training, 10% Test, and 10% Validation.

**2. Feature Engineering (Spectrogram Generation):**
Audio was processed in Mono at a 16 kHz Sample Rate. Using `librosa`, frequencies were mapped to a Mel scale and converted to Decibels with the following parameters: 
* **128 Mel Bins**
* **2048 FFT Size**
* **512 Hop Length**

This yields a normalized input tensor of shape `(1, 128, 313)`.

**3. CNN Architecture:**
We tested 4 PyTorch CNN variants:
* Progressive filter increment across convolutional blocks: 32 -> 64 -> 128 -> 256.
* Final Fully Connected layer for 12 classes, Adam optimizer, and Cross-Entropy Loss.
* The winning architecture (**Model 2**) dropped BatchNorm in favor of `MaxPool2d` and `Dropout(0.10)`, trained for 100 Epochs (LR=0.0001).

###  Results
**Model 2** achieved an **80.5% Test Set Accuracy**. 
The *Confusion Matrix* analysis shows excellent pattern recognition on specific signals (e.g., rooster 97.3%, dog 96.1%), with minor overlaps on inherently similar frequencies (like chirping birds and insects).

###  Repository Structure & How to Run
* `metadata/`: Contains the original ESC-50 CSV and Excel files with label mappings and human baseline accuracy metrics.
* `src/`: Pure Python scripts, such as `spectogram_creation.py` (to generate and save Numpy spectrogram matrices in `.npz`).
* `notebooks/`: Jupyter Notebooks containing DataLoaders, training loops with *Early Stopping* (`testing.ipynb`), and exploratory plots.
