# Sentiment Triplet Extraction Tasks
Aspect Sentiment Triplet Extraction is a mechanism of extracting sentiment in the form of key notable triplets for fine grained sentiment analysis.This mechanism is implemented using advanced transformer neural networks and visulaized by Ngrok API's.
# Requirements
**Note**: For installing the pre-requisites or libraries check requirements.txt file or download the packages as below
* torch==1.9.0+cu102
* matplotlib==3.2.2
* numpy==1.19.5
* networkx==2.6.2
* Werkzeug==1.0.1
* spacy==2.2.4
* Flask==1.1.4
* tqdm==4.62.0
* flask_ngrok==0.0.25
* scikit_learn==0.24.2
* transformers==4.10.2
# Pre-trained and saved models of all Transformer Neural Networks
## Pre-trained Transformer Models
**Note**: All the pre-trained models implemented for this mechanism can be downloaded from below shared links
* [Albert](https://drive.google.com/drive/folders/1f_vjy4g0WnFS1_5k0gojvCRQvmLC018e?usp=sharing)
* [Bert](https://drive.google.com/drive/folders/1XFd6dUwKyOcE9tR9yGh3ynaX809UZA2t?usp=sharing)
* [Distil_bert](https://drive.google.com/drive/folders/1rSHNn9ABy356cq3UZe3kTQBW3WzBhB4U?usp=sharing)
* [Roberta](https://drive.google.com/drive/folders/1-WrFAmUM368OKq1LUNzHzUSyKwaESF5B?usp=sharing)
* [Xlnet](https://drive.google.com/drive/folders/1whKsduaaHxCZxGYXfbpO97jN2rDGQvx3?usp=sharing)
## Saved Models fine tuned for chosen datasets
**Note**: All the saved models implemented and fine tuned for both pair extraction and triplet extraction in this mechanism can be downloaded from below shared links
* [Albert](https://drive.google.com/drive/folders/1HJPrLCWHsoupoRKtDh23W0C-DMepSohq?usp=sharing)
* [Bert](https://drive.google.com/drive/folders/1-drrVBZFMKHdEnXtQ54n_eYQOxW17Mvn?usp=sharing)
* [Distil-Bert](https://drive.google.com/drive/folders/1-DTyyPm5SNZWc6rSRhMycLyLQHTqYSFA?usp=sharing)
* [Roberta](https://drive.google.com/drive/folders/1-_CoRNGbM75O2aoGgsHqLED3fV8KLi6C?usp=sharing)
* [Xlnet](https://drive.google.com/drive/folders/1ld0p9z4qJNou8Nsa6mCzEw8Ij99WanaG?usp=sharing)
## Training the transformer models
* Pair Extraction Task
```
    python main.py -- Model model name --task pair --mode train --dataset dataset name
```
* Triplet Extraction Task
```
    python main.py -- Model model name --task triplet --mode train --dataset dataset name
```
## Testing the transformer models
* Pair Extraction Task
```
    python main.py -- Model model name --task pair --mode test --dataset dataset name
```
* Triplet Extraction Task
```
    python main.py -- Model model name --task triplet --mode train --dataset dataset name
```
## Predicting the Sentiment Triplets
```
    python main.py -- Model model name --task triplets --mode predict --dataset dataset name
```
## Visualizing the Sentiment Triplets
```
    python app.py -- Model model name --task triplet --mode predict --dataset dataset name
```
## Install Ngrok for Local Web-App
```
!pip install flask_ngrok
```
* After installation set up a tunnel through the local end point of the Web-App
* Authenticate as ngrok user with  provided auth-token id
```
ngrok authtoken token id
```
## Visualization of Sentiment Triplets Using a Publicly Exposed API
![image]()
