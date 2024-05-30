## LncLSTA
This project is a PyTorch Lightning implementation of a machine learning model for RNA sequence analysis. It uses a Long Short Transformer model to analyze RNA sequences and predict their subcellular localization.

### Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

Prerequisites
The project requires the following Python packages:
```
einops==0.4.1
gensim==4.1.2
matplotlib==3.5.1
numpy==1.22.0
pandas==1.3.1
pytorch_lightning==1.6.4
rotary_embedding_torch==0.1.5
scikit_learn==1.5.0
scipy==1.8.0
seaborn==0.11.2
torch==1.12.0
tqdm==4.62.0
```

You can install these packages using pip:  

`pip install -r requirements.txt` 


### Running the Project
To run the project, execute the main.py:  

`python main.py`

### Data availability
The /prepared folder provides the training dataset, validation dataset, and independent test set.

### Project Structure
The project has the following structure:  

**Config.py**: Configuration file, contains all the configuration information of the project. 

**dataset.py**: Contains the processing logic of the dataset.   

**main.py**: The main entry file of the project.   

**MetricTracker.py**: Used to track and record various indicators during the training process.

**model/**: Contains all the model files.   

**train_step.py**: Defines the training steps of the model.  


