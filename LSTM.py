import torch
import torch.nn as nn
from datetime import date
from datetime import datetime
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from skorch import NeuralNet
import pandas as pd
from skorch.callbacks import LRScheduler
from sklearn.model_selection import GridSearchCV
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import joblib

# pour installer les packages 
# conda install pytorch==1.9.0 cudatoolkit=11.3 -c pytorch
# pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# conda install -c anaconda scikit-learn
# pip install -U scikit-learn

# conda install -c conda-forge skorch
# pip install -U skorch

# Fonction qui traite le DataFrame, il est subit la normalisation standard
# Input : 
#     - un DataFrame qu'on souhaite normaliser
# Output : 
#     - le DataFrame normaliser
#     - la moyenne (ou les moyennes) du DataFrame
#     - l'écart-type (ou les écarts-types) du DataFrame
def prepross_data(data):
    #enlève les valeurs vides
    data.dropna(axis=0,inplace=True)
    mean = data.mean()
    std = data.std()
    data =(data-mean)/std
    return mean,std,data

#Constante

numFeature = 5


# Fonction qui créer un vecteur x à partir d'un index et d'un DataFrame
# Input : 
#     - l'index sur la première valeur du vecteur
#     - un DataFrame contenant toutes les valeurs
# Output : un vecteur x, contenant les valeur de cloture du bitcoin des N prochaines minutes à partir de l'index. 
def create_input(index,pdData):
    return pdData["cloture"].iloc[index:index+numFeature+1].values.tolist()


# Méthode qui permet de convertir un index de notre DataFrame en un dateTime
def convert_index_to_datetime(index):
    #la première date de notre base de donnée
    date = datetime.datetime.strptime(str(index)[:6],"%y%m%d")
    hour = int(str(index)[7:-1])//60
    minute = int(str(index)[7:-1])%60
    index_datetime = date + datetime.timedelta(hours=hour) + datetime.timedelta(minutes=minute)
    return index_datetime


#Methode qui créer un batch contenant l'ensemble des vecteurs et des labels associées utilisé lors de l'apprentissage.
# input: DataFrame contenant les données
# output: le batch contenant tous les vectueurs du DataFrame avec les labels
def create_batchs_and_labels(data):

    iterator = 0
    labels =[]
    batch =  []
    #On boucle sur la base de donnée :
    while (iterator <= data.shape[0]-numFeature-1):   

        dataExtracted = create_input(iterator,data)
        if (len(dataExtracted) == numFeature+1):
            sequence = []
            for x in dataExtracted[:-1]:
                sequence.append([x])
            batch.append(sequence)
            labels.append([dataExtracted[-1]])
        iterator+=1
    return torch.FloatTensor(batch),torch.FloatTensor(labels)

class Dataset(Dataset):    
    def __init__(self, data_file):
        pf = pd.read_csv(str(data_file),sep=';',index_col='time')
        self.mean,self.std,self.prepross_pf = prepross_data(pf)
        self.batch,self.labels = create_batchs_and_labels(self.prepross_pf)
        
    def __len__(self):
        return self.batch.shape[0]

    def __getitem__(self, idx):
        return self.batch[idx],self.labels[idx]
    
# training = Dataset("./bitcoin.csv")
training = Dataset("./data/train/bitcoin.csv")
train_dataloader = DataLoader(training, batch_size=10000, shuffle=True)

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=300, output_size=1,num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size,hidden_layer_size,num_layers,batch_first = True)
        # linear layer pour préduire 
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_batch):
        self.hidden_cell = (torch.zeros(self.num_layers,input_batch.size(0),self.hidden_layer_size,device=input_batch.device),
                            torch.zeros(self.num_layers,input_batch.size(0),self.hidden_layer_size,device=input_batch.device))
        lstm_out, self.hidden_cell = self.lstm(input_batch, self.hidden_cell)
        predictions = self.linear(lstm_out[:,-1,:])
        return predictions
    
net_regr = NeuralNet(
    LSTM,
    optimizer=torch.optim.Adam,
    max_epochs=1000,
    lr=0.001,
    callbacks=[
        LRScheduler(policy='StepLR', step_size=100, gamma=0.1)
    ],
    batch_size=50000,
    iterator_train__shuffle=True,
    criterion=nn.MSELoss,
    device='cuda' 
)

torch.cuda.empty_cache()

net_regr.fit(training)

# Sauvegarde du modèle 
saveModel = datetime.today().strftime('%Y%m%d_%H_%M')
# cf doc for more information https://skorch.readthedocs.io/en/stable/user/save_load.html
net_regr.save_params(f_params='./resultat/save_models/'+saveModel+'.pkl')

# loading need to initialize net
# net_regr.initialize()  # This is important!
# net_regr.load_params(f_params='some-file.pkl')


############## chargement du model ##############

# model_name = "trained_model"

# print("load model")
# net_regr.initialize()  # This is important!
# net_regr.load_params(f_params='../resultat/save_models/'+model_name+'.pkl')

# sans charger un modéle existant
# net_regr = grid.best_estimator_

################################################

#Fonction qui prédit à un pas de temps à partir des valeurs réelles

def prediction_un_pas_temps(reel_value,numberData):
    
    #on a besoin des numFeature première valeur car on va se baser sur elles pour faire la première prédiction
    value_predict = reel_value[:numFeature].tolist()
    
    for index in range(0,numberData-numFeature):
        
        #sequence to make Tensor for the prediction with the model
        prediction_sequence = [[]]
        for value in reel_value[index:index+numFeature]:
            prediction_sequence[0].append([value])
        prediction_vector = torch.FloatTensor(prediction_sequence)
        predicted_value = net_regr.predict(prediction_vector).item()
        value_predict.append(predicted_value)
        
    return value_predict

#Prédicateur naif qui prédit la valeur précédante.
#La première valeur ne sera pas prédite

def prediction_naif(reel_value,numberData):
    
    #la première valeur ne sera pas prédit car il faut une valeur minimum.
    value_predict = reel_value[:1].tolist()
    
    for index in range(0,numberData-1):
        
        value_predict.append(reel_value[index])
        
    return value_predict

#Fonction qui prédit à plusieurs pas de temps à partir des valeurs réelles

def prediction_plusieurs_pas_temps(first_reel_value,numberData):
    
    #on a besoin des numFeature première valeur car on va se baser sur elles pour faire la première prédiction
    value_predict = first_reel_value.tolist()
    
    for index in range(0,numberData-numFeature):

        #sequence to make Tensor for the prediction with the model
        prediction_sequence = [[]]
        for value in value_predict[index:index+numFeature]:
            prediction_sequence[0].append([value])
        prediction_vector = torch.FloatTensor(prediction_sequence)
        predicted_value = net_regr.predict(prediction_vector).item()
        value_predict.append(predicted_value)
    return value_predict

def calcul_metrics(reel_value,estimate_value,numberData):
    erreurs = reel_value-estimate_value
    mse = np.sum(np.power(erreurs,2))/numberData
    mae = np.sum(np.abs(erreurs))/numberData
    rmse = np.sqrt(mse)
    metric = "Differente métrique pour la prédiction à un pas de temps: \n"
    metric += "Le carré moyen des erreurs MSE :" + str(mse) + "\n"
    metric += "l'erreur moyenne absolue MAE :" + str(mae) + "\n"
    metric += "L’erreur quadratique moyenne RMSE :" + str(rmse) + "\n\n\n"
    return metric

def affichage(reel_value,estimate_value,name,metrics):
    plt.figure(figsize=(18,12))
    plt.title('cours du bitcoin du : '+ name)
    plt.ylabel('valeur en euro')
    plt.xlabel('durée en minutes')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    plt.plot(reel_value,label="valeur réel")
    plt.plot(estimate_value,label="valeur prédict",ls='--')
    plt.legend()
    plt.figtext(0.11, -0.01, metrics )
    plt.savefig('./resultat/predictions/'+name+'.jpg')

def retransforme(values,std,mean):
    return values * std + mean

files = os.listdir("../data/test/")
for filename in files:
    print("\nTest file : "+filename)
    date_str = filename[10:-4]
    filename = "./data/test/"+filename
    
    #clear and prepross data
    newData = pd.read_csv(filename,sep=';')
    numberData=newData["cloture"].size
    print("nombre de donnée :",numberData)

    mean,std,newData = prepross_data(newData)
    
    #add prediction to data frame
    newData['valeur_predict_un_pas'] = prediction_un_pas_temps(newData['cloture'].to_numpy(),numberData)
    newData['valeur_predict_plusieurs_pas'] = prediction_plusieurs_pas_temps(newData['cloture'][:numFeature].to_numpy(),numberData)
    newData['valeur_predict_naif'] = prediction_naif(newData['cloture'].to_numpy(),numberData)
    
    print(newData.head(10))
    
    #retransforme data and display it
    newData = retransforme(newData,std["cloture"],mean["cloture"])
    
    #calcul métrique prédiction à un pas de temps
    metric = calcul_metrics(newData['cloture'].to_numpy(),newData['valeur_predict_un_pas'].to_numpy(),numberData)
    affichage(newData['cloture'].to_numpy(),newData['valeur_predict_un_pas'].to_numpy(),"2022 08 24 avec un pas temps",metric)
    
    #calcul métrique prédiction naïve
    metric = calcul_metrics(newData['cloture'].to_numpy(),newData['valeur_predict_naif'].to_numpy(),numberData)
    affichage(newData['cloture'].to_numpy(),newData['valeur_predict_naif'].to_numpy(),"2022 08 24 prediction naïve",metric)
    
    #calcul prédiction sur une journée
    metric = calcul_metrics(newData['cloture'].to_numpy(),newData['valeur_predict_plusieurs_pas'].to_numpy(),numberData)
    affichage(newData['cloture'].to_numpy(),newData['valeur_predict_plusieurs_pas'].to_numpy(),"2022 08 24 plusieurs pas temps",metric)
    
    
#     
