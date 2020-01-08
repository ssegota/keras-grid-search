# -*- coding: utf-8 -*-

import tensorflow
from keras.models import Sequential
from keras.layers import Dense
import keras
#import uuid for uniquely naming solutions
from uuid import uuid4
#from keras.models import model_from_json

#supress warnings because they serve no purpose
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

#deter
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error  
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import max_error
from sklearn.metrics import r2_score
"""
optimizer_get: funkcija za kompilaciju optimizera sa danim parametrima
param_dict - dictionary sa parametrima

Popis keyeva:
sgd
    lr          (float)
    momentum    (float)
    nesterov    (bool) 
rmsprop
    lr          (float)
    rho         (float)
adagrad
    lr          (float)
adadelta
    lr          (float)
    rho         (float)
adam
    lr          (float)
    beta1       (float)
    beta2       (float)
    amsgrad     (bool)
adamax
    lr          (float)
    beta1       (float)
    beta2       (float)
nadam
    lr          (float)
    beta1       (float)
    beta2       (float)
    
"""
def get_optimizers(params):
    optimizer_list = []

    count=0

    print("LENGTH OF PARAMETERES:", len(params))
    for param_dict in params:
        print("OPTIMIZER GENERATION - ", count, "/", len(optimizer_params))
        count -=-1
        optimizer_type = param_dict["types"]
        optimizers= ['sgd', 
                         'adam', 
                         'rmsprop', 
                         'adagrad',
                         'adadelta', 
                         'adamax',
                         'nadam']
        
        if optimizer_type=='sgd':
            optimizer=keras.optimizers.SGD(lr=param_dict["lr"], 
                                          momentum=param_dict["momentum"],
                                          nesterov=param_dict["nesterov"])
            optimizer_list.append(optimizer)
            
        elif optimizer_type=='rmsprop':
            optimizer=keras.optimizers.RMSprop(lr=param_dict["lr"], 
                                               rho=param_dict["rho"])
            optimizer_list.append(optimizer)
            
        elif optimizer_type=='adagrad':
            optimizer=keras.optimizers.Adagrad(lr=param_dict["lr"])
            optimizer_list.append(optimizer)
            
        elif optimizer_type=='adadelta':
            optimizer=keras.optimizers.Adadelta(lr=param_dict["lr"], 
                                                rho=param_dict["rho"])
            optimizer_list.append(optimizer)
            
        elif optimizer_type=='adam':
            optimizer=keras.optimizers.Adam(lr=param_dict["lr"], 
                                            beta_1=param_dict["beta1"],
                                            beta_2=param_dict["beta2"],
                                            amsgrad=param_dict["amsgrad"])
            optimizer_list.append(optimizer)
            
        elif optimizer_type=='adamax':
            optimizer=keras.optimizers.Adamax(lr=param_dict["lr"], 
                                              beta_1=param_dict["beta1"], 
                                              beta_2=param_dict["beta2"])
            optimizer_list.append(optimizer)
            
        elif optimizer_type=='nadam':
            optimizer=keras.optimizers.Nadam(lr=param_dict["lr"], 
                                             beta_1=param_dict["beta1"], 
                                             beta_2=param_dict["beta2"])
            optimizer_list.append(optimizer)
            
        else:
            print("Chosen optimizer unsupported, choose between: ", optimizers)

    return optimizer_list
        
"""
Stvara i trenira modele Koristeći parametre. Trenirani modeli spremljeni u 
    JSON sa težinama u H5.

Argumnti

X                   (Array)                     Ulazni Podaci
Y                   (Array)                     Izlazni Podaci
input_shape         (tuple of ints)             oblik ulaznih podataka 
loss                (list of strings)           lista korištenih lossova 
layers              (list of tuple of Ints)     Različite konfiguracije ANN - 
                                                svaki tuple je niz brojeva koji 
                                                odgovaraju broju neurona u 
                                                svakom skrivenom sloju
activations         (list of tuple of Strings)  lista tupleova iste duljine kao
                                                i gornjih layera, koji sadrže 
                                                aktivacije za svaki od gore 
                                                navedenih slojeva 
adjusted_optimizers (list of keras.Optimizers)  lista optimizera dobivenih 
                                                korištenjem funkcije 
                                                optimizer_get() 
epochs              (list of ints)              lista sa brojvima epoha 
eval_metrics        (list of strings)           lista metrika za evaluaciju 
                                                modela tijekom treniranja
prediction_metric   (string)                    metrika za određivanje 
                                                kvalitete predikcije 
                                                Moguće metrike:
                                                auc - Area Under Curve
                                                accuracy - preciznost
                                                mae - Mean Average Error
                                                mse - Mean Square Error
                                                msle - Mean Square Log Error
                                                max - Maximal Error
                                                r2 - R2 score
verbose             (bool)                      ispis model summarya i training 
                                                informationa


"""
def create_model(X, 
                 Y, 
                 input_shape_, 
                 loss, layers, 
                 activations, 
                 adjusted_optimizers, 
                 epochs, 
                 eval_metrics, 
                 prediction_metric, 
                 verbose=True):

    param_combinations = [[l_,la_,a_,op_,e_]    for l_ in loss
                                                for la_ in layers
                                                for a_ in activations
                                                for op_ in adjusted_optimizers
                                                for e_ in epochs]
        
    X_ = X[0:int(.75*len(X))]
    Y_ = Y[0:int(.75*len(Y))]
    x = X[int(.75*len(X)):]
    y = Y[int(.75*len(Y)):]
    print("Number of Grid Search Components:", len(param_combinations))
    #add layers and appropriate activations
    counter=0
    for p in param_combinations:
        #increase counter
        counter-=-1
        if verbose:
            print(50*'-')
        print("MODELS TRAINING - ", counter, "/", len(param_combinations))
        if verbose:
            print(50*'-')

        #print(p)
        model = Sequential()
        firstpass = True
        if len(p[1]) != len(p[2]):
            continue
        for i,j in zip(p[1], p[2]):
            #add input layer info the the first layer added to model
            if firstpass:
                model.add(Dense(i, activation=j,input_shape=input_shape_))
                firstpass=False
            #print(i,j)
            else:
                model.add(Dense(i, activation=j))
        
        #output layer
        #model.add(Dense(1, activation='softmax'))
        firstpass=True
        model.compile(loss=p[0],
                      optimizer=p[3],
                      metrics=eval_metrics)
        model.fit(X_,Y_, epochs=p[4], verbose=verbose)
        #print model summaries
        if verbose:
            model.summary()
        y_predicted=model.predict(x, verbose=verbose)
        
        #scoring
        if prediction_metric == 'auc':
            score = roc_auc_score(y, y_predicted)
            print("MODEL", counter, "AUC SCORE = ", score)
        elif prediction_metric == 'accuracy':
            score = accuracy_score(y, y_predicted)
            print("MODEL", counter, "ACCURACY SCORE = ", score)
        elif prediction_metric == 'f1':    
            score = f1_score(y, y_predicted)
            print("MODEL", counter, "F1 SCORE = ", score)
        elif prediction_metric == 'recall':    
            score = recall_score(y, y_predicted)
            print("MODEL", counter, "RECALL SCORE = ", score)
        elif prediction_metric == 'mae':    
            score = mean_absolute_error(y, y_predicted)
            print("MODEL", counter, "MAE SCORE = ", score)
        elif prediction_metric == 'mse':    
            score = mean_squared_error(y, y_predicted)
            print("MODEL", counter, "MSE SCORE = ", score)
        elif prediction_metric == 'msle':    
            score = mean_squared_log_error(y, y_predicted)
            print("MODEL", counter, "MSLE SCORE = ", score)
        elif prediction_metric == 'max':    
            score = max_error(y, y_predicted)
            print("MODEL", counter, "MAX-ERROR SCORE = ", score)
        elif prediction_metric == 'r2':    
            score = r2_score(y, y_predicted)
            print("MODEL", counter, "R2 SCORE = ", score)
        else:
            print("Non existant metric selected, exiting...")
            return
        #save model
        name=str(uuid4())
        json_data=model.to_json()
        with open(str(round(score,2))+name+".json", "w") as json_file:
            json_file.write(json_data)
        model.save_weights(str(round(score,2))+name+".h5")
    

#definiranje parametara modela
losses=['mean_squared_error']
layers_list = [(4,5,5,4,1), 
               (2,2,1)]
activations_list = [('sigmoid','relu','relu','sigmoid', 'sigmoid'), 
                    ('sigmoid', 'softmax', 'sigmoid')]
epochs = [5,10,20]
evaluation_metrics = ['accuracy']
#optimizer
#definiranje parametara za solvere
#potrebno jeza svaki solver definirati odvojeno parametre
#složiti ih u dictionaryje - za keyeve pogledati u funkciju optimizer_get()
types = ["sgd"]
lr = [0.1, 0.01]
momentums = [0.0, 0.02]
nesterov = [True, False]


optimizer_params_sgd = [{"types": t, 
                         "lr": lr_, 
                         "momentum": momentums_, 
                         "nesterov": nesterov_} for t in types
                                                for lr_ in lr
                                                for momentums_ in momentums
                                                for nesterov_ in nesterov]

types = ["adam"]
lr = [0.2, 0.02]
beta1 = [0.5, 0.02]
beta2 = [0.9, 0.95]
amsgrad = [True, False]

optimizer_params_adam = [{"types": t, 
                          "lr": lr_, 
                          "beta1": beta1_, 
                          "beta2": beta2_, 
                          "amsgrad": amsgrad_}  for t in types
                                                for lr_ in lr
                                                for beta1_ in beta1
                                                for beta2_ in beta2
                                                for amsgrad_ in amsgrad]

optimizer_params=optimizer_params_sgd+optimizer_params_adam
print("Number of optimizers to be created: ",len(optimizer_params))


optimizer_list = get_optimizers(optimizer_params)


#Učitavanje podataka
from numpy import loadtxt
# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
input_shape=(8,)
create_model(X,
             y,
             input_shape,
             losses, 
             layers_list, 
             activations_list, 
             optimizer_list, 
             epochs, 
             evaluation_metrics, 
             'auc', 
             False)
