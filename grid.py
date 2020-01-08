# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Input
import keras
from uuid import uuid4
from keras.models import model_from_json
#supress warnings because they serve no purpose
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

"""
param_dict - dictionary sa parametrime

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
def optimizer_get(param_dict):
    optimizer_type = param_dict["types"]
    optimizer_list= ['sgd', 'adam', 'rmsprop', 'adagrad','adadelta', 'adamax','nadam']
    
    if optimizer_type=='sgd':
        optimizer=keras.optimizers.SGD(lr=param_dict["lr"], 
                                      momentum=param_dict["momentum"],
                                      nesterov=param_dict["nesterov"])
        return optimizer
    elif optimizer_type=='rmsprop':
        optimizer=keras.optimizers.RMSprop(lr=param_dict["lr"], rho=param_dict["rho"])
        return optimizer
    elif optimizer_type=='adagrad':
        optimizer=keras.optimizers.Adagrad(lr=param_dict["lr"])
        return optimizer
    elif optimizer_type=='adadelta':
        optimizer=keras.optimizers.Adadelta(lr=param_dict["lr"], rho=param_dict["rho"])
        return optimizer
    elif optimizer_type=='adam':
        optimizer=keras.optimizers.Adam(lr=param_dict["lr"], beta_1=param_dict["beta1"],beta_2=param_dict["beta2"],amsgrad=param_dict["amsgrad"])
        return optimizer
    elif optimizer_type=='adamax':
        optimizer=keras.optimizers.Adamax(lr=param_dict["lr"], beta_1=param_dict["beta1"], beta_2=param_dict["beta2"])
        return optimizer
    elif optimizer_type=='nadam':
        optimizer=keras.optimizers.Nadam(lr=param_dict["lr"], beta_1=param_dict["beta1"], beta_2=param_dict["beta2"])
        return optimizer
    else:
        print("Chosen optimizer unsupported, choose between: ", optimizer_list)
        return None
    
        
"""
stvara i trenira modele Koristeći parametre. Trenirani modeli spremljeni u JSON sa težinama u H5.

Argumnti

X - Ulazni Podaci (Array)
Y - Izlazni Podaci (Array)
input_shape - oblik ulaza (tuple of ints)
loss - lista korištenih lossova (list of strings)
layers - lista tupleova koji sadrže različite konfiguracije mreže - (list of tuple of Ints)
        svaki tuple je niz brojeva koji odgovaraju broju neurona u svakom skrivenom sloju
activations - lista tupleova iste duljine kao i gornjih layera, koji sadrže aktivacije za svaki od gore navedenih slojeva - (list of tuple of Strings)
adjusted_optimizers - lista optimizera dobivenih korištenjem funkcije optimizer_get() - (list of keras.Optimizers)
epochs - lista sa brojvima epoha - (list of ints)
verbose - ispis model summarya i training informationa - (bool)

"""
def create_model(X, Y, input_shape_, loss, layers, activations, adjusted_optimizers, epochs, verbose=True):

    param_combinations = [[l_,la_,a_,op_,e_] for l_ in loss
                                                  for la_ in layers
                                                  for a_ in activations
                                                  for op_ in adjusted_optimizers
                                                  for e_ in epochs]
        
    
    print("Number of Grid Search Components:", len(param_combinations))
    #add layers and appropriate activations
    counter=0
    for p in param_combinations:
        print(round(counter/len(param_combinations)*100,2),"% DONE")
        #increase counter
        counter-=-1
        #print(p)
        model = Sequential()
        firstpass = True
        if len(p[1]) != len(p[2]):
            continue
        for i,j in zip(p[1], p[2]):
            #add input layer info the the first layer added to model
            if firstpass:
                model.add(Dense(i, activation=j, name=str(i)+str(j)+str(uuid4()),input_shape=input_shape_))
                firstpass=False
            #print(i,j)
            else:
                model.add(Dense(i, activation=j,name=str(i)+str(j)+str(uuid4())))
        
        #output layer
        #model.add(Dense(1, activation='softmax'))
        firstpass=True
        model.compile(loss=p[0],
                      optimizer=p[3],
                      metrics=['accuracy'])
        model.fit(X,Y, epochs=p[4], verbose=verbose)
        #prin model summaries
        if verbose:
            model.summary()
        #save model
        name=str(uuid4())
        json_data=model.to_json()
        with open(name+".json", "w") as json_file:
            json_file.write(json_data)
        model.save_weights(name+".h5")
    

#definiranje parametara modela
losses=['mean_squared_error']
layers_list = [(4,5,5,4,1), (2,2,1)]
activations_list = [('sigmoid','relu','relu','sigmoid', 'sigmoid'), ('sigmoid', 'softmax', 'sigmoid')]
epochs = [5,10,20]

#optimizer
#definiranje parametara za solvere
#potrebno jeza svaki solver definirati odvojeno parametre
#složiti ih u dictionaryje - za keyeve pogledati u funkciju optimizer_get()
types = ["sgd"]
lr = [0.1, 0.01]
momentums = [0.0, 0.02]
nesterov = [True, False]


optimizer_params_sgd = [{"types": t, "lr": lr_, "momentum": momentums_, "nesterov": nesterov_} for t in types
                                                                                            for lr_ in lr
                                                                                            for momentums_ in momentums
                                                                                            for nesterov_ in nesterov]

types = ["adam"]
lr = [0.2, 0.02]
beta1 = [0.5, 0.02]
beta2 = [0.9, 0.95]
amsgrad = [True, False]

optimizer_params_adam = [{"types": t, "lr": lr_, "beta1": beta1_, "beta2": beta2_, "amsgrad": amsgrad_} for t in types
                                                                                                         for lr_ in lr
                                                                                                         for beta1_ in beta1
                                                                                                         for beta2_ in beta2
                                                                                                         for amsgrad_ in amsgrad]
optimizer_params=optimizer_params_sgd+optimizer_params_adam



optimizer_list = []

for op in optimizer_params:
   # print(op)
    optimizer_list.append(optimizer_get(op))

#Učitavanje podataka
from numpy import loadtxt
# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
input_shape=(8,)
create_model(X,y,input_shape,losses, layers_list, activations_list, optimizer_list, epochs, False)
