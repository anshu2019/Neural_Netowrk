# Import the required librray
import numpy as np
import sys
import re
import csv
import math
eps = np.finfo(float).eps
from numpy import log2 as log
from math import exp
from csv import reader
import time

# time start........................................................................
start = time.time()

#commandline param
path1=sys.argv[1]
path2=sys.argv[2]
path3 =sys.argv[3]
path4=sys.argv[4]
path5=sys.argv[5]
no_epoch=int(sys.argv[6])
hidden_unit =int(sys.argv[7])
init_flg =int(sys.argv[8])
l_rate=float(sys.argv[9])

no_fatures = 128
no_class = 10

# Load a file
def load_dataFile(filepath):
    dataset = list()
    with open(filepath, 'r') as file:
        csv_reader = reader(file, delimiter=',')
        for row in csv_reader:
            if not row:
                   continue
            dataset.append(row)
    return dataset

# Make string data from files as float
def str_to_float(lis):
    return [[float(j.strip()) for j in i] for i in lis]

def set_weights(M,N,flag):
    weights = []    
    if flag == 1:
        for i in range(0,M):      
            w =np.random.uniform(-0.1,0.1,N)  
            w[0] = 0
            weights.append(w)
    elif flag==2:
         for i in range(0,M):  
            w =np.zeros(N)
            weights.append(w)
    return weights

#get labels and input
def prepare_data(data):
    A = np.array(data)
    X = A[:, 1:]
    Y = A[:, 0]
    return X, Y

# sigmoid function
def sigmoid_foward(input_layer):
    return 1.0/(1.0+exp(-input_layer))
    
def linear_forward(m,n):
    m = np.array(m)
    a=[]
    for i in range(len(n)):
        x = np.dot(n[i] ,m.T)
        a.append(x)
    return a
    
#Softmax activation for output layers
def softmax_forward(hiden_layer):
    
    exp_layer = np.exp(hiden_layer)
    softmax_layer = exp_layer/np.sum(exp_layer)
    return softmax_layer
    
def crossentropy_forward(Y, Y_hat):        
    j1 = np.log(Y_hat)
    j2 = np.dot(Y.T,j1)
    j3 = np.multiply(-1,j2)
    return j3
    
    
#Feedforward
def NN_Forward(X,Y,alpha, beeta):
    X = [np.append(1,X)] # fold bias term for 0th element
    
    beeta = np.array(beeta)
    alpha = np.array(alpha)
    m = int(Y)     
    Y = np.zeros(10)
    Y[m] =1
    Y=np.reshape(Y, (10,1)) 
    
    
    z=[]    
    a= linear_forward(X,alpha) 
    for w in a:        
        i_sig = sigmoid_foward(w)
        z.append(i_sig)
       
    z = [np.append(1,np.array(z))] # fold bias term for 0th element
    
    b = linear_forward(z,beeta)
    Y_hat = softmax_forward(b) 
    J = crossentropy_forward(Y,Y_hat)
    return a,z,b,Y_hat, J

def sigmoid_backward(a,z,gz):    
    z=np.array(z)
   
    minusb = np.subtract(1,z)
    g = np.multiply(gz,z)
    ga = np.multiply(g,minusb)    
    return ga

def softmax_backward(a,b,gb):   
    b = np.matrix(b)    
    diag_b = np.diagflat(b)    
    bi = np.multiply(b,b.T)
    bdot = np.subtract(diag_b,bi)  
    ga = np.dot(gb.T,bdot)
    return ga

def linear_backward(z,beeta, b,gb):
    z = np.array(z)
    gb = np.array(gb)
    beeta = np.array(beeta)

    gw = np.dot(gb.T,z)
    ga = np.dot(beeta.T,gb.T)
    gw=np.array(gw)
    ga=np.array(ga)

    return gw,ga

def crossentropy_backward(y, y_hat, b, gb):
    gbminusb = -1 *gb
    yht_div =np.divide(y,y_hat)
    gahat =np.multiply(gbminusb,yht_div)
    return gahat
    
#Backward propagation
def NN_Backward(X,Y,alpha,beeta,a,z,b,Y_hat, J):
    m = int(Y)    
    Y = np.zeros(10)
    Y[m] =1
    Y=np.reshape(Y, (10,1))
      
    gj = 1    
    gyhat =crossentropy_backward(Y,Y_hat,J,gj)
    gb = softmax_backward(b,Y_hat,gyhat)
    gbeeta,gz = linear_backward(z,beeta,b,gb)
    
    zmod = np.delete(z, 0) 
    gzmod = np.delete(gz, 0)
    ga = sigmoid_backward(a,zmod,gzmod)    
    ga=[np.array(ga)]
    X=np.append(1,X)
    X= [np.array(X)]
    galpha,gx = linear_backward(X,alpha,a,ga)   
    return galpha,gbeeta

#update weight()
def update_weight(alpha,beeta,g_alpha,g_beeta):
    aa = np.multiply(l_rate,g_alpha)
    bb = np.multiply(l_rate,g_beeta)
    alpha = np.array(alpha) - np.array(aa)
    beeta = np.array(beeta) - np.array(bb)
    return alpha, beeta

def get_mean_crossentropy(X,Y,alpha,beeta):
    J=0  
    length = len(X)
    for i in range(length):
        a,z,b,Y_hat, j = NN_Forward(X[i],Y[i], alpha,beeta)
        J=J+j
    mean_crossentropy = float(J)/length
    return mean_crossentropy      
            
    
# apply stochastic gradient descent to get theeta value
def get_theeta_by_sgd(train_x,train_y,alpha,beeta,test_x,text_y):
    length= len(train_x)
    for epoch in range(no_epoch):
        J=0
        Jt=0
        for i in range(length):
            a,z,b,Y_hat, j = NN_Forward(train_x[i],train_y[i], alpha,beeta)
            J=J+j
            g_alpha,g_beeta = NN_Backward(train_x[i],train_y[i],alpha,beeta,a,z,b,Y_hat, j)
            alpha, beeta = update_weight(alpha, beeta,g_alpha,g_beeta)
            
        #Calculate Cross entropy for each epoch   
        Jtr =get_mean_crossentropy(train_x,train_y,alpha,beeta)
        errMetricFile.writelines("epoch="+str(epoch+1)+" crossentropy(train): "+str(Jtr)+"\n")
        Jts =get_mean_crossentropy(test_x,text_y,alpha,beeta)
        errMetricFile.writelines("epoch="+str(epoch+1)+" crossentropy(test): "+str(Jts)+"\n")
        print(Jtr)
        print(Jts)
    return alpha, beeta

def evaluate_data(X,Y,alpha,beeta,file):
    prediction = list()
    error=0    
    for i in range(len(X)):
            a,z,b,Y_hat, j = NN_Forward(X[i],Y[i], alpha,beeta)            
            predict =np.argmax(Y_hat)
            if(Y[i] !=predict):
                error = error +1                
            prediction.append(predict)
            file.writelines(str(predict)+'\n')
    err = float(error)/len(X)
    print(str(err))
    return prediction,str(err)

# load data
errMetricFile = open(path5,"w")
trnOutLbl = open(path3,"w")
tesOutLbl = open(path4,"w")

train_data = load_dataFile(path1)
train_data =str_to_float(train_data)
X_train, Y_train = prepare_data(train_data)
no_fatures = len(X_train[0])

test_data = load_dataFile(path2)
test_data =str_to_float(test_data)
X_test, Y_test = prepare_data(test_data)

#initialize weights , either random or 0 based on flag ...
alpha= set_weights(hidden_unit,no_fatures + 1,init_flg)
beeta= set_weights(no_class,hidden_unit + 1,init_flg) 

n_alpha, n_beeta =get_theeta_by_sgd(X_train,Y_train,alpha,beeta,X_test,Y_test)


result_train, errtrn = evaluate_data(X_train,Y_train,n_alpha,n_beeta,trnOutLbl)
errMetricFile.writelines("error(train): "+str(errtrn)+"\n")

result_test, errtes = evaluate_data(X_test,Y_test,n_alpha,n_beeta,tesOutLbl)
errMetricFile.writelines("error(test): "+str(errtes)+"\n")

errMetricFile.close()
trnOutLbl.close()
tesOutLbl.close()
# code ends here.................................................................
end = time.time()
elapsed = end - start
print("done in - %f" %(elapsed))
