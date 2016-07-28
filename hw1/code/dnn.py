'''
Description: Deep Neural Network
'''

import os
import sys
import time
import copy
import theano
import numpy as np
import cost
import activate
import learningRate

class DNN():
    def __init__(self,struct=None,actiFunc=None,costFunc=None,learningRateFunc=None):
        '''
        Description: initialize dnn model
        Parameters:
            struct: dnn structure described in string; separated with '-'; no default value; example:'39-128-100'
            actiFunc: activation function defined in activation.py described in string; default is 'ReLU';  example:'ReLU'
            costFunc: cost function defined in cost.py described in string; default is 'meanSquare'; example:'meanSquare'
            learningRateFunc: a float number or learning rate defined in learningRate.py; default is 0.01; example:0.01,'adagrad'
        Component:
            self.struct: string
        Example:
            d=dnn.DNN(struct='39-128-40',actiFunc='ReLU',costFunc='meanSquare',learningRateFunc=0.01)
        '''

        self.set_matrixDot()
        self.set_matrixAdd()
        
        self.set_defaultParam()
        self.struct=struct
        if actiFunc!=None:
            self.actiFunc=actiFunc
        if costFunc!=None:
            self.costFunc=costFunc
        if learningRateFunc!=None:
            self.learningRateFunc=learningRateFunc

        self.init_net()
        self.set_actiFunc()
        self.set_costFunc()
        self.set_learningRateFunc()
        return

    def set_matrixDot(self):
        '''
        Description: define theano matrix dot to speed up with gpu
        Component:
            self.dot=theano.function
        '''

        X=theano.tensor.matrix(dtype='float32')
        Y=theano.tensor.matrix(dtype='float32')
        Z=theano.tensor.dot(X,Y)
        self.dot=theano.function([X,Y],Z)
        return

    def set_matrixAdd(self):
        '''
        Description: define theano matrix addition to speed up with gpu
        Component:
            self.add=theano.function
        '''

        X=theano.tensor.matrix(dtype='float32')
        Y=theano.tensor.matrix(dtype='float32')
        Z=X+Y
        self.add=theano.function([X,Y],Z)
        return

    def set_defaultParam(self):
        '''
        Description: set default parameters
        Component:
            self.actiFunc: string
            self.costFunc: string
            self.learningRate: string or float
        '''

        self.actiFunc='ReLU'
        self.costFunc='meanSquare'
        self.learningRateFunc=0.01

        return

    def set_actiFunc(self):
        '''
        Description: set activation function and its differential
        Component:
            self.activate: function defined in activate.py
            self.activate_diff: function defined in activate.py
        '''

        if self.actiFunc=='ReLU':
            self.activate=activate.ReLU
            self.activate_diff=activate.ReLU_diff
        return

    def set_costFunc(self):
        '''
        Description: set cost function and its differential
        Component:
            self.cost: function defined in cost.py
            self.cost_diff: function defined in cost.py
        '''

        if self.costFunc=='meanSquare':
            self.cost=cost.meanSquare
            self.cost_diff=cost.meanSquare_diff
        return

    def set_learningRateFunc(self):
        '''
        Description: set learning rate
        Component:
            self.learningRate: function defined in learningRate.py
        '''

        if self.learningRateFunc=='adagrad':
            self.learningRate=learningRate.adagrad
        else:
            try:
                float(self.learningRateFunc)
                self.learningRate=learningRate.constant
            except ValueError:
                print('Undefined learning rate %s' % (self.learningRateFunc))
                sys.exit(0)
        return

    def init_net(self):
        '''
        Description: initialize neural nets
        Component:
            self.nets: an list of m by n matrice where m is the number of neuraons in the second layer and n is the number of neurons in the first layer plus one
            self.netNum: number of nets; integer
            self.layerNum: number of layers; integer
        '''

        self.nets,self.beforeActi,self.afterActi,self.weightGrad=[],[],[],[]
        layer=self.struct.split('-')
        
        i=0
        while i < len(layer)-1:
            self.nets.append(np.random.random((int(layer[i+1]),int(layer[i])+1)).astype(dtype='float32'))
            i += 1
        self.netNum,self.layerNum=len(self.nets),len(self.nets)+1
        return

    def forward(self,X):
        '''
        Description: forward
        Parameter:
            X: features of training examples; m by n matrix where m is the number of features and n is the number training examples
        Return:
            None
        Component:
            self.beforeActi: results before activation function; a list of m by n matrice where m is the number of neurons in that layer and n is the number of training examples
            self.afterActi: results after activation function; format same as above
        Example:
            d.forward(X)
        '''

        r=np.matrix(X).astype(dtype='float32')
        self.beforeActi[:]=[]
        self.afterActi[:]=[]
        nets=copy.deepcopy(self.nets)

        self.beforeActi.append(copy.deepcopy(r))
        self.afterActi.append(copy.deepcopy(r))
        for net in nets:
            x=np.concatenate((r,np.ones((1,r.shape[1])).astype(dtype='float32')),axis=0)
            r=self.dot(net,x)
            self.beforeActi.append(copy.deepcopy(r))
            r=self.activate(r)
            self.afterActi.append(copy.deepcopy(r))
        raw=self.afterActi.pop()
        res=activate.softMax(raw)
        self.afterActi.append(copy.deepcopy(res))
        return

    def calculate_error(self,label):
        '''
        Description: calculate error
        Parameter:
            label: labels of training examples; m by n matrix where m is the number of label categories and n is the number of training examples
        Return:
            cost; float
        Example:
            d.calculate_error(label)
        '''

        r,label,nets=copy.deepcopy(self.afterActi[-1]),np.matrix(label).astype(dtype='float32'),copy.deepcopy(self.nets)
        return self.cost(r,label,nets)

    def backpropagation(self,label):
        '''
        Description: backpropagation
        Parameter:
            label: labels training examples; m by n matrix where m is the number of label categories and n is the number of training examples
        Return:
            None
        Component:
            self.weightGrad: gradients to be updated; format same as self.nets but in reverse order
        Example:
            d.backpropagation(label)
        '''

        batchSize=self.beforeActi[0].shape[1]
        self.weightGrad[:]=[]
        label,nets=np.matrix(label).astype(dtype='float32'),copy.deepcopy(self.nets)
        beforeActi,afterActi=copy.deepcopy(self.beforeActi),copy.deepcopy(self.afterActi)

        delta=np.multiply(self.activate_diff(beforeActi[-1]),self.cost_diff(afterActi[-1],label,nets))
        a,b=self.activate_diff(beforeActi[-1]),self.cost_diff(afterActi[-1],label,nets)#debug
        oneArr=np.ones((1,batchSize)).astype(dtype='float32')
        a=np.concatenate((afterActi[self.layerNum-2],oneArr),axis=0)
        c_partial=self.dot(delta,np.transpose(a))/batchSize
        self.weightGrad.append(copy.deepcopy(c_partial))
        for i in range(1,self.netNum):
            x=self.dot(np.transpose(nets[self.netNum-i]),delta)
            x=np.delete(x,x.shape[0]-1,0)
            delta=np.multiply(self.activate_diff(beforeActi[self.layerNum-1-i]),x)
            oneArr=np.ones((1,batchSize)).astype(dtype='float32')
            a=np.concatenate((afterActi[self.layerNum-2-i],oneArr),axis=0)
            c_partial=self.dot(delta,np.transpose(a))/batchSize
            self.weightGrad.append(copy.deepcopy(c_partial))
        self.weightGrad.reverse()
        return

    def update(self):
        '''
        Description: update gradients
        Return:
            None
        Example:
            d.update()
        '''

        for i in range(len(self.nets)):
            self.nets[i]=self.nets[i]-self.learningRate(self.learningRateFunc)*self.weightGrad[i]
        return

    def predict(self,X):
        '''
        Description: predict
        Parameter:
            X: features of testing examples; m by n matrix where m is the number of features and n is the number of testing examples
        Return:
            predicted category; n dimension array where each element is the predicted category of a testing example
        Example:
            result=d.predict(X)
        '''

        r=np.matrix(X).astype(dtype='float32')
        batchSize=r.shape[1]
        nets=copy.deepcopy(self.nets)

        for net in nets:
            x=np.concatenate((r,np.ones((1,batchSize)).astype(dtype='float32')),axis=0)
            r=self.dot(net,x)
            r=self.activate(r)
        return np.argmax(r,axis=0)

    def score(self,r,label):
        '''
        Description: calculate the accuracy in this batch
        Parameter:
            r: predicted result; n dimension array
            label: label; n dimension array
        Return:
            correct: number of correct test examples; int
            total: total test examples in this batch; int
        Example:
            correct,total=d.score(r,label)
        '''

        correct,total=0,r.size
        for i in range(total):
            if r.item(i)==label.item(i):
                correct += 1
        return (correct,total)

    def save_model(self):
        '''
        Description: save neural nets to file in current directory
        Output:
            model.npz
        Example:
            d.save_model()
        '''

        np.savez('model.npz',*self.nets)
        return

    def load_model(self):
        '''
        Description: load neural net model to self.nets from current directory if it exists
        Example:
            d.load_model()
        '''

        if not os.path.isfile('model.npz'):
            return

        npzfile=np.load('model.npz')
        self.netNum,self.layerNum=len(npzfile.files),len(npzfile.files)+1
        self.nets[:]=[]
        for i in range(self.netNum):
            name='arr_'+str(i)
            self.nets.append(copy.deepcopy(npzfile[name]))
        return
