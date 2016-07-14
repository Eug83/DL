import os
import sys
import theano
import numpy as np
import cost
import activate
import learningRate

class DNN():
    def __init__(self,struct=None,actiFunc=None,costFunc=None,learningRateFunc=None):
        self.set_matrixMult()
        
        self.set_defaultParam()
        if struct!=None:
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

    def set_matrixMult(self):
        X=theano.tensor.matrix(dtype='float32')
        Y=theano.tensor.matrix(dtype='float32')
        Z=theano.tensor.dot(X,Y)
        self.multi=theano.function([X,Y],Z)
        return

    def set_defaultParam(self):
        self.struct='39-128-39'
        self.actiFunc='ReLU'
        self.costFunc='norm1'
        self.learningRateFunc=0.001

        return

    def set_actiFunc(self):
        if self.actiFunc=='ReLU':
            self.activate=activate.ReLU
            self.activate_diff=activate.ReLU_diff
        return

    def set_costFunc(self):
        if self.costFunc=='norm1':
            self.cost=cost.norm1
            self.cost_diff=cost.norm1_diff
        return

    def set_learningRateFunc(self):
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
        self.nets=[]
        layer=self.struct.split('-')
        
        i=0
        while i < len(layer)-1:
            self.nets.append(np.random.randn(int(layer[i+1]),int(layer[i])+1).astype(dtype='float32'))
            i += 1
        self.netNum,self.layerNum=len(self.nets),len(self.nets)+1
        return

    def forward(self,X):
        r=np.matrix(X).astype(dtype='float32')
        self.beforeActi,self.afterActi=[],[]
        self.beforeActi.append(r)
        self.afterActi.append(r)
        for net in self.nets:
            x=np.concatenate((r,np.ones((1,r.shape[1])).astype(dtype='float32')),axis=0)
            r=self.multi(net.astype(dtype='float32'),x)
            self.beforeActi.append(r)
            r=self.activate(r).astype(dtype='float32')
            self.afterActi.append(r)
        return

    def calculate_error(self,label):
        r,label=self.afterActi[len(self.afterActi)-1],np.matrix(label)
        return self.cost(r,label)

    def backpropagation(self,label):
        batchSize=self.beforeActi[0].shape[1]
        self.weightGrad=[]
        delta=np.multiply(self.activate_diff(self.beforeActi[self.layerNum-1]),self.cost_diff(self.afterActi[self.layerNum-1],label))
        oneArr=np.ones((1,batchSize)).astype(dtype='float32')
        a=np.concatenate((self.afterActi[self.layerNum-2],oneArr),axis=0)
        c_partial=np.dot(delta,np.transpose(a))/delta.shape[1]
        self.weightGrad.append(c_partial)
        for i in range(1,self.netNum):
            x=np.dot(np.transpose(self.nets[self.netNum-i]),delta)
            x=np.delete(x,x.shape[0]-1,0)
            delta=np.multiply(self.activate_diff(self.beforeActi[self.layerNum-1-i]),x)
            oneArr=np.ones((1,batchSize)).astype(dtype='float32')
            a=np.concatenate((self.afterActi[self.layerNum-2-i],oneArr),axis=0)
            c_partial=np.dot(delta,np.transpose(a))/delta.shape[1]
            self.weightGrad.append(c_partial)
        return

    def update(self):
        for i in range(len(self.nets)):
            self.nets[i]=self.nets[i]-self.learningRate(self.learningRateFunc)*self.weightGrad[len(self.weightGrad)-1-i]
        return

    def predict(self):
        return

    def save_model(self):
        return

    def load_model(self):
        return
