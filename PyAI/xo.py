import random

#x=1,o=0
X=[
    [1,0,0,0,0,0,1],
    [0,1,0,0,0,1,0],
    [0,0,1,0,1,0,0],
    [0,0,0,1,0,0,0],
    [0,0,1,0,1,0,0],
    [0,1,0,0,0,1,0],
    [1,0,0,0,0,0,1]]
T=0

data=[
    [
        [
            [0,0,1,1,0,0,0],
            [0,1,0,0,1,1,0],
            [0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0],
            [1,0,0,0,0,1,0],
            [0,1,1,0,1,0,0],
            [0,0,0,1,1,0,0]
            ],0],
    [
        [
            [1,0,0,0,0,0,1],
            [0,1,0,0,0,1,0],
            [0,0,1,0,1,0,0],
            [0,0,0,1,0,0,0],
            [0,0,1,0,1,0,0],
            [0,1,0,0,0,1,0],
            [1,0,0,0,0,0,1]
            ],1],
    [
        [
            [0,0,0,0,0,0,1],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,1,0],
            [0,0,0,1,1,0,0],
            [0,0,1,0,0,1,0],
            [0,1,0,0,0,1,0],
            [1,0,0,0,0,0,1]
            ],1],
    [
        [
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,1,0,0],
            [0,0,1,1,0,0,0],
            [0,0,1,0,1,0,0],
            [0,1,0,0,1,0,0],
            [1,0,0,0,0,1,0]
            ],1],
    [
        [
            [0,0,0,0,0,1,0],
            [0,1,0,0,1,0,0],
            [0,0,1,0,1,0,0],
            [0,0,0,1,0,0,0],
            [0,0,1,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,1,0,0,0,0,0]
            ],1],
    [
        [
            [0,0,1,1,1,0,0],
            [0,1,0,0,0,1,0],
            [1,0,0,0,0,0,1],
            [1,0,0,0,0,0,1],
            [1,0,0,0,0,0,1],
            [0,1,0,0,0,1,0],
            [0,0,1,1,1,0,0]
            ],0],
    [
        [
            [0,0,0,1,1,0,0],
            [0,1,1,0,0,1,0],
            [1,0,0,0,0,0,1],
            [1,0,0,0,0,0,1],
            [0,1,0,0,1,1,1],
            [0,0,1,1,0,0,0],
            [0,0,0,0,0,0,0]
            ],0],
    [
        [
            [0,0,0,0,0,0,0],
            [0,0,0,1,1,1,0],
            [0,1,1,0,0,0,1],
            [0,1,0,0,0,0,1],
            [1,0,0,0,0,0,1],
            [0,1,0,0,0,1,0],
            [0,0,1,1,1,0,0]
            ],0],
    [
        [
            [0,0,1,1,0,0,0],
            [0,1,0,0,1,0,0],
            [0,1,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,1,0,0,1,0,0],
            [0,0,1,1,0,0,0],
            [0,0,0,0,0,0,0]
            ],0]
    ]


random.shuffle(data)



def show():#show the data
    global data
    for i in data:
        for x in i[0]:
            for y in x:
                if y==0:
                    print(end=' ')
                else:
                    print(end='@')
            print()
        print(end='\n\n\n')
#show the data
#show()


import numpy as np

def sigmoid(x):
    # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))
 
def deriv_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)
def mse_loss(y_true, y_pred):
    # y_true和y_pred是相同长度的numpy数组。
    return ((y_true - y_pred) ** 2).mean()



class Neuron():

    def __init__(self,r=0.1):

        """self.w = []
        v=[]
        for i in range(3):
            for i in range(3):
                v.append(np.random.normal())
            self.w.append(v)
            v=None
            v=[]"""
        
        self.w = np.random.normal()
        #print(type(self.w))
        self.b = np.random.normal()
        
        self.r=r

    def feedforward(self, inputs):
        # weight inputs, add bias, then use the activation function
        total = self.b
        """for i in range(len(inputs)):
            for x in range(len(i)):
                w=self.w[i][x]
                total+=w*inputs[i][x]"""
        total+= self.w*inputs
                

        return sigmoid(total)
    def change(self,inputs,miss,omiss):
        total=self.b
        """for i in range(len(inputs)):
            for x in range(len(i)):
                w=self.w[i][x]
                total+=w*inputs[i][x]"""
        #print(type(self.w),type(inputs))
        total+= self.w*inputs
        num=total
        
        #ws = []
        #v=[]
        """for i in range(len(inputs)):
            for x in range(len(i)):
                #v.append(inputs[i][x]*deriv_sigmoid(num))
                self.w[i][x]-=self.r*miss*\
                               (inputs[i][x]*deriv_sigmoid(num))*omiss
            #ws.append(v)"""
        self.w-=self.r*miss*(inputs*deriv_sigmoid(num))*omiss
                
        self.b -= self.r*miss*deriv_sigmoid(num)*omiss

class Nout(Neuron):
    def __init__(self,r):
        super().__init__(r)
        self.w = []
        v=[]
        for i in range(3):
            for i in range(3):
                v.append(np.random.normal())
            self.w.append(v)
            v=None
            v=[]
    def feedforward(self, inputs):
        # weight inputs, add bias, then use the activation function
        total = self.b
        for i in range(len(inputs)):
            for x in range(len(inputs[i])):
                w=self.w[i][x]
                total+=w*inputs[i][x]
        #total+= self.w*inputs
                

        return sigmoid(total)
    def rtCh(self,inputs):
        total=self.b
        for i in range(len(inputs)):
            for x in range(len(inputs[i])):
                w=self.w[i][x]
                total+=w*inputs[i][x]
        num=total
        
        ws=[]
        v=[]
        for i in range(len(inputs)):
            for x in range(len(inputs[i])):
                v.append(self.w[i][x]*deriv_sigmoid(num))
            ws.append(v)
            v=None
            v=[]
        return ws
    def change(self,inputs,miss):
        total=self.b
        for i in range(len(inputs)):
            for x in range(len(inputs[i])):
                w=self.w[i][x]
                total+=w*inputs[i][x]
        num=total
        
        #ws = []
        #v=[]
        for i in range(len(inputs)):
            for x in range(len(inputs[i])):
                #v.append(inputs[i][x]*deriv_sigmoid(num))
                self.w[i][x]-=self.r*miss*\
                               (inputs[i][x]*deriv_sigmoid(num))
            #ws.append(v)
                
        self.b -= self.r*miss*deriv_sigmoid(num)


class Model:
    def __init__(self):
        self.ON=Nout(0.1) # Out Neuron
        self.N=[] # Neuron
        v=[]
        for i in range(3):
            for x in range(3):
                v.append(Neuron())
            self.N.append(v)
            v=None
            v=[]
    def feedforward(self, inputs):
        inputs=Conv(inputs)
        h=[]
        v=[]
        #print(inputs)
        for i in range(len(inputs)):
            for x in range(len(inputs[i])):
                say=self.N[i][x].feedforward(inputs[i][x])
                v.append(say)
            h.append(v)
            v=None
            v=[]
        return self.ON.feedforward(h)
    def train(self,count):
        global data,T
        for i in range(count):
            for x in data:
                #for z in range(len(x)):
                #print('f',x,z)
                #print(x[z])
                T=x[1]
                #print(T)
                #print(x[z][0],x[z])
                #print('f')
                #print(x[z])
                conv=Conv(x[0])
                #print('d')
                gt=self.feedforward(x[0])
                #print(x)
                h=[]
                v=[]
                for ii in range(len(conv)):
                    for xx in range(len(conv[ii])):
                        v.append(self.N[ii][xx].feedforward(conv[ii][xx]))
                    h.append(v)
                    v=None
                    v=[]
                
                miss=-2 * (T - gt)
                in_miss=self.ON.rtCh(h)
                for ii in range(len(conv)):
                    for xx in range(len(conv[ii])):
                        self.N[ii][xx].change(conv[ii][xx],miss,in_miss[ii][xx])
                #print(x)
                self.ON.change(h,miss)
                #print(x)
                #print('d')
                #print(T,gt)
                #print(i)
                if i % 10 == 0:
                    print(i,abs(T-gt))

                        
                
                

def Conv(W):
    global X
    rt=[]
    v=[]
    q=0
    for i in range(0,4,2):#0, len(X)-3, 2
        for o in range(0,4,2):#0, len(X[i])-3, 2
            for y in range(3):
                for x in range(3):
                    #print(W)
                    XX=X[i+y][o+x]
                    #print(W)
                    #print(W)
                    WWW=W[i+y]
                    #print(o,x,WWW)
                    WW=WWW[o+x]
                    q+=XX*WW
            v.append(q)
        rt.append(v)
        v=None
        v=[]
        #print('f',rt)
    return rt
        

    
                

"""
weights = np.array([0, 1]) # w1 = 0, w2 = 1

bias = 4

n = Neuron(weights, bias)

# inputs

x = np.array([2, 3])   # x1 = 2, x2 = 3

print(n.feedforward(x)) # 0.9990889488055994

"""

#practice the model

mode=Model()
mode.train(1000)






#test the model
input("Pause enter to continue...")
"""test = [
    [0,1,0,0,0,0,1],
    [0,0,1,0,0,0,1],
    [0,0,1,0,0,1,0],
    [0,0,0,1,0,1,0],
    [0,0,0,0,1,0,0],
    [0,0,0,1,0,1,0],
    [0,0,1,0,0,0,1]
    ]
"""
test = [
    [0,0,1,1,0,0,0],
    [0,1,0,0,1,1,0],
    [0,1,0,0,0,0,1],
    [1,0,0,0,0,1,0],
    [1,0,0,0,0,1,0],
    [0,1,1,0,1,0,0],
    [0,0,0,1,1,0,0]
    ]

for i in test:
    for x in i:
        if x == 0:
            print(end=' ')
        else:
            print(end='@')
    print()
print('\n'*3)

print(mode.feedforward(test))

input("Pause enter to continue...")
