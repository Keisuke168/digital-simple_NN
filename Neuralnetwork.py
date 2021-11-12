import numpy as np
import math
import copy
from read_binary import read_binaryfloat
import matplotlib.pyplot as plt

#シグモイド関数
def sigmoid(x):
    return 1/(1+math.exp(-x))

#二乗和誤差
def sum_squared_error(output,answer):
    sum = 0.0
    for i,j in zip(output,answer):
        sum += (i-j)**2
    return sum

#単一のニューロンオブジェクト
class Neuron:
    def __init__(self,pre_layer_dim,activation_function):
        self.pre_layer_dim = pre_layer_dim
        self.activation_function = activation_function
        self.weights = np.random.randn(self.pre_layer_dim)
    
    def get_outputs(self,x):
        self.input = x
        self.u = np.dot(x,self.weights)
        self.g = self.activation_function(self.u)
        return self.g
    
    def update_weigtht(self,diff):
        self.weights -= diff

#ニューラルネットクラス
class NeuralNetwork:
    def __init__(self,input_dim):
        self.input_dim = input_dim
        self.network = []
        self.neuron_num = []
    
    #損失の取り方の設定
    def set_loss_func(self,loss_func):
        self.loss_func = loss_func

    #次元・活性化関数を指定してadd
    def add_layer(self,dimentions,activation_function):
        if len(self.network)==0:
            self.network.append([Neuron(self.input_dim,activation_function) for i in range(dimentions)])
            self.neuron_num.append(dimentions)
        else:
            self.network.append([Neuron(self.neuron_num[-1],activation_function) for i in range(dimentions)])
            self.neuron_num.append(dimentions)
    
    #ネットワークを表示
    def show_networks(self):
        print('input dimention = ',self.input_dim)
        for l in self.neuron_num:
            print(l)

    #todo 活性化関数指定してるのに微分はシグモイド固定なのどうにかする
    def del_sigmoid(self,x):
        return sigmoid(x)*(1-sigmoid(x))
    
    #学習用関数
    def fit(self,x_data,y_data,epoch,lr=0.1):
        N = len(x_data)
        log = []
        for cnt in range(epoch):
            loss=0
            for x,y in zip(x_data,y_data):
                loss += self.loss_func(self.get_result(x),y)
            print(cnt,loss)
            log.append(loss)
            #Jから各重みへの勾配を保存する行列
            error = []
            for l in range(len(self.network)):
                error.append(np.zeros((self.neuron_num[l],self.network[l][0].pre_layer_dim)))
            
            #バッチ数まとめて計算してから更新
            for x,y in zip(x_data,y_data):
                delta =[]
                for i in range(len(self.network)):
                    delta.append(np.zeros((self.neuron_num[i])))
                #出力層のみ出力の誤差を用いる
                for k in range(self.neuron_num[-1]):
                    delta[-1][k] += -2*(y[k]-self.get_result(x)[k])*self.del_sigmoid(self.network[-1][k].u)
                    for w in range(len(error[-1][k])):
                        error[-1][k][w] += lr/N*delta[-1][k]*self.network[-1][k].input[w]
                #以下一個上の層で計算したdeltaを使う
                for i in range(len(self.network)-1):
                    forcus_layer = -(i+2)
                    for h in range(self.neuron_num[forcus_layer]):
                        #結合してる層から誤差を逆伝播
                        for k in range(self.neuron_num[forcus_layer+1]):
                            delta[forcus_layer][h] += -2*delta[forcus_layer+1][k]
                        delta[forcus_layer][h] *= self.del_sigmoid(self.network[forcus_layer][h].u)
                        for w in range(len(error[-1][h])):
                            error[-1][h][w] += lr/N*delta[-1][h]*self.network[-1][h].input[w]
            #W更新
            for l in range(len(self.network)):
                for n in range(len(self.network[l])):
                    self.network[l][n].update_weigtht(error[l][n])
        return log

    #出力を計算
    def get_result(self,input):
        for layer in self.network:
            output =[n.get_outputs(input) for n in layer]
            input = copy.copy(input)
        return output


if  __name__ == '__main__':

    class1 = read_binaryfloat("class1.dat")
    class1=class1.reshape(150,2)
    class2 = read_binaryfloat("class2.dat")
    class2=class2.reshape(150,2)
    x = np.concatenate([class1,class2])
    y = [[1,0] for i in range(150)]
    y += [[0,1] for i in range(150)]

    nn  = NeuralNetwork(2)
    nn.set_loss_func(sum_squared_error)
    #中間層
    nn.add_layer(2,sigmoid)
    #出力層
    nn.add_layer(2,sigmoid)
    nn.show_networks()

    #学習
    log = nn.fit(x,y,500)

    #損失のプロット
    plt.plot(log)
    plt.xlabel("number of iterations")
    plt.ylabel("total error")
    plt.show()
    
    result = [nn.get_result(i) for i in x]
    
    plt.plot([r[0] for r in result],label="out_node1")
    plt.plot([r[1] for r in result],label="out_node2")
    plt.xlabel("number of samples")
    plt.legend()
    plt.show()

    #テストデータ
    test = read_binaryfloat("test.dat")
    test = test.reshape((int(len(test)/2),2))
    result = [nn.get_result(i) for i in test]
    for i,y in zip(test,result):
        if(y[0]>y[1]):
            c='red'
        else:
            c='black'
        plt.scatter(i[0],i[1],marker='o',color = c)
    
    plt.scatter(class1[:,0],class1[:,1],marker='+',color ='orange',label='class1')
    plt.scatter(class2[:,0],class2[:,1],marker='*',color ='blue',label='class2')
    plt.legend()
    plt.show()

