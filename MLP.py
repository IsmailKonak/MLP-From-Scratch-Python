import numpy as np 
import matplotlib.pyplot as plt
from mpmath import exp
import time
from datetime import datetime

class MLP:
    def __init__(self):
        iin = [[0,0],[1,0],[0,1],[1,1]]
        out = [[0],[1],[1],[0]]
        g_in = []
        g_out = []
        for i in range(20):
            for item1 in iin:
                g_in.append(item1)
            for item2 in out:
                g_out.append(item2)

        self.input_layer = np.matrix(g_in)
        self.hidden_layer = np.matrix(np.zeros([1,4]))
        self.output_layer = np.matrix(np.zeros([1,1]))
        self.real_output_layer = np.matrix(g_out)

        mean = 0.5
        variance = 0.1
        self.weights1 = np.matrix(np.random.normal(mean,variance,size=(4,2)))
        self.weights2 = np.matrix(np.random.normal(mean,variance,size=(1,4)))

        self.i_l_num = self.input_layer[0].shape[1]
        self.h_l_num = self.hidden_layer[0].shape[1]
        self.o_l_num = self.output_layer[0].shape[1]

        self.biases1 = np.matrix(np.zeros([1,4]))
        self.biases2 = np.matrix(np.zeros([1,1]))

        self.loss = []
        self.loss_geç = []
        self.accuracy = []
        self.output_list_1 = []
        self.output_list_0 = []
        self.learning_rate_w = 0.03
        self.learning_rate_b = 0.03

    def linear_comb(self,X,W):
        wT = W.T
        return float(np.dot(X,wT))

    def sigmoid(self,linear_com):
        return float(1/(1+ exp(-1*(linear_com))))

    def d_sigmoid(self,x):
        return float(self.sigmoid(x)*(1-self.sigmoid(x)))

    def loss_f(self,d):
        result = np.sum(np.square((self.real_output_layer[d][0,:]-self.output_layer[0,:])))/self.o_l_num
        return result

    def d_loss_f(self,d,k):
        result = -2*((self.real_output_layer[d][0,k])-(self.output_layer[0,k]))
        return float(result)

    def scatter_graph(self,epoch):
        plt.scatter(0, 0,[150],c="red")
        plt.scatter(1, 1,[150],c="blue")
        alphalist1= []
        alphalist0= []
        
        for i in range(len(self.output_list_1)):
            alphalist1.append((i+1)/(len(self.output_list_1)*10))
        for i in range(len(self.output_list_0)):
            alphalist0.append((i+1)/(len(self.output_list_0)*10))

        plt.scatter(self.output_list_1, self.output_list_1,c="blue",alpha=alphalist1)
        plt.scatter(self.output_list_0, self.output_list_0,c="red",alpha=alphalist0)
        plt.show()
    
    def lossv(self):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.plot(self.loss,color="red",label="Loss")
        ax1.tick_params(axis='y')
        ax2 = ax1.twinx()
        ax2.set_ylabel("Accuracy")
        ax2.plot(self.accuracy,color="green",label="Accuracy")
        ax2.tick_params(axis='y')
        fig.tight_layout()
        fig.legend(loc='upper left')
        plt.title("Loss / Accuracy")
        plt.show()

    def threshold(self,x):
        if x>0.5:
            return 1
        elif x<=0.5:
            return 0


    def backpropagation_bias_2(self,d):
        for i in range(self.o_l_num):
            for j in range(self.h_l_num):
                E_to_a = self.d_loss_f(d,i)
                a_to_z = self.d_sigmoid(self.linear_comb(self.hidden_layer, self.weights2[i,:])+self.biases2[0,i])
                z_to_b = 1
                d_ratio_2 = E_to_a*a_to_z*z_to_b
                self.biases2[0,i] = self.biases2[0,i] - (self.learning_rate_b)*d_ratio_2

    def backpropagation_bias_1(self,d):
        interfers_all_list = []
        for j in range(self.h_l_num):
            for k in range(self.i_l_num):
                for i in range(self.o_l_num):
                    i_all__E_to_a = self.d_loss_f(d, i)
                    i_all__a_to_z = (self.d_sigmoid(self.linear_comb(self.hidden_layer,self.weights2[i,:])+self.biases2[0,i]))
                    i_all__z_to_a_prev = self.weights2[i,j]
                    interfers_all = i_all__E_to_a*i_all__a_to_z*i_all__z_to_a_prev
                    interfers_all_list.append(interfers_all)
                i_all = sum(interfers_all_list)
                interfers_all_list = []
                a_prev_to_z = self.d_sigmoid(self.linear_comb(self.input_layer[d,:], self.weights1[j,:])+self.biases1[0,j])
                z_to_b = 1
                d_ratio_1 = i_all*a_prev_to_z*z_to_b
                self.biases1[0,j] = self.biases1[0,j] - (self.learning_rate_b)*d_ratio_1
    
    def backpropagation_weigth_2_3(self,d):
        for i in range(self.o_l_num):
            for j in range(self.h_l_num):
                E_to_a = self.d_loss_f(d, i)
                a_to_z = (self.d_sigmoid(self.linear_comb(self.hidden_layer,self.weights2[i,:])+self.biases2[0,i]))
                z_to_w = self.hidden_layer[0,j]
                d_ratio_23 = E_to_a*a_to_z*z_to_w
                self.weights2[i,j] = self.weights2[i,j] - self.learning_rate_w*d_ratio_23

    def backpropagation_weigth_1_2(self,d):
        interfers_all_list = []
        for i in range(self.h_l_num):
            for j in range(self.i_l_num):
                for k in range(self.o_l_num):
                    interfers_all__E_to_a = self.d_loss_f(d, k)
                    interfers_all__a_to_z = (self.d_sigmoid(self.linear_comb(self.hidden_layer,self.weights2[k,:])+self.biases2[0,k]))
                    interfers_all__z_to_w = self.weights2[k,i]
                    interfers_all = interfers_all__E_to_a*interfers_all__a_to_z*interfers_all__z_to_w
                    interfers_all_list.append(interfers_all)
                i_all = sum(interfers_all_list)
                interfers_all_list = []
                E_to_z = (i_all)*(self.d_sigmoid(self.linear_comb(self.input_layer[d,:], self.weights1[i,:])+self.biases1[0,i]))
                d_ratio_12 =  E_to_z*self.input_layer[d,j]
                self.weights1[i,j] = self.weights1[i,j] - self.learning_rate_w*d_ratio_12

    def train(self,epoch):
        accuracy_now=0
        start_time  = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        for epoch_num in range(epoch):
            e_start = time.time()
            print("Epoch: "+str(epoch_num+1)+"/"+str(epoch))
            d = 0
            datanum = 0
            for input_data in self.input_layer:
                s_start = time.time()

                #ForwardPropagation

                #ForwardPropagation between layer 1 and layer 2
                for weight1_num in range(self.h_l_num):
                    z1 = self.linear_comb(input_data, self.weights1[weight1_num,:])+self.biases1[0,weight1_num]
                    a1 = self.sigmoid(z1)
                    self.hidden_layer[0,weight1_num] = float(a1)
                # print("ForwardPropagation between layer 1 and layer 2, Done!")

                #ForwardPropagation between layer 2 and layer 3
                for weight2_num in range(self.o_l_num):
                    z2 = self.linear_comb(self.hidden_layer,self.weights2[weight2_num,:])+self.biases2[0,weight2_num]
                    a2 = self.sigmoid(z2)
                    self.output_layer[0,weight2_num] = float(a2)

                    if self.real_output_layer[d][0,:] == self.threshold(a2):
                        if self.threshold(a2) == 1:
                            self.output_list_1.append(a2)
                            accuracy_now += (1)/self.input_layer.shape[0]
                        if self.threshold(a2) == 0:
                            self.output_list_0.append(a2)
                            accuracy_now += (1)/self.input_layer.shape[0]
                        
                    
                # print("ForwardPropagation between layer 2 and layer 3, Done!")


                # BackPropagation 

                #BackPropagation between layer 2 and layer 3
                self.backpropagation_bias_2(d)
                self.backpropagation_weigth_2_3(d)
                # print("BackPropagation between layer 2 and layer 3, Done!")

                #BackPropagation between layer 1 and layer 2
                self.backpropagation_bias_1(d)
                self.backpropagation_weigth_1_2(d)
                # print("BackPropagation between layer 1 and layer 2, Done!")  
                self.loss_geç.append(self.loss_f(d))

                s_end = time.time()
                datanum +=1
                sys.stdout.write("\r")
                sys.stdout.write(f" {datanum}/{self.input_layer.shape[0]} --- duration: {round(s_end-s_start,2)} --- loss: {self.loss_f(d)}")
                sys.stdout.flush()
                
                d +=1

            self.accuracy.append(accuracy_now/((epoch_num+1)))
            self.loss.append(sum(self.loss_geç)/(len(self.loss_geç)))
            self.loss_geç = []


            e_end = time.time()
            print(f" Time elapsed for epoch {epoch_num} is {round((e_end-e_start),5)}")
            situation = ""

            if epoch_num+1 > 1:
                if self.loss[-1]< self.loss[-2]:
                    situation = "Good"

                elif self.loss[-1] >= self.loss[-2]: 
                    situation = "Bad"

            print(f" Loss: {self.loss[-1]} => {situation}")
            print(f" Accuracy: {self.accuracy[-1]} \n")




        end_time  = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print("Start Time= ",start_time)
        print("End Time = ", end_time)
        self.lossv()
        #self.scatter_graph(epoch)
