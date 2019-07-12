
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tkinter


# In[2]:


def data_processing(raw_data, r, c):
    data = [x.replace(' ', '0') for x in raw_data]
    data = [x[:c] for x in data]
    data = list(filter(lambda x: x != '\n', data))
    data = [list(map(int,x)) for x in data]
    data = np.array(data)

    return data

def read_file(file_name):
    
    file_address = 'DataSet\\Hopfield_dataset\\'+file_name+'.txt'
    with open(file_address) as f:
        raw_data = f.readlines()
        
    if((file_name=='Basic_Training') or (file_name=='Basic_Testing')):
        repeat_data_list = []
        row = 12
        col = 9
        data = data_processing(raw_data, row, col)
    else:
        row = 10
        col = 10
        data = data_processing(raw_data, row, col)
        repeat_data_list = [2,3,5,10,11,14]
        for i in repeat_data_list:
            data = np.append(data,data[(i-1)*row:i*row]).reshape((-1,col))
    return data, row, col, len(repeat_data_list)

def data_preprocess(data, n):
    data[data==0] = -1
    return data.reshape((n,-1))

def train_hopfield(data, row, col, n):
    dim = row*col
    W = np.zeros((dim,dim))
    for i in range(n):
        x = data[i:i+1,:]
        X = x.T.dot(x)
        if(i == 0):
            tmp_X = X
        else:
            tmp_X = tmp_X+X
        W = tmp_X/dim-(np.identity(dim)*(n/dim))
    return W

def predict(test_data, W):
    theta = W.sum(axis = 1)
    prediction_x = np.sign(np.matmul(W,test_data.T).reshape(-1)-theta)
    index, = np.where(prediction_x == 0)
    if(index.size != 0):
        for j in index:
            prediction_x[index] = test_data[0][index]
    prediction_X = prediction_x.reshape([row,col])
    return prediction_X 

def converge_check(data, prediction, test_x, sample_n):
    equal = 1

    #if(np.array_equal(prediction.reshape(-1), data[sample_n:sample_n+1,:][0])):
    #    equal = 0
    if(np.array_equal(prediction.reshape(-1), test_x[0])):
        equal = 0
    return equal

def predict2end(data, test_x, W, sample_n):
    iteration = 1
    prediction = predict(test_x, W)
    while(converge_check(data, prediction, test_x, sample_n)):
        test_x = prediction.reshape((1,-1))
        prediction = predict(test_x, W)
        iteration = iteration+1
        if(iteration>=100):
            break
    return prediction, iteration


# In[10]:


from tkinter import *
from tkinter.ttk import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

#global data, test_data, row, col, train_sample_n, test_sample_n, prediction_list, show_fig_n, iteration_list, repeat_n

def clicked_train():
    global data, test_data, row, col, train_sample_n, test_sample_n, prediction_list, show_fig_n, iteration_list, repeat_n
    W = train_hopfield(data, row, col, train_sample_n)
    t = 0
    f = 0
    
    train_sample_n = train_sample_n-repeat_n
    test_sample_n = test_sample_n-repeat_n
    data = data[:train_sample_n,:]
    test_data = test_data[:test_sample_n,:]
    
    prediction_list = np.array([])
    iteration_list = list()
    for i in range(test_sample_n):
        test_x = data[i:i+1,:]
        prediction,iteration = predict2end(data, test_x, W, i)
        prediction_list = np.append(prediction_list,prediction)
        iteration_list.append(iteration)

        if(np.array_equal(prediction.reshape(-1), data[i:i+1,:][0])):
            t = t+1
    acc = t/train_sample_n
    prediction_list = prediction_list.reshape((train_sample_n,-1))
    
    show_fig_n = 0
    #test fig
    fig1 = Figure(figsize=(2,2))
    zero_fig = fig1.add_subplot(111)
    plot_number(fig1, zero_fig, test_data[0:1,:][0], 1, 4, row, col)
    fig2 = Figure(figsize=(2,2))
    one_fig = fig2.add_subplot(111)
    plot_number(fig2, one_fig, test_data[1:2,:][0], 4, 4, row, col)
    fig3 = Figure(figsize=(2,2))
    two_fig = fig3.add_subplot(111)
    plot_number(fig3, two_fig, test_data[2:3,:][0], 7, 4, row, col)
    
    #recall fig
    fig1 = Figure(figsize=(2,2))
    zero_fig = fig1.add_subplot(111)
    plot_number(fig1, zero_fig, prediction_list[0], 1, 5, row, col)
    fig2 = Figure(figsize=(2,2))
    one_fig = fig2.add_subplot(111)
    plot_number(fig2, one_fig, prediction_list[1], 4, 5, row, col)
    fig3 = Figure(figsize=(2,2))
    two_fig = fig3.add_subplot(111)
    plot_number(fig3, two_fig, prediction_list[2], 7, 5, row, col)
    
    var_acc.set('All Sample Accuracy:\n'+str(round(acc,4)))
    lbl_acc = Label(window, textvariable=var_acc)
    lbl_acc.grid(column=10, row=0)
    
    var_iteration1.set('iteration次數:'+str(iteration_list[0]))
    lbl_iteration1 = Label(window, textvariable=var_iteration1) 
    lbl_iteration1.grid(column=1, row=7)
    var_iteration2.set('iteration次數:'+str(iteration_list[1]))
    lbl_iteration2 = Label(window, textvariable=var_iteration2) 
    lbl_iteration2.grid(column=4, row=7)
    var_iteration3.set('iteration次數:'+str(iteration_list[2]))
    lbl_iteration3 = Label(window, textvariable=var_iteration3) 
    lbl_iteration3.grid(column=7, row=7)
    
    var_sample2.set('Testing Data')
    lbl_sample2 = Label(window, textvariable=var_sample2) 
    lbl_sample2.grid(column=0, row=4)
    var_sample3.set('Prediction Recall')
    lbl_sample3 = Label(window, textvariable=var_sample3) 
    lbl_sample3.grid(column=0, row=5)

def plot_number(fig, sub_fig, data, pig_col, pig_row, r ,c):
    sub_fig.imshow(data.reshape(r,c), cmap='gray')
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().grid(column=pig_col, row=pig_row, columnspan=3)
    canvas.show()

def combobox_selected(eventObject):
    global data, test_data, row, col, train_sample_n, test_sample_n, repeat_n
    file_name = combo_data.get()
    
    data, row, col, repeat_n = read_file(file_name+'_Training')
    test_data, row, col, repeat_n = read_file(file_name+'_Testing')
    train_sample_n = int(len(data)/row)
    test_sample_n = int(len(test_data)/row)
    data = data_preprocess(data, train_sample_n)
    test_data = data_preprocess(test_data, test_sample_n)
    
    fig1 = Figure(figsize=(2,2))
    zero_fig = fig1.add_subplot(111)
    plot_number(fig1, zero_fig, data[0:1,:][0], 1, 3, row, col)
    fig2 = Figure(figsize=(2,2))
    one_fig = fig2.add_subplot(111)
    plot_number(fig2, one_fig, data[1:2,:][0], 4, 3, row, col)
    fig3 = Figure(figsize=(2,2))
    two_fig = fig3.add_subplot(111)
    plot_number(fig3, two_fig, data[2:3,:][0], 7, 3, row, col)

    var_sample1.set('Training Data')
    lbl_sample1 = Label(window, textvariable=var_sample1) 
    lbl_sample1.grid(column=0, row=3)
    
def clicked_next():
    global data, test_data, row, col, train_sample_n, test_sample_n, prediction_list, show_fig_n, iteration_list
    #print(prediction_list,prediction_list[0],len(prediction_list))
    prediction_list = prediction_list.reshape((train_sample_n,-1))
    
    show_fig_n = (show_fig_n+3)%train_sample_n
    fig1 = Figure(figsize=(2,2))
    zero_fig = fig1.add_subplot(111)
    plot_number(fig1, zero_fig, data[show_fig_n:show_fig_n+1,:][0], 1, 3, row, col)
    fig2 = Figure(figsize=(2,2))
    one_fig = fig2.add_subplot(111)
    plot_number(fig2, one_fig, data[show_fig_n+1:show_fig_n+2,:][0], 4, 3, row, col)
    fig3 = Figure(figsize=(2,2))
    two_fig = fig3.add_subplot(111)
    plot_number(fig3, two_fig, data[show_fig_n+2:show_fig_n+3,:][0], 7, 3, row, col)
    
    #test fig
    fig1 = Figure(figsize=(2,2))
    zero_fig = fig1.add_subplot(111)
    plot_number(fig1, zero_fig, test_data[show_fig_n:show_fig_n+1,:][0], 1, 4, row, col)
    fig2 = Figure(figsize=(2,2))
    one_fig = fig2.add_subplot(111)
    plot_number(fig2, one_fig, test_data[show_fig_n+1:show_fig_n+2,:][0], 4, 4, row, col)
    fig3 = Figure(figsize=(2,2))
    two_fig = fig3.add_subplot(111)
    plot_number(fig3, two_fig, test_data[show_fig_n+2:show_fig_n+3,:][0], 7, 4, row, col)
    
    #recall fig
    fig1 = Figure(figsize=(2,2))
    zero_fig = fig1.add_subplot(111)
    plot_number(fig1, zero_fig, prediction_list[show_fig_n], 1, 5, row, col)
    fig2 = Figure(figsize=(2,2))
    one_fig = fig2.add_subplot(111)
    plot_number(fig2, one_fig, prediction_list[show_fig_n+1], 4, 5, row, col)
    fig3 = Figure(figsize=(2,2))
    two_fig = fig3.add_subplot(111)
    plot_number(fig3, two_fig, prediction_list[show_fig_n+2], 7, 5, row, col)
    
    var_iteration1.set('iteration次數:'+str(iteration_list[show_fig_n]))
    lbl_interation1 = Label(window, textvariable=var_iteration1) 
    lbl_interation1.grid(column=1, row=7)
    var_iteration2.set('iteration次數:'+str(iteration_list[show_fig_n+1]))
    lbl_interation2 = Label(window, textvariable=var_iteration2) 
    lbl_interation2.grid(column=4, row=7)
    var_iteration3.set('iteration次數:'+str(iteration_list[show_fig_n+2]))
    lbl_interation3 = Label(window, textvariable=var_iteration3) 
    lbl_interation3.grid(column=7, row=7)

window = Tk()

window.title("Perceptron")
window.geometry('1200x1000')
var_acc = StringVar()
var_iteration1 = StringVar()
var_iteration2 = StringVar()
var_iteration3 = StringVar()

var_sample1 = StringVar() 
var_sample2 = StringVar() 
var_sample3 = StringVar() 

lbl_data = Label(window, text="Select Dataset")
lbl_data.grid(column=0, row=0)
combo_data = Combobox(window, width=10)
combo_data['values']= ('Basic','Bonus')
combo_data.grid(column=1, row=0)
combo_data.bind("<<ComboboxSelected>>", combobox_selected)

train_btn = Button(window, text="Start training", command = clicked_train)
train_btn.grid(column=3, row=0)
train_btn = Button(window, text="Next 3 Sample", command = clicked_next)
train_btn.grid(column=4, row=0)




window.mainloop()

