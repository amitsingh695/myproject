import torch
import torch.nn as nn
import numpy as np
import warnings
import time
import pandas as pd
from torch.autograd import Variable
warnings.simplefilter("ignore")

ip_dimension=1
op_dimension=1
num_layers=2
hidden_size=80
seq_len=250
batch_size=64
dropout=0.3

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=80, num_layers=2,dropout=0.3,batch_size=64,seq_len=250,output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.dropout = dropout
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size,num_layers,dropout=dropout,batch_first=True)

        self.linear = nn.Linear(hidden_layer_size*seq_len, output_size)

        self.hidden_cell = (torch.zeros(num_layers,batch_size,self.hidden_layer_size).cuda(),
                            torch.zeros(num_layers,batch_size,self.hidden_layer_size).cuda())

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.linear(lstm_out.reshape(len(input_seq), -1))
        return predictions
def val(model,val_seq,val_lab,loss_function):
    model.eval()
    loss=0
    L = len(val_seq)
    end=int(L-model.batch_size)
    with torch.no_grad():
        for j in range(0,end,model.batch_size):
              model.hidden_cell = (torch.zeros(model.num_layers,model.batch_size,model.hidden_layer_size).cuda(),
                              torch.zeros(model.num_layers,model.batch_size, model.hidden_layer_size).cuda())
              
              ip_data = torch.FloatTensor(val_seq[j:j+batch_size])
              ip_data = ip_data.cuda()
              ip_data = Variable(ip_data)

              lab = torch.FloatTensor(val_lab[j:j+batch_size])
              lab = lab.cuda()
              lab = Variable(lab)
              
              ip_data = ip_data.resize(batch_size,seq_len,1)
              
              y_pred = model(ip_data)
              single_loss = loss_function(y_pred, lab)
              loss += single_loss.item()
              # break
    model.train()
    return loss

def get_errors(model,val_seq,val_lab):
    model.eval()
    L = len(val_seq)
    end=int(L-model.batch_size)
    errors =[]
    pred=[]
    with torch.no_grad():
        for j in range(0,end,model.batch_size):
            model.hidden_cell = (torch.zeros(model.num_layers,model.batch_size,model.hidden_layer_size).cuda(),
                          torch.zeros(model.num_layers,model.batch_size, model.hidden_layer_size).cuda())

            ip_data = torch.FloatTensor(val_seq[j:j+batch_size])
            ip_data = ip_data.cuda()
            ip_data = Variable(ip_data)

            lab = torch.FloatTensor(val_lab[j:j+batch_size])
            lab = lab.cuda()
            lab = Variable(lab)

            ip_data = ip_data.resize(batch_size,seq_len,1)

            y_pred = model(ip_data)
            
            errs=torch.abs(y_pred-lab).detach().cpu().numpy().flatten()
            errors.extend(errs)
            pred.extend(y_pred.detach().cpu().numpy().flatten())
#             if j>200:
#                 break

    return errors,pred

model = LSTM(ip_dimension,hidden_size,num_layers,dropout,batch_size,seq_len,op_dimension)
model = model.cuda()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
lr=0.0001
print(model)
training=False
load=False
decay=True

def file_load_np_array(file_name,format):
    arr=np.loadtxt(file_name,dtype=format)
    return arr

def create_inout_sequences(input_data, tw):
    inout_seq = []
    inout_lab = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append(train_seq)
        inout_lab.append(train_label)
    return inout_seq,inout_lab
def save_model(model,optimizer,epoch,loss,file_name):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, "output/"+file_name)
def decay_lr(optimizer,lr):
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
    
# param_name="pid05520"

file_name_arr=[]
with open("htr_ts_mapping_new.csv") as f:
    for lines in f:
        l_arr=lines.strip().split()
        file_name_arr.append([l_arr[0],l_arr[1]])

for file_pid in file_name_arr:
    
    if training==True:
        file_loc="merged_"+"_".join(file_pid)+".npy"
        # data=file_load_np_array(file_loc,'str')
        data=np.load(file_loc)
        start_range=0#3000000
        stop_range=int(np.shape[0]*0.8)#3600000
        full_seq,full_lab=create_inout_sequences(data[start_range:stop_range,-1].astype(np.float),seq_len)
        st_epoch = 0
        best_loss = 1000000000000
        
        train_split,val_split=0.90,0.10
        train_pos=int(len(full_seq)*train_split)
        train_seq=full_seq[:train_pos]
        train_lab=full_lab[:train_pos]
    #     val_pos=train_pos+int(len(full_seq)*val_split)
        val_seq=full_seq[train_pos:]
        val_lab=full_lab[train_pos:]
    #     test_seq=full_seq[val_pos:]
    #     test_lab=full_lab[val_pos:]
        
        if load==True:
            checkpoint = torch.load("output/checkpoint"+param_name+".tar")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            st_epoch = checkpoint['epoch']
            bt_m = torch.load("output/best_model_trend_present"+param_name+".tar")
            best_loss = bt_m['loss']
            if decay ==True:
                lr=lr*0.1
                optimizer=decay_lr(optimizer,lr)
        # print(loss,epoch)
        
            file = open('output/loss'+param_name+'.txt',mode='r')
            all_of_it = file.read()
            file.close()

            file = open('output/loss'+param_name+'.txt',mode='a')
            file.writelines(all_of_it)
            file.close()
        
        
        epochs = 90
        L = len(train_seq)
        end=int(L-model.batch_size)

        for i in range(st_epoch,epochs):    
            loss=0
            model.train()
            print(i)    
            for j in range(0,end,model.batch_size):
                # st=time.time()
                optimizer.zero_grad()
                model.hidden_cell = (torch.zeros(model.num_layers,model.batch_size,model.hidden_layer_size).cuda(),
                                torch.zeros(model.num_layers,model.batch_size, model.hidden_layer_size).cuda())

                ip_data = torch.FloatTensor(train_seq[j:j+model.batch_size])
                ip_data = ip_data.cuda()
                ip_data = Variable(ip_data)

                lab = torch.FloatTensor(train_lab[j:j+model.batch_size])
                lab = lab.cuda()
                lab = Variable(lab)

                ip_data = ip_data.resize(model.batch_size,seq_len,1)

                y_pred = model(ip_data)

                single_loss = loss_function(y_pred,lab )
                single_loss.backward()
                loss += single_loss.item()
                optimizer.step()
                # et=time.time()
            #     print(et-st)
                # break


            # break

            print('epoch: ',i,'train loss: ',loss)
            val_loss=val(model,val_seq,val_lab,loss_function)
            print('epoch: ',i,'val loss: ',val_loss)
            # print(type(best_loss))
            if i==0:
              best_loss=val_loss
              save_model(model,optimizer,i,best_loss,"best_model_trend_present"+param_name+".tar")
            elif val_loss<best_loss:
              best_loss=val_loss
              save_model(model,optimizer,i,best_loss,"best_model_trend_present"+param_name+".tar")
            elif decay==True:
              lr=lr*0.05
              optimizer=decay_lr(optimizer,lr)
            save_model(model,optimizer,i,val_loss,"checkpoint"+param_name+".tar")
            f = open("output/loss"+param_name+".txt", "a")
            f.write(str(i)+" "+str(loss)+" "+str(val_loss)+"\n")
            f.close() 

        print('epoch: ',i,'train loss: ',loss)
        val_loss=val(model,val_seq,val_lab,loss_function)
        print('epoch: ',i,'val loss: ',val_loss)


    if training==False:
        file_loc=['input/cho_pid5520_test.txt']
        data=file_load_np_array(file_loc[0],format='float')#[0:400000]
        test_seq,test_lab=create_inout_sequences(data[:,-1],seq_len)
            
        checkpoint = torch.load("output/best_model_trend_present"+param_name+".tar")
        model.load_state_dict(checkpoint['model_state_dict'])
        errors,pred=get_errors(model,test_seq,test_lab)
        
        alpha = 0.55
        df = pd.DataFrame(errors, columns=['errors'])
        smoothed_errors=df.ewm(alpha=alpha).mean().to_numpy().flatten()
        
        def get_threshold(smoothed_errors):
            val=[]
            m=np.mean(smoothed_errors)
            s=np.std(smoothed_errors)
            for i in range(1,11):
                thres=m+i*s
                gt_indices=np.where(smoothed_errors>=thres)[0]
                lt_indices=np.where(smoothed_errors<thres)[0]
                delta_m=m-np.mean(smoothed_errors[lt_indices])
                delta_s=s-np.std(smoothed_errors[lt_indices])
                no_a_pts=len(gt_indices)

                #Getting the sequences of anomalous points, difference between indices will be greater than 1 when sequence breaks
                cont_seq=gt_indices[1:]-gt_indices[:-1]
                len_cont_seq=len(np.where(cont_seq>1)[0])+1
                val.append(((delta_m/m)+(delta_s/s))/(no_a_pts+len_cont_seq**2))
            return m+(np.argmax(val)+1)*s
        def anomaly_scores(smoothed_errors):
            thres=get_threshold(smoothed_errors)
            ana_score=np.zeros(len(smoothed_errors))
            m=np.mean(smoothed_errors)
            s=np.std(smoothed_errors)
            gt_indices=np.where(smoothed_errors>=thres)[0]
            lt_indices=np.where(smoothed_errors<thres)[0]

            cont_seq=gt_indices[1:]-gt_indices[:-1]
            cont_seq_br_points=np.where(cont_seq>1)[0]+1
            cont_seq_br_pts=cont_seq_br_points.tolist()
            cont_seq_br_pts.insert(0,0)
            cont_seq_br_pts.insert(len(cont_seq_br_pts),len(gt_indices))
            print(thres,len(cont_seq_br_pts),len(gt_indices))
            ana_seq_id_len={}
            for i in range(0,len(cont_seq_br_pts)-1):
                start=cont_seq_br_pts[i]
                end=cont_seq_br_pts[i+1]
                ana_seq=smoothed_errors[gt_indices[start:end]]
                if len(gt_indices[start:end])>1:
                    ana_seq_id_len[gt_indices[start]]=gt_indices[start:end]
        #         print(ana_seq)
                ana_seq_score=(np.max(ana_seq)-thres)/(m+s)
                ana_score[gt_indices[start:end]]=ana_seq_score

            return ana_score,ana_seq_id_len
        ana_score,ana_seq_id_len=anomaly_scores(smoothed_errors)
        np.savetxt('output/ana_score'+param_name+'.txt',ana_score)
        np.savetxt('output/pred'+param_name+'.txt',pred)
