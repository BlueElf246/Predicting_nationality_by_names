import pandas as pd
import torch
import string
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from  torch import nn
import numpy as np
import unicodedata
from dataset_creator import create_model
class LSTMModel_Embed(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, EMBED_SIZE, n_categories):
    super(LSTMModel_Embed, self).__init__()
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.embed = nn.Embedding(input_size, EMBED_SIZE)
    self.lstm = nn.LSTM(input_size = EMBED_SIZE, hidden_size = hidden_size, num_layers = num_layers, batch_first = True, bidirectional = True)
    self.in2output = nn.Linear(hidden_size + hidden_size, n_categories)

  def forward(self, x, hidden_state):

    argmax_x = torch.argmax(x,dim=2)
    embed = self.embed(argmax_x)
    output, hidden_state = self.lstm(embed, hidden_state)

    output = self.in2output(output[:, -1, :]) #last char embed of name include all information of previous ones.
    return output

  def init_hidden(self, batch_size):
    # Initialization two new tensors which are cell and hidden state.
    h0 = torch.zeros(self.num_layers*2,batch_size,self.hidden_size)
    c0 = torch.zeros(self.num_layers*2,batch_size,self.hidden_size)
    hidden = (h0,c0)
    return hidden

  def save_model(self, path):
    torch.save(self.state_dict(), path)

  @staticmethod
  def load_model(path, input_size, hidden_size, num_layers, embed_size, category):
    model = LSTMModel_Embed(input_size, hidden_size, num_layers, embed_size, category)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

class train_model():
    def __init__(self, batch_size, model_name, epoch, layer, n_category=11):
        self.EMBED_SIZE = 10
        self.all_letter = string.ascii_letters + " -"
        self.vocab_size = len(self.all_letter)
        self.hidden_size = 128
        self.learning_rate = 0.001
        self.epochs = epoch
        self.batch_size=batch_size
        self.model_name = model_name
        self.numlayer=layer
        self.num_category=n_category
    def create_lstm_model(self,is_embedding=False):
        # LSTM model with Embedding layer
        model = LSTMModel_Embed(input_size=self.vocab_size,hidden_size= self.hidden_size, num_layers=self.numlayer, EMBED_SIZE=self.EMBED_SIZE, n_categories=self.num_category)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        # defining loss function
        loss_func = nn.CrossEntropyLoss()
        # loss_fn = nn.NLLLoss()
        return model, optimizer, loss_func, self.epochs

    def accuracy_2(self,pred, label):  # Return how much true label match with predictions(max prob. class index)
        return torch.sum(torch.argmax(pred, dim=1) == label).item()

    def lstm_train_test_process(self,train_loader, valid_loader,has_embedding=False):

        # batch_list = [128, 64, 16, 8]
        main_accuracy_dict = {}
        model, optimizer_128, loss_func, epochs = self.create_lstm_model(has_embedding)
        train_loader, valid_loader = train_loader, valid_loader
        best_accuracy = 0  # best accuracy of each batch_size

        for i in range(epochs):
            total_train_loss = 0
            for batch in train_loader:  # batch[0].size = torch.Size([128, 20, 58]) , batch[1].size = torch.Size([128])
                model.zero_grad()
                # performing forward pass
                init_hidden = model.init_hidden(self.batch_size)
                # logits = model(b_input_ids,init_hidden)
                outputs1 = model(batch[0], init_hidden)
                # computing loss
                loss = loss_func(outputs1, batch[1])
                total_train_loss += loss.item()
                # performing a backward pass
                loss.backward()
                # update parameters
                optimizer_128.step()

            # calculate the average loss
            avg_train_loss = total_train_loss / len(train_loader)

            val_accuracy = []
            val_loss = []
            for batch in valid_loader:
                # gradients ignored for a while for testing (it speeds up computation)
                with torch.no_grad():
                    init_hidden = model.init_hidden(self.batch_size)
                    pred = model(batch[0], init_hidden)
                # computing loss
                loss = loss_func(pred, batch[1])
                val_loss.append(loss.item())
                # Validation process
                acc = self.accuracy_2(pred, batch[1])
                val_accuracy.append(acc / self.batch_size)
            val_mean_acc = np.mean(val_accuracy)
            val_mean_loss = np.mean(val_loss)

            if val_mean_acc > best_accuracy:
                best_accuracy = val_mean_acc

            # print performance
            print(
                f'Epoch no: {i + 1} | Train_loss: {round(avg_train_loss, 5)} | Val_loss: {round(val_mean_loss, 5)} | Val_Accuracy: {round(val_mean_acc, 2)}')
        print(f"Best accuracy is : %{best_accuracy * 100} for batch_size ={self.batch_size}")
        main_accuracy_dict[self.batch_size] = best_accuracy
        model.save_model(f'{self.model_name}.pt')
        return main_accuracy_dict
import glob
list_name = glob.glob("/Users/datle/Desktop/mybiggestprj/true_*")
print(list_name)

def read_csv(list_df_name):
    df_final=[]
    for x in list_df_name:
        label= x.split("/")[-1].split('.')[0][5:]
        print(label)
        df = pd.read_csv(x)
        df = df.drop(columns=['label', 'result','Unnamed: 0'])
        df['label'] = [label]*len(df)
        df_final.append(df)
    return pd.concat(df_final, axis=0).reset_index(drop=True)
def label_encoder(df, col):
    le = LabelEncoder()
    return le.fit_transform(df[col]), le
df = read_csv(list_name)
batch_size=64
n_category = len(df['label'].unique())
df['label'], le = label_encoder(df, 'label')
src='final_model'
train_loader, valid_loader = create_model().train_test_loader(batch_size=batch_size, df=df, test_size=0.1)
train = train_model(batch_size=batch_size, model_name=src, epoch=8, layer=2, n_category=n_category)\
    .lstm_train_test_process(train_loader, valid_loader)
