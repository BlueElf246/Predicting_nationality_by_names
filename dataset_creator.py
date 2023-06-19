from faker import Faker
from googletrans import Translator
import pandas as pd
import torch
import string
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from  torch import nn
import numpy as np
import unicodedata
language_locale_dict = {
    'English': 'en_US',
    'French': 'fr_FR',
    'German': 'de_DE',
    'Spanish': 'es_ES',
    'Italian': 'it_IT',
    'Japanese': 'ja_JP',
    'Chinese': 'zh_CN',
    'Russian': 'ru_RU',
    'Dutch': 'nl_NL',
    'Portuguese': 'pt_BR',
    'Canada': 'en_CA',
    'Korean': 'ko_KR',
    'Mexico': 'es_MX',
    "India": 'en_IN',
    "United Kingdom": 'en_GB',
    'Australia': 'de_AT'
}


class Fake_dataset():
    def __init__(self, label1_c, label0_c, number_of_example):
        # number_of_example = [#nth example label1, #nth example label0]
        self.label1_c = label1_c #English, French,...
        self.label0_c = label0_c
        self.dct = {}
        self.dct[label1_c] = Faker(language_locale_dict[label1_c])
        for label_0 in label0_c:
            self.dct[label_0] = Faker(language_locale_dict[label_0])
        self.translator = Translator(service_urls=['translate.google.com'])
        self.number_of_example_list=number_of_example
        self.country1=['French', 'German', 'Spanish', 'Italian', 'Canada',
                       'Mexico', 'United Kingdom', 'Australia', 'India', 'Portuguese', "English"]
        self.country0=['Japanese', 'Russian','Chinese','Korean']
        self.all_letter = string.ascii_letters + " -"
        self.vocab_size = len(self.all_letter)
    def unicode2Ascii(self, word):
        u = ''.join(
            c for c in unicodedata.normalize('NFD', word) if (unicodedata.category(c) != "Mn" and c in self.all_letter))
        return u
    def translate_to_eng(self, src): # src: English, French,...
        src_faker = self.dct[src]
        if src in self.country1:
            print('ping')
            return self.unicode2Ascii(src_faker.name())
        dict_name = language_locale_dict[src]
        if src == "Chinese":
            dict_name = dict_name[:2]+"-cn"
        name = src_faker.name()
        print(name)
        name_trans = self.translator.translate(name, src=dict_name[:2], dest='en')
        if name_trans is None:
            return None
        else:
            return name_trans.text
    def create_dataset(self, df_name_num):
        ds=[]
        number_label1, number_label0 = self.number_of_example_list[0], self.number_of_example_list[1]
        if len(self.label0_c) > 1:
            number_label1 = number_label1*len(self.label0_c)
        print(number_label1)
        for x in range(number_label1):
            label1_name = self.translate_to_eng(self.label1_c)
            if label1_name is None:
                continue
            print(x,label1_name)
            ds.append([label1_name, 1])
        for x in self.label0_c:
            for y in range(number_label0):
                label0_name = self.translate_to_eng(x)
                print(x, label0_name)
                if label0_name is None:
                    continue
                ds.append([label0_name, 0])
        df = pd.DataFrame(ds, columns=['name', 'label'])
        df.to_csv(f"{self.label1_c}{df_name_num}.csv")
        return df
# ds_creator = Fake_dataset("Japanese", ['English'], [10,10])
# df = ds_creator.create_dataset()
class create_model():
    def __init__(self):
        self.all_letter = string.ascii_letters + " -"
        self.vocab_size = len(self.all_letter)

    def letter2index(self,letter):
        return self.all_letter.find(letter)

    def name2tensor(self,name):
        name_letter_tensor = torch.zeros(1, len(name), self.vocab_size)
        for idx, letter in enumerate(name):
            name_letter_tensor[0][idx][self.letter2index(letter)] = 1
        return name_letter_tensor

    def padding(self,df, eval=False):
        X_tensor = []
        y_tensor = []
        # extract largest number char in dataset
        max_char = max(df['name'].apply(len))
        df['name'] = df['name'].apply(lambda x: x + '-' * (max_char - len(x)))
        print(df)
        for x in range(len(df)):
            X_tensor.append(self.name2tensor(df['name'].iloc[x]))
            if eval != True:
                y_tensor.append(torch.tensor(df['label'].iloc[x]).view(1, ))
        X_tensor1 = torch.cat(X_tensor, dim=0)
        if eval == True:
            return X_tensor1, y_tensor
        y_tensor1 = torch.cat(y_tensor, dim=0)
        return X_tensor1, y_tensor1
    def train_test_loader(self, batch_size, df, test_size=0.1):
        X_tensor_reshaped, y_tensor_reshaped = self.padding(df)
        X_train, X_test, y_train, y_test = train_test_split(X_tensor_reshaped.numpy()
                                                            , y_tensor_reshaped.numpy(), test_size=test_size, random_state=42,
                                                            stratify=y_tensor_reshaped.numpy())
        train_data = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, drop_last=True)

        valid_data = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
        valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size, drop_last=True)
        return train_loader, valid_loader

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
    def __init__(self, batch_size, model_name, epoch, layer):
        self.EMBED_SIZE = 10
        self.all_letter = string.ascii_letters + " -"
        self.vocab_size = len(self.all_letter)
        self.hidden_size = 128
        self.learning_rate = 0.001
        self.epochs = epoch
        self.batch_size=batch_size
        self.model_name = model_name
        self.numlayer=1
    def create_lstm_model(self,is_embedding=False):
        # LSTM model with Embedding layer
        model = LSTMModel_Embed(input_size=self.vocab_size,hidden_size= self.hidden_size, num_layers=self.numlayer, EMBED_SIZE=self.EMBED_SIZE, n_categories=2)
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


# create_model.train_test_loader(batch_size=self.batch_size, df=self.df, test_size=0.1)