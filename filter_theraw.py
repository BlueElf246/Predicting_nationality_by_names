import pandas as pd
import numpy as np
from dataset_creator import create_model, LSTMModel_Embed
import torch
import unicodedata
class filter(create_model):
    def preprocess(self,df, label):
        df.columns = ["first_name", 'last_name', 'gender', 'idk']
        df['name'] = df['first_name'] + " " + df['last_name']
        df = df.drop(['first_name', 'last_name', 'gender', 'idk'], axis=1)
        df['label'] = [label] * len(df)
        return df
    def unicode2Ascii(self, word):
        u = ''.join(
            c for c in unicodedata.normalize('NFD', word) if (unicodedata.category(c) != "Mn" and c in self.all_letter))
        return u
    def Filter(self, df_name, label):
        df = pd.read_csv(df_name)
        df = df.iloc[:20000]
        df_new = self.preprocess(df, label)
        df_new = df_new.dropna(subset=['name'])
        df_new['name'] = df_new['name'].apply(self.unicode2Ascii)
        df_new = df_new[(df_new['name'].str.len() <= 24) & (df_new['name'].str.len() > 3)]
        # df_new['name'] = df_new['name'].apply(self.padding)
        X_tensor1,_ = self.padding(df_new, eval=True)
        # for x in range(len(df_new)):
        #     X_tensor.append(self.name2tensor(df_new['name'].iloc[x]))
        # X_tensor1 = torch.cat(X_tensor, dim=0)
        label1 = []
        hidden_size = 128
        model = LSTMModel_Embed.load_model(f'{label}.pt', self.vocab_size, hidden_size, 1, embed_size=10, category=2)
        model.eval()
        score = []
        for x in range(X_tensor1.size()[0]):
            x = torch.unsqueeze(X_tensor1[x], dim=0)
            with torch.no_grad():
                hidden_state = model.init_hidden(batch_size=1)
                output = model(x, hidden_state)
                score.append(output)
                output1 = torch.argmax(output, dim=1).item()
                s = output[0][output1].item()
                if s < 1.8:  # if output1 =0 then +1
                    if output1 == 1:
                        output1 = 0
                label1.append(output1)
        df_new['label'] = label1
        df_new['result'] = score
        df_false = pd.DataFrame(df_new[df_new['label'] == 0])
        df_true = pd.DataFrame(df_new[df_new['label'] == 1])
        df_new.to_csv(f'result_{label}.csv')
        df_false.to_csv(f'false_{label}.csv')
        df_true.to_csv(f'true_{label}.csv')
