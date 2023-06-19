import pandas as pd
from dataset_creator import *
batch_size=16
src = "English"
ds_creator = Fake_dataset(src, ['French','India','Italian','Spanish'], [1000,1000])
df = ds_creator.create_dataset(df_name_num='')

# df = pd.read_csv("French.csv")
train_loader, valid_loader = create_model().train_test_loader(batch_size=batch_size, df=df, test_size=0.1)
train = train_model(batch_size=batch_size, model_name=src, epoch=8, layer=2).lstm_train_test_process(train_loader, valid_loader)