# Predicting_nationality_by_names
*This is the project of predicting nationality by names.
*In this project I use the Faker library to build a small binary lstm model to clean the data from names-dataset(link:https://github.com/philipperemy/name-dataset)
*The reason why I choose this dataset instead of by creating fake name to train large lstm model is this 3.5GB data will capture the wide variety of names of a countries, which provide more valuable information than using fake names.
*The downside of this large dataset is it has a lot of noise, since the data taken from 530M facebook users. The user may not name their name as their actual name.
*From that problem, I have create a binary lstm model to clean the raw_dataset.

The workflow:

Given raw_dataset of specific country -> create small dataset where target country will label 1 and a list of other countries label 0 using Faker library -> Train binary lstm on small dataset -> Use trained lstm model to clean the raw dataset and save it in cleaned dataset folder -> train multi languages LSTM model.

* Some insights and reports:
  The process of training small lstm is quite easy and fast, it depend of how many examples you want faker library to create, the common accuracy around 88% -> 96% for ~10 epochs, 1 hidden layer, 128 hiddens unit
  Traning the large LSTM took longer since I use two hidden layer and dataset is large ~100k examples.
  The final accuracy is 85.6% with batch_size=64

* How you use them?
* First you have to download the raw dataset link:https://github.com/philipperemy/name-dataset
* clone my github repo, git clone https://github.com/BlueElf246/Predicting_nationality_by_names.git
* In the run.py file
src is the target language use want to clean from src raw dataset

the second argument in Fake_dataset is list of countries that will be label 0, the last in number of example for 1 label, 0 label
for example:

src="English"

ds_creator = Fake_dataset(src, ['French','India','Italian','Spanish'], [1000,1000])

English ds will have 4k example, French, india,... each will have 1k examples

This will save the created file as src{df_name_num}.csv

df = ds_creator.create_dataset(df_name_num='')


This two lines will create pytoch dataset and train it using lstm model, use can modify epoch, num_category in train_model class init.

train_loader, valid_loader = create_model().train_test_loader(batch_size=batch_size, df=df, test_size=0.1)

train = train_model(batch_size=batch_size, model_name=src, epoch=8, layer=2).lstm_train_test_process(train_loader, valid_loader)

*Finaly run the file run_filter.py to filter raw_dataset

f.Filter(name[11], "English"),the first the path to raw_dataset the second argument is model saved file name use want to use for example, English.pt.

*In train_multi_lstm.py, we will combine all true_*.csv dataset into a large and train it.

list_name = glob.glob("/Users/datle/Desktop/mybiggestprj/true_*")
change it to "/cleaned_dataset/true_*.csv"

*In this file, I bring all the model architecture and training function separately instead of using the same archtecture as small lstm.

The input size will be (number of example, length of char in a batch, 54), where 54 is the vector size encoded from a character.
For more in detail, read my report.

*Reference: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

Thanks for reading <3




