from filter_theraw import filter
from dataset_creator import create_model
name=['/Users/datle/Downloads/name_dataset/data/MX.csv',
 '/Users/datle/Downloads/name_dataset/data/CA.csv',
 '/Users/datle/Downloads/name_dataset/data/ES.csv',
 '/Users/datle/Downloads/name_dataset/data/DE.csv',
 '/Users/datle/Downloads/name_dataset/data/KR.csv',
 '/Users/datle/Downloads/name_dataset/data/IN.csv',
 '/Users/datle/Downloads/name_dataset/data/FR.csv',
 '/Users/datle/Downloads/name_dataset/data/JP.csv',
 '/Users/datle/Downloads/name_dataset/data/GB.csv',
 '/Users/datle/Downloads/name_dataset/data/CN.csv',
 '/Users/datle/Downloads/name_dataset/data/AT.csv',
 '/Users/datle/Downloads/name_dataset/data/US.csv',
 '/Users/datle/Downloads/name_dataset/data/IT.csv',
 '/Users/datle/Downloads/name_dataset/data/PT.csv',]
f = filter()
f.Filter(name[11], "English")