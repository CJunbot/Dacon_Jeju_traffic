import pandas as pd


def isnull_in_data(train, test):
    print("Train sfa missed values:\n")
    print(train.isnull().sum())
    print("-----------------------------------")
    print("Test sfa missed values:\n")
    print(test.isnull().sum())


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 400)
train = pd.read_parquet('../data/train_bus1.parquet')
test = pd.read_parquet('../data/test.parquet')

print(train.shape)
print("-----------------------------------")
print(train.corrwith(train['target']).sort_values())
print("-----------------------------------")
print(isnull_in_data(train, test))
print("-----------------------------------")
print(train.describe())
print("-----------------------------------")
print(train.head(100))
#print(train.describe(include=['O']))
#print(test.describe(include=['O']))
#print(sex.describe(include=['O']))

