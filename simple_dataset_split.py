from pandas import read_csv

df = read_csv("data/archive/sign_mnist/sign_mnist.csv")

#Sampling all the csv rows in random order
df = df.sample(frac=1, ignore_index=True, random_state=4294967295) #.reset_index(inplace=True, drop=True)

total = df.shape[0]

train_index = 0.8 * total
validation_index = 0.9 * total

train = df.iloc[:train_index,:]
validation = df.iloc[train_index:validation_index,:]
test = df.iloc[validation_index:,:]

train.to_csv("data/archive/train/train/train.csv")
validation.to_csv("data/archive/train/validation/validation.csv")
test.to_csv("data/archive/test/test.csv")