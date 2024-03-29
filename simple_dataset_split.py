from pandas import read_csv

df = read_csv("data/archive/sign_mnist/sign_mnist.csv")

#Sampling all the csv rows in random order
df = df.sample(frac=1, ignore_index=True, random_state=4294967295)

total = df.shape[0]

train_index = int(0.6 * total)
validation_index = int(0.8 * total)

train = df.iloc[:train_index,:].reset_index(drop=True)
validation = df.iloc[train_index:validation_index,:].reset_index(drop=True)
test = df.iloc[validation_index:,:].reset_index(drop=True)

train.to_csv("data/archive/train/train/train.csv", index=False)
validation.to_csv("data/archive/train/validation/validation.csv", index=False)
test.to_csv("data/archive/test/test.csv", index=False)