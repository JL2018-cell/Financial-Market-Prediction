from sklearn.model_selection import train_test_split
#Remove some data in given dataset randomly.
def sampling(df, test_size):
    X_train, X_test = train_test_split(df, test_size = test_size, shuffle = True)
    return X_train
