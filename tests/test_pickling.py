import pickle
# test pickle
print("loading pickle")
with open('data/training_data.pkl', "rb") as fp:
    df = pickle.load(fp)
print('Data check..')
print(df)