import pickle

with open('data_dict.pickle', 'rb') as f:
    data_dict = pickle.load(f)

print(data_dict)