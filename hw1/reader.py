import pickle

filename = '/tmp/tmpovm_ynvc'
f = open(filename, 'ab')
loaded_stuff = pickle.load(f)
