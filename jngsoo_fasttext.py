import fasttext

# Skipgram model
model = fasttext.load_model('model.bin')
print(model.words) # list of words in dictionary

print(model['채용'])