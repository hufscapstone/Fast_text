# from gensim.models import FastText
# sentences = [["This", "is", "an", "apple"], ["This", "is","a", "pen"]]
#
# model = FastText(sentences, min_count=1)
# is_vector = model['is']
# apples_vector = model['apples']
#

from __future__ import print_function
from gensim.models import KeyedVectors

# Creating the model
ko_model = KeyedVectors.load_word2vec_format('model.vec')

# Getting the tokens
words = []
for word in ko_model.vocab:
    words.append(word)

# Printing out number of tokens available
print("Number of Tokens: {}".format(len(words)))

# Printing out all the tokenized words
print("Words are below\n{}".format(words))

# Printing out the dimension of a word vector
print("Dimension of a word vector: {}".format(
    len(ko_model[words[0]])
))

# Print out the vector of a word
print("Vector components of a word: {}".format(
    ko_model[words[0]]
))

print(words[0])

# Pick a word
find_similar_to = '페이'

# Finding out similar words [default= top 10]
for similar_word in ko_model.similar_by_word(find_similar_to):
    print("Word: {0}, Similarity: {1:.2f}".format(
        similar_word[0], similar_word[1]
    ))


#
# word_add = ['채용']
# word_sub = ['카카']
#
# # Word vector addition and subtraction
# for resultant_word in ko_model.most_similar(
#     positive=word_add, negative=word_sub
# ):
#     print("Word : {0} , Similarity: {1:.2f}".format(
#         resultant_word[0], resultant_word[1]
#     ))

