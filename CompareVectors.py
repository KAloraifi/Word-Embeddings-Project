import gensim
from gensim.models import Word2Vec

FullCorpus_model = Word2Vec.load('FullCorpus_model.bin')
PartOneOfCorpus_model = Word2Vec.load('PartOneOfCorpus_model.bin')
PartTwoOCorpus_model = Word2Vec.load('PartTwoOCorpus_model.bin')

ListOfWords = [
               'hotel', 'the', 'gallery', 'is', 'situated', 'in', 'such', 'south', 'kensingtion', 'that',
               'offered', 'us', 'image', 'starwood', 'preferred', 'guest', 'member', 'given', 'small', 'gift',
               'upon', 'check', 'only', 'decor', 'lobby', 'ground', 'floor', 'area', 'stylish', 'modern',
               'reception', 'staff', 'geeting', 'me', 'aloha', 'bit', 'out', 'but', 'guess', 'they', 'are',
               'briefed', 'say', 'keep', 'up', 'coroporate', 'couple', 'fridge', 'magnets', 'box', 'found'
               ]

#//TODO: Get 3 vctors one for each word in the list using the models above.


#//TODO: Allisgn the two smaller vectors and create a combined vector.


#//TODO: Compare the resulted vector with the vector obtained from the full corpus model (Tip: compare using distance)



