from gensim.models import Word2Vec

# define training data
FullCorpus = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
              ['this', 'is', 'the', 'second', 'sentence'],
              ['yet', 'another', 'sentence'],
              ['one', 'more', 'sentence'],
              ['and', 'the', 'final', 'sentence']]
FirstPartOfCorpus = FullCorpus[:len(FullCorpus) // 2]
SecondPartofCorpus = FullCorpus[len(FullCorpus) // 2:]
# train model
model = Word2Vec(FullCorpus, min_count=1)
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
# access vector for one word
print(model['sentence'])
# save model
model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
print(new_model)
