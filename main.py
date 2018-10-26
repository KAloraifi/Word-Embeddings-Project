import logging
import os

import gensim
from gensim.models import Word2Vec

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)


def show_file_contents(input_file):
    with open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            print(line)
            break


def read_input(input_file):
    """This method reads the input file which is in gzip format"""

    logging.info("reading file {0}...this may take a while!".format(input_file))
    with open(input_file, 'rb') as f:
        for i, line in enumerate(f):

            if (i % 10000 == 0):
                logging.info("read {0} reviews".format(i))
            # do some pre-processing and return list of words for each review
            # text
            yield gensim.utils.simple_preprocess(line)


if __name__ == '__main__':
    abspath = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(abspath, "reviews_data.txt")

    # read the tokenized reviews into a list
    # each review item becomes a series of words
    # so this becomes a list of lists
    FullCorpus = list(read_input(data_file))
    PartOneOfCorpus = FullCorpus[:len(FullCorpus) // 2]
    PartTwoOCorpus = FullCorpus[len(FullCorpus) // 2:]
    logging.info("Done reading dataset file")

    # build vocabulary and train model
    model = Word2Vec(FullCorpus, size=150, window=10, min_count=2)

# summarize the loaded model
print(model)

# summarize vocabulary
words = list(model.wv.vocab)
print(words)

# Testing part:

# print 6 most similar words
print(model.wv.most_similar(positive='low'))

# print similarity between two words in percentage
w1 = 'dirty'
w2 = 'small'
w3 = "bad"
print("The similarity between " + w1 + ", " + w2 + " and " + w3 +" is {}".format(model.wv.similarity(w1, w2, w3)))
