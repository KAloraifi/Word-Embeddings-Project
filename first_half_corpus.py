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
    """This method reads the input file which is in txt format"""

    logging.info("reading file {0}...this may take a while".format(input_file))
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
    logging.info("Done reading data file")

    model = Word2Vec(PartOneOfCorpus, size=150, window=10, min_count=2)

# summarize the loaded model
print(model)
