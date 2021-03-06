import numpy
from gensim.models import Word2Vec

FullCorpus_model = Word2Vec.load('FullCorpus_model.bin')
PartOneOfCorpus_model = Word2Vec.load('PartOneOfCorpus_model.bin')
PartTwoOCorpus_model = Word2Vec.load('PartTwoOCorpus_model.bin')


def getVectorsFromModel(model, wordsList):
    """
    This method retrieves vector representation of every word in wordsList from the provided model,
    then adds it to a list of vector representations and return it.

    Similar to Word2Vec's way of accessing a specified word's vector (e.g. model["hello"] returns vector)
    but for a list of words.
    """

    vectorsList = []
    for w in wordsList:
        vector = model[w]
        vectorsList.append(vector)
    return vectorsList


def combineVectors_SummationStrategy(VectorList1, VectorList2):
    """
    This method takes two vector lists and combine each element (vector) with its corresponding element (vector)
    and then add the new vector in a new list. Repeat...
    Return the new list
    """

    combinedVectorsList = []
    for vector1, vector2 in zip(VectorList1, VectorList2):
        newVector = vector1 + vector2
        combinedVectorsList.append(newVector)
    return combinedVectorsList

def combineVectors_AverageStrategy(VectorList1, VectorList2):
    """
    This method takes two vector lists and combine each element (vector) with its corresponding element (vector)
    and then add the new vector in a new list. Repeat...
    Return the new list
    """

    combinedVectorsList = []
    for vector1, vector2 in zip(VectorList1, VectorList2):
        newVector = (vector1 + vector2)/2
        combinedVectorsList.append(newVector)
    return combinedVectorsList

def compareVectors(VectorList1, VectorList2):
    """
    This method takes two vector lists and compare each element (vector) with its corresponding element (vector)
    and then calculate the difference between them and add it to totalDistance. Repeat...
    Return totalDistance
    """
    totalDistance = 0
    for vector1, vector2 in zip(VectorList1, VectorList2):
        distance = numpy.linalg.norm(vector1 - vector2)
        totalDistance += distance
    return totalDistance


# This will create a list of all common words in the two corpus parts models.
part_one_corpus_set = set(list(PartOneOfCorpus_model.wv.vocab))
part_two_corpus_set = set(list(PartTwoOCorpus_model.wv.vocab))
ListOfCommonWords = list(part_one_corpus_set.intersection(part_two_corpus_set))

#Create vectors list for each model
FullCorpus_VectorList = getVectorsFromModel(FullCorpus_model, ListOfCommonWords)
PartOneOfCorpus_VectorList = getVectorsFromModel(PartOneOfCorpus_model, ListOfCommonWords)
PartTwoOCorpus_VectorList = getVectorsFromModel(PartTwoOCorpus_model, ListOfCommonWords)

#combine vectors using different strategies
Combined_VectorsList = combineVectors_SummationStrategy(PartOneOfCorpus_VectorList, PartTwoOCorpus_VectorList)
Combined_VectorsList2 = combineVectors_AverageStrategy(PartOneOfCorpus_VectorList, PartTwoOCorpus_VectorList)

#Calculate distance between combined vector and the full corpus vector
TotalDis = compareVectors(FullCorpus_VectorList, Combined_VectorsList)
TotalDis2 = compareVectors(FullCorpus_VectorList, Combined_VectorsList2)
AverageDis = TotalDis / len(ListOfCommonWords)
AverageDis2 = TotalDis2 / len(ListOfCommonWords)

print("Average distance for all vectors using summation strategy is: {:.2f}".format(AverageDis))
print("Average distance for all vectors using average strategy is: {:.2f}".format(AverageDis2))

