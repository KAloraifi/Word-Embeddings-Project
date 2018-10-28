import gensim
import numpy
from gensim.models import Word2Vec

FullCorpus_model = Word2Vec.load('FullCorpus_model.bin')
PartOneOfCorpus_model = Word2Vec.load('PartOneOfCorpus_model.bin')
PartTwoOCorpus_model = Word2Vec.load('PartTwoOCorpus_model.bin')

def getVectorsFromModel(model, wordsList):
    vectorsList = []
    for w in wordsList:
        vector = model[w]
        vectorsList.append(vector)
    return vectorsList

def getTotalDis(VectorList1,VectorList2):
    totalDis = 0
    for vector1,vector2 in zip(VectorList1, VectorList2):
        A = numpy.array(vector1)
        B = numpy.array(vector2)
        dist = numpy.linalg.norm(A - B)
        totalDis += dist
    return totalDis

def combineVectors(VectorList1,VectorList2):
    combinedVectorsList = []
    for vector1,vector2 in zip(VectorList1, VectorList2):
        A = numpy.array(vector1)
        B = numpy.array(vector2)
        combinedVectorsList = A+B
    return combinedVectorsList

ListOfCommonWords =\
                [
                'scone', 'downtowns', 'satisfactorily', 'meticulously', 'remit', 'eyelid', 'absolutly', 'aproximately',
                'disppointing', 'tons', 'rancheros', 'someplace', 'caveat', 'iin', 'abigail', 'prowess', 'haste',
                'accomidating', 'artisan', 'northumberland', 'thrown', 'regulating', 'haggard', 'garcia', 'clubbing',
                'etre', 'allo', 'negligible', 'blowdrying', 'sakes', 'reaaly', 'taht', 'reschedule',
                'nickeled', 'accumulates', 'differnent', 'trek', 'appologies', 'clanked','implementing', 'traffic',
                'streetlife', 'perched', 'whistled', 'wirless', 'ventilators', 'classless', 'goers', 'bitterly',
                'facing', 'inverted', 'bein', 'jacuzzis', 'allbeit', 'berber', 'leery', 'miserables','rama', 'flannels'
               ]
#This will be used later to get all common words in both models.
# ListOfCommonWords = list(set(list(PartOneOfCorpus_model.wv.vocab)).intersection(list(PartTwoOCorpus_model.wv.vocab)))

#Get 3 vctors one for each word in the list using the models above.
FullCorpus_VectorList = getVectorsFromModel(FullCorpus_model, ListOfCommonWords)
PartOneOfCorpus_VectorList = getVectorsFromModel(PartOneOfCorpus_model, ListOfCommonWords)
PartTwoOCorpus_VectorList = getVectorsFromModel(PartTwoOCorpus_model, ListOfCommonWords)

#Allisgn the two smaller vectors and create a combined vector.
CombinedVectorsList = combineVectors(PartOneOfCorpus_VectorList, PartTwoOCorpus_VectorList)

#//Compare the resulted vector with the vector obtained from the full corpus model (Tip: compare using distance)
TotalDis = getTotalDis(FullCorpus_VectorList, CombinedVectorsList)
print("Average distance for all vectors is :",TotalDis/len(ListOfCommonWords))
