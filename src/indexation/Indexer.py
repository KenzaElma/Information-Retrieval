#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import Counter
import pickle as pkl
import numpy as np
import re


from .TextRepresenter import PorterStemmer
tr = PorterStemmer()


class IndexerSimple:
    """ C'est un module pour cree l'index et l'index inverse de la collection.
    """

    def __init__(self, collection, path, collectionName):
        """Indexer

        Arguments:
            collection dict -- dictionnaire des documents (key:id,value:Document)
            path str -- le chemin ou stocker les index.
            collectionName str -- le nom de la collection cacm ou cism.
        """
        self.collection = collection
        self.index = self.getIndex(path, collectionName)
        self.index_inverse = self.getIndexInverse()
        self.tfs = Counter(
            {word: sum(self.index_inverse[word].values()) for word in self.index_inverse})
        self.numberWords = sum(self.tfs.values())

    def getIndexInverse(self):
        """Construit l'index inverse de la collection 

        Returns:
            [type] -- [description]
        """

        idfs = dict()
        for id, doc in self.collection.items():
            tfs_forDoc = self.getTfsForDoc(doc)
            for w in tfs_forDoc.keys():
                if(w in idfs.keys()):
                    idfs[w][id] = tfs_forDoc[w]
                else:
                    idfs[w] = dict()
                    idfs[w][id] = tfs_forDoc[w]
        return idfs

    def getIndex(self, path, collectionName):
        """Charger ou construit l'index de la collaction.

        Arguments:
            path str -- le chemin ou l'index est stocker.
            collectionName str -- nome de la collection pour retrouver l'index.

        Returns:
            dict(doc(int,str)) -- {d1:{w1d1:occur1,w2d2:occur2,.......,wndn:occurn}}
        """
        if(self.checkIfIndexesCreated(path, collectionName)):
            return pkl.load(open(path+"/"+collectionName+"index.p", "rb"))
        else:
            return self.save(path, collectionName)

    def calculateIndex(self):
        """Construit l'index {d1:{w1d1:occur1,w2d2:occur2,.......,wndn:occurn}}
        """
        self.index = {doc: tr.getTextRepresentation(
            f'{self.collection[doc].title}') for doc in self.collection}

    def getTfsForDoc(self, doc):
        return self.index[doc.id]

    def getIDFForTerm(self, term):
        return np.log((1+(len(self.collection)))/(1+len(self.index_inverse[term])))

    def getTfIDFsForDoc(self, doc):
        tfs_doc = self.getTfsForDoc(doc)
        return dict(map(lambda w: (w, tfs_doc[w]*self.getIDFForTerm(w)), tfs_doc))

    def getTfsForStem(self, stem):
        return self.index_inverse[stem]

    def getTfForStemOnDoc(self, stem, idoc):
        try:
            return self.index[idoc][stem]
        except KeyError:
            return 0

    def getTfIDFsForStem(self, stem):
        d = self.index_inverse[stem].items()
        return dict(map(lambda docOcc: (docOcc[0], self.getIDFForTerm(stem)*docOcc[1]), d))

    def save(self, path, collectionName):
        self.calculateIndex()
        pathIndex = path+"/"+collectionName+"index.p"
        pkl.dump(self.index, open(pathIndex, "wb+"))
        return self.index

    def checkIfIndexesCreated(self, path, collectionName):
        try:
            f = open(path+"/"+collectionName+"index.p")
            f.close()
            return True
        except FileNotFoundError:
            return False

    def getStrDoc(self, idoc: int):
        return f'{self.collection[idoc]}'

    def getHyperlinksTo(self,idoc):
        return list(filter( lambda i:  idoc in self.collection[i].liens ,self.collection))

    def getHyperlinksFrom(self,idoc):
        return self.collection[idoc].liens

class MapReduceIndexer:
    pass
