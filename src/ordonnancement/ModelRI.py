#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:14:40 2020

@author: 3701222
"""

from abc import ABC, abstractmethod
from collections import Counter
import itertools
import random
from itertools import islice
import numpy as np

from indexation.TextRepresenter import PorterStemmer






class IRModel(ABC):

    def __init__(self, index):
        self.index = index

    @abstractmethod
    def getScores(self, query):
        pass

    def getRanking(self, query):
        """Retourne le classement et les score des document pour la Requete donné en parametre

        Arguments:
            query str -- la reqete.

        Returns:
            dict(int,float) -- dictionnaire tel que Key: Idoc et Value: Score
        """

        ranking = {k: v for k, v in sorted(self.getScores(query).items(), key=lambda item: item[1], reverse=True)}
        return ranking

    def getDocumentsContainingTerms(self, qVect):
        idocList = []
        for w in qVect:
            try:
                for idoc in self.index.index_inverse[w]:
                    idocList.append(idoc)
            except KeyError:
                pass
        return idocList


class Vectoriel(IRModel):
    """Model Victoriel 
    """

    def __init__(self, index, weighter, normalized=False):
        """Model Victoriel.

        Arguments:
            index Indexer -- L'objet Indexer.
            weighter Weighter -- L'objet Weighter.

        Keyword Arguments:
            normalized boolean -- boolean pour la normalization ou non (default: {False})
        """
        super().__init__(index)
        self.weighter = weighter
        self.normalized = normalized
    def score(self, iDoc, query):
        qVect = self.weighter.getWeightsForQuery(query)

        dVect = self.weighter.getWeightsForDoc(iDoc)

        communWords = set(dVect.keys()) & set(qVect.keys())

        qVect_values, dVect_values = tuple(
            zip(*[(qVect[w], dVect[w]) for w in communWords]))
        qVect_values, dVect_values = np.array(
            qVect_values), np.array(dVect_values)

        if(self.normalized):

            norm1 = np.sqrt(np.power(qVect_values, 2).sum())
            norm2 = np.sqrt(np.power(dVect_values, 2).sum())

            return 1 - np.dot(qVect_values, dVect_values)/(norm1*norm2)
        else:
            return np.dot(qVect_values, dVect_values)

    def getScores(self, query):
        """retourne les scores des documents pour une requete.

        Arguments:
            query str -- la Requete.

        Returns:
            dict -- les documents avec leur scores.
        """
        qVect = self.weighter.getWeightsForQuery(query)

        idocList = self.getDocumentsContainingTerms(qVect)
        documentsContainigQuery = {idoc: self.score(
            idoc, query) for idoc in idocList}

        return documentsContainigQuery


class ModeleLangue(IRModel):
    def __init__(self, index, _lambda=0.1):
        super().__init__(index)
        self._lambda = _lambda

    def getScores(self, query):
        ps = PorterStemmer()
        qVect = ps.getTextRepresentation(query)
        idocList = self.getDocumentsContainingTerms(qVect)

        def f(idoc):
            return (idoc, self.sequence_probality_on_doc(idoc, qVect))

        return dict(map(f, idocList))

    def sequence_probality_on_doc(self, idoc, words):
        return sum([self.index.tfs[w]*((1-self._lambda)*self.document_model(idoc, w)+self._lambda*self.collection_model(w)) for w in words])
    def document_model(self, idoc, w):
        D = sum(self.index.index[idoc].values())
        return self.index.getTfForStemOnDoc(w, idoc)/D

    def collection_model(self, w):
        return (self.index.tfs[w]+1)/(1+self.index.numberWords)


class Okapi(IRModel):

    def __init__(self, index, k1=1.2, b=0.75):
        super().__init__(index)
        self.k1, self.b = k1, b

    def getScores(self, query):
        ps = PorterStemmer()
        qVect = ps.getTextRepresentation(query)

        avgd = sum(map(lambda i: sum(map(
            lambda w: self.index.index[i][w], self.index.index[i])), self.index.index))/len(self.index.index)

        def f(idoc):
            stems = set(self.index.index[idoc].keys()) & set(qVect.keys())
            def func(stem):

                idf = np.log(1 + len(self.index.index) /
                             (1 + len(self.index.index_inverse[stem])))
                tf = self.index.index[idoc][stem]
                D = sum(
                    map(lambda w: self.index.index[idoc][w], self.index.index[idoc]))
                return idf*(tf/(tf*self.k1*(1-self.b + (self.b * (D/avgd)))))
            return sum(map(func, stems))

        idocList = self.getDocumentsContainingTerms(qVect)
        documentsContainigQuery = {idoc: f(idoc) for idoc in idocList}
        return documentsContainigQuery




class PageRank(IRModel):
    def __init__(self, index, model, n = 50, k = 15, eps=0.0001, d=0.85 ):
        self.index = index
        self.model = model
        self.eps = eps
        self.d = d
        self.n = n 
        self.k = k

    def get_nodes(self, S, n, k):
        Vq = S.copy()
        for idoc in S:
            nexts = self.index.getHyperlinksFrom(idoc)
            nexts = list(filter(lambda i: not i in Vq, nexts))
            if(len(nexts) > k):
                Vq += random.sample(nexts, k)
            else:
                Vq += nexts


        return Vq

    def get_graph(self,nodes):
        G = np.zeros((len(nodes),len(nodes)))

        for idoc in nodes:
            nexts = self.index.getHyperlinksFrom(idoc)
            nexts = list(filter(lambda i: i in nodes, nexts))
            for next_doc in nexts:
                G[self.encode[idoc]][self.encode[next_doc]] = 1
        return G / np.maximum(G.sum(1).reshape(len(nodes), 1), 1)
            
    
    def page_rank(self,G):
        P = np.ones(len(G))

        while(True):
            P_ = (1-self.d)*np.ones(len(G)) + self.d*G.T.dot(P)
            delta = abs(P_-P).sum()
            if(delta<=self.eps):
                return P_
            P = P_

    def getScores(self,query):
        ranking = self.model.getRanking(query)
        S = list(islice(ranking.keys(), self.n))
        Vq = self.get_nodes(S,self.n,self.k)
        self.encode = { Vq[code]:code for code in range(len(Vq))}
        G = self.get_graph(Vq)
        ranking = self.page_rank(G)
        

        return dict(zip(self.encode.keys(),ranking))