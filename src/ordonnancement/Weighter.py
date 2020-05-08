#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:57:54 2020

@author: 3701222
"""


from abc import ABC, abstractmethod
from indexation.TextRepresenter import PorterStemmer
import numpy as np
from collections import Counter


ps = PorterStemmer()


class Weighter(ABC):
    """Une class abstraite Weighter pour un Document, Mot ou Requête.
    """

    def __init__(self, index):
        """Weighter

        Arguments:
            index Object -- l'objet index.
        """
        self.index = index

    @abstractmethod
    def getWeightsForDoc(self, idDoc):
        """Retourne les poids des termes pour un document dont l’identifiant est idDoc.

        Arguments:
            idDoc int -- l'identifiant d'un document.
        """
        pass

    @abstractmethod
    def getWeightsForStem(self, stem):
        """Retourne les poids du terme stem pour tous les documents qui le contiennent.

        Arguments:
            stem str -- le mot.
        """
        pass

    @abstractmethod
    def getWeightsForQuery(self, query):
        """Retourne les poids des termes de la requête.

        Arguments:
            query str -- La Requête.
        """
        pass


class Sub_Weighter1(Weighter):
    """w(t,d) = tf(t,d).
       w(t,q) = 1 si t ∈ q, O sinon;

    Arguments:
        Weighter Weighter -- Herite de la class Weighter.
    """

    def __init__(self, index):
        super().__init__(index)

    def getWeightsForDoc(self, idDoc):
        return self.index.index[idDoc]

    def getWeightsForStem(self, stem):
        return {i: self.getWeightsForDoc(i)[stem] for i in self.index.index_inverse[stem]}

    def getWeightsForQuery(self, query):
        return {w: 1 for w in ps.getTextRepresentation(query)}


class Sub_Weighter2(Weighter):
    """w(t,d) = tf(t,d).
       w(t,q) = tf(t,q).

    Arguments:
        Weighter Weighter -- Herite de la class Weighter.
    """

    def __init__(self, index):
        super().__init__(index)

    def getWeightsForDoc(self, idDoc):
        return self.index.index[idDoc]

    def getWeightsForStem(self, stem):
        return {i: self.getWeightsForDoc(i)[stem] for i in self.index.index_inverse[stem]}

    def getWeightsForQuery(self, query):
        return Counter(ps.getTextRepresentation(query))


class Sub_Weighter3(Weighter):
    """w(t,d)=tf(t,d).
       w(t,q) = idf(t) si t ∈ q 0 sinon;

    Arguments:
        Weighter Weighter -- Herite de la class Weighter.
    """

    def __init__(self, index):
        super().__init__(index)

    def getWeightsForDoc(self, idDoc):
        return self.index.index[idDoc]

    def getWeightsForStem(self, stem):
        return {i: self.getWeightsForDoc(i)[stem] for i in self.index.index_inverse[stem]}

    def getWeightsForQuery(self, query):
        ps = PorterStemmer()
        N = len(self.index.index)

        def idf(stem):
            try:
                df = len(self.index.index_inverse[stem])
                return np.log((1+N)/(1+df))
            except KeyError:
                return np.log(1+N)

        return {w: idf(w) for w in ps.getTextRepresentation(query)}


class Sub_Weighter4(Weighter):
    """w(t,d)=1+log(tf(t,d)) si t ∈ d 0 sinon.
       w(t,q) = idf(t) si t ∈ q 0 sinon.

    Arguments:
        Weighter Weighter -- Herite de la class Weighter.
    """

    def __init__(self, index):
        super().__init__(index)

    def getWeightsForDoc(self, idDoc):
        return {w: 1+np.log(occor) for w, occor in self.index.index[idDoc].items()}

    def getWeightsForStem(self, stem):
        return {i: self.getWeightsForDoc(i)[stem] for i in self.index.index_inverse[stem]}

    def getWeightsForQuery(self, query):
        ps = PorterStemmer()
        N = len(self.index.index)

        def idf(stem):
            try:
                df = len(self.index.index_inverse[stem])
                return np.log((1+N)/(1+df))
            except KeyError:
                return np.log(1+N)

        return {w: idf(w) for w in ps.getTextRepresentation(query)}


class Sub_Weighter5(Weighter):
    """w(t,d)=(1+log(tf(t,d)))*idf(t) si t ∈ d 0 sinon. 
       w(t,q)=(1+log(tf(t,q)))*idf(t) si t ∈ q 0 sinon..

    Arguments:
        Weighter Weighter -- Herite de la class Weighter.
    """

    def __init__(self, index):
        super().__init__(index)

    def getWeightsForDoc(self, idDoc):
        return {w: 1+np.log(occor)*self._idf(w) for w, occor in self.index.index[idDoc].items()}

    def getWeightsForStem(self, stem):
        return self.index.index_inverse[stem]

    def getWeightsForQuery(self, query):
        ps = PorterStemmer()
        c = dict(Counter(ps.getTextRepresentation(query)))
        return {w: 1+np.log(occur)*self._idf(w) for w, occur in c.items()}

    def _idf(self, stem):
        N = len(self.index.index)
        try:
            df = len(self.index.index_inverse[stem])
            return np.log((1+N)/(1+df))
        except KeyError:
            return np.log(1+N)
