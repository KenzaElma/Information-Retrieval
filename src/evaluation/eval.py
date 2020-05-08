from abc import ABC, abstractmethod


class EvalMesure(ABC):

    @abstractmethod
    def evalQuery(self, liste, query):
        pass


class Precision(EvalMesure):
    def evalQuery(self, liste, query, rang=20):
        releventDocumentReturned = set(liste[:rang]) & set(
            query.listIdsOfRelevantDocuments)
        try:
            return len(releventDocumentReturned)/rang
        except ZeroDivisionError:
            return 0


class Rappel(EvalMesure):
    def evalQuery(self, liste, query, rang=20):
        releventDocumentReturned = set(liste[:rang]) & set(
            query.listIdsOfRelevantDocuments)
        try:
            return len(releventDocumentReturned)/len(query.listIdsOfRelevantDocuments)
        except ZeroDivisionError:
            return 1


class Falpha(EvalMesure):
    def __init__(self, alpha=1):
        self.alpha = alpha

    def evalQuery(self, liste, query, rang=20):
        p = Precision().evalQuery(liste, query, rang)
        r = Rappel().evalQuery(liste, query, rang)
        try:
            return (1+self.alpha**2) * ((p*r)/(self.alpha**2*p + r))
        except ZeroDivisionError:
            return 0

def evaluate(queries, model, rang=20):
    def f(query):
        query = queries[query]
        liste = list(model.getRanking(query.text).keys())
        p = Precision().evalQuery(liste, query, rang)
        r = Rappel().evalQuery(liste, query, rang)
        f = Falpha().evalQuery(liste, query, rang)
        return (p, r, f)

    return list(map(f, queries))
