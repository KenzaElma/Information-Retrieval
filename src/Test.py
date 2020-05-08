#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np


import sys
sys.path.append('./')


from indexation.parser import Parser
from indexation.Indexer import IndexerSimple
from ordonnancement.Weighter import Sub_Weighter1
from ordonnancement.ModelRI import Okapi


path = "./data/cacm+cisi/cacm/cacm.txt"
collectionName = "cacm"
pathSaveIndexes = "./src/indexation/index"

start = time.time()
parser = Parser(path)
collection = parser.cacm_cisi_parser()


indexsimple = IndexerSimple(collection , pathSaveIndexes, collectionName)
end = time.time()

print(end - start)
query = "Computer"
weighter = Sub_Weighter1(indexsimple)
start = time.time()
modelRi = Okapi(indexsimple)
ranking = modelRi.getRanking(query)

print(list(ranking.keys()))


print(collection[2973])

end = time.time()
print(end - start)

