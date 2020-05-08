#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

class Query:
    
    def __init__(self,id,text):
        self.id=id
        self.text = text
        self.listIdsOfRelevantDocuments = []
    
    def __repr__(self):
        return "Id: "+str(self.id)+"\nText: "+self.text




class QueryParser:
    def __init__(self,queries_file_path,results_file_path):
        self.queries_file_path = queries_file_path
        self.results_file_path = results_file_path

    def _build_Cacm_queries_list(self):
        
        
        storing = False
        result = list()
        ids = list()
        temp = ""
        with open(self.queries_file_path) as queries_file:
            for line in queries_file :
                if line[:2] == ".I" :
                    id = int(re.findall(r'\d+',line)[0])
                    ids.append(id)
                if line[:2] == ".W":
                    storing = True
                    if temp != "":
                        result.append(temp)
                        temp = ""
                elif line[:1] == ".":
                    storing = False
                elif storing:
                    temp += line
        
        result.append(temp)

        return { ids[i] : Query(ids[i],result[i] ) for  i in range(len(ids)) } 
    
    def cacm_queries_parser(self):
        queries = self._build_Cacm_queries_list()
        with open(self.results_file_path) as results_file:
            for result in results_file:
                splitted = result.split()
                id = int(splitted[0])
                to = int(splitted[1])
                queries[id].listIdsOfRelevantDocuments.append(to) 
        return queries