#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 00:58:16 2020

@author: YACINE 
"""


import re
import subprocess

from collections import Counter
import pickle as pkl
import numpy as np
from .TextRepresenter import PorterStemmer
import re


class Document:
    """ 
        Un Document
    """

    def __init__(self, id, title=""):
        self.id = int(id)
        self.title = title
        self.date = None
        self.auteurs = []
        self.expressions = []
        self.texte = ""
        self.liens = []

    def addAuteur(self, nom, prenom):
        """Ajout d'un auteur.

        Arguments:
            nom str -- le nom de l'auteur
            prenom str -- le prenom de l'auteur
        """
        if (prenom == ''):
            self.auteurs.append([nom])
        else:
            self.auteurs.append([nom, prenom])

    def addLien(self, lien):
        """Ajout d'un lien

        Arguments:
            lien str -- lien vers l'article de document
        """
        try:
            if(self.id!= int(lien)):
                self.liens.append(int(lien))
                self.liens = list(set(self.liens))
        except ValueError:
            return 

    def __repr__(self):
        """La representation d'un document sous forme d'une string
        """

        return f'ID: {self.id}\nTitle: {self.title}\nDate:{self.date if self.date else "YYYY-MM-DD"}\nAuthors: {self.auteurs}\nText:{self.texte[:20] if len(self.texte) >0 else ""}\nLiens: {self.liens}'


class Parser:
    def __init__(self, path):
        """Un Parser d'une Collection

        Arguments:
            path str -- Le chemin vers le fichier contienant les documents.
        """
        self.path = path

    def cacm_cisi_parser(self):
        """CACM & CISI Parser

        Returns:
            dict(int,Document) -- Retourne un dictionnaire de ID, Document 
        """
        file = open(self.path, 'r')
        res = {}
        balise = None
        doc = None
        newBalise = False
        while True:
            line = file.readline()
            if(not line):
                break
            words = line.split()
            if(len(words) > 0):
                pattern = re.compile("^\.[ITBAWKX]")

                if(pattern.match(words[0])):
                    newBalise = True
                    balise = words[0][1]
                else:
                    newBalise = False

                if(balise == "I"):
                    doc = Document(words[1])

                if(balise == 'T' and not newBalise):
                    doc.title += line

                if(balise == 'B'):
                    date = re.compile(r'(\d{4})').search(line)

                    if(date != None):
                        doc.date = int(date.group(0))
                if(balise == 'A'):
                    auteur = re.compile(
                        r'([a-zA-Z \'.-]+)\s*,\s*([a-zA-Z \'.-]+)\s*\n').search(line)
                    if(auteur != None):
                        nom = auteur.group(1)
                        prenom = auteur.group(2)
                    else:
                        nom = line[:-1]
                        prenom = ""
                    doc.addAuteur(nom, prenom)
                if(balise == 'W' and not newBalise):
                    doc.texte += line
                if(balise == 'X'):
                    doc.addLien(words[0])

            
            res[doc.id] = doc
            
        file.close()

        return res

    def build_collection_from_Json(self, id, content):
        """Contruit une collection à partir d'un Json

        Arguments:
            id str -- l'identifaint a choisir dans le JSON.
            content str -- le contenu a prendre en consedration dans le JSON.

        Returns:
            list -- return list des Document.
        """
        import pandas as pd
        path = self.path[0]
        return pd.read_json(path)[[id, content]].apply(lambda w: Document(w[0], w[1]), axis=1).values

    def buildDocumentCollectionRegex(self):
        s_out = bytes()
        cmd = "awk '/^.I/,/^.[BAKWX].*/ { print }' "+self.path +\
            " |sed -r 's/^.I ([0-9]+)/{\\1}/'|sed -r 's/^.W|^.T//'|sed '/^$/d'| sed '/\.B/d' | tr '\n' ' ' | tr '{' '\n'"

        out, _ = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()

        return list(map(lambda l: Document(l[0], l[1]), filter(lambda li: len(li) == 2, map(lambda line: line.split('}'), out.decode("utf-8").splitlines()))))
