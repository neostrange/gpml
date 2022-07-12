import json
from tokenize import String
import requests
#from allennlp.predictors.predictor import Predictor
#from allennlp_models import pretrained
#import allennlp_models.tagging
from spacy import Language
import GPUtil
import spacy
from spacy.matcher import Matcher, DependencyMatcher
from spacy.tokens import Doc, Token, Span
from spacy.language import Language
import textwrap
from transformers import logging

#from gpml.util.RestCaller import callAllenNlpApi

logging.set_verbosity_error()

from py2neo import Graph
from py2neo import *

#graph = Graph("bolt://10.1.48.224:7687", auth=("neo4j", "neo123"))

#try:
#    dd.set_extension("SRL", default=dict())
#except:
#    pass

try:
    Token.set_extension("SRL", default=dict())
except:
    pass

#this is specific to spacy v3 
#configuring and importing spacy custom plugin for srl
if spacy.Language.has_factory("srl") is False:
    @spacy.Language.factory("srl", 
                    assigns=["token._.SRL"],
                    requires=["token.tag"],
                    retokenizes = False)
    def srl(nlp, name):
        return SemanticRoleLabel()

class SemanticRoleLabel:

    list_exceptions = []

    def __init__(self, ):
        self.apiName = "semantic-role-labeling"

    def __call__(self, doc):
        res_srl = self.srl_doc(ss = doc.text)
        for tok in doc:
            if tok.pos_ in ["VERB", "AUX"]:
                ii = tok.i
                try:
                    #search for the frame that is centered on this verb
                    frame_verb = [el for el in res_srl["verbs"] if el["tags"][ii] == "B-V"][0]
                    dict_args = self.post_process_verbframe(frame_verb) 

                    #skip cases of {'V': [8]}  
                    if len(list(dict_args.keys())) > 1:
                        tok._.SRL = dict_args
                except Exception as e:
                    self.list_exceptions.append("EXCEPTION:" + doc.text + "|||" + tok.text)
        return doc

    def srl_doc(self, ss):
        res_srl = self.callAllenNlpApi(self.apiName, ss)
        #res_srl.replace('"',"'")
        #res_srl= json.loads(res_srl)
        return res_srl

    def post_process_verbframe(self,frame_verb):
        tags = frame_verb["tags"]
        dict_args = {}
        current_role = None

        for jj in range(len(tags)):
            if current_role is None:
                if tags[jj] == "O":
                    pass
                else:
                    #begin a tag here
                    if tags[jj][0] == "B":
                        #may have one or multiple dashes (B-ARG1, B-ARGM-DIR) 
                        key = tags[jj][ tags[jj].find("-")+1:]
                        current_role = {key: [jj]}
                    else:
                        raise Exception("cannot be {} after O".format(tags[jj])) 
            else:
                if tags[jj] == "O":
                    #a role is ended
                    dict_args.update(current_role)
                    current_role = None
                elif tags[jj][0] == "I":
                    #continue the current role
                    current_role[list(current_role.keys())[0]].append(jj)
                elif tags[jj][0] == "B":
                    #a new tag follows immediately the previous tag (without any O in-between)
                    dict_args.update(current_role)
                    key = tags[jj][ tags[jj].find("-")+1:]
                    current_role = {key: [jj]}

        return dict_args

    def callAllenNlpApi(self, apiName, string):
        URL = "https://demo.allennlp.org/api/"+apiName+"/predict"

        PARAMS = {"Content-Type": "application/json"}

        payload = {"sentence":string}
        
        r = requests.post(URL, headers=PARAMS, data=json.dumps(payload))

        print(r.text)

        return json.loads(r.text)
# end of class: SemanticRoleLabel
