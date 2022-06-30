import coreferee, spacy
nlp = spacy.load("en_core_web_trf")
nlp.add_pipe('coreferee')


doc = nlp("Marie Curie received the Nobel Prize in Physics in 1903. She became the first woman to win the prize and the first person — man or woman — to win the award twice.")
coref = []

if len(doc._.coref_chains) > 0:
    for chain in doc._.coref_chains:
        for x in range(len(chain)-1):
            print(doc[chain[x].token_indexes[0]].idx)
           # mention = {'from_index': doc[cluster[x+1]].idx, 'to_index': doc[cluster[0]].idx}
           # coref.append(mention)