import spacy

import crosslingual_coreference

text = (
    "Do not forget about Momofuku Ando! He created instant noodles in Osaka. At"
    " that location, Nissin was founded. Many students survived by eating these"
    " noodles, but they don't even know him."
)


nlp = spacy.load("en_core_web_trf")
nlp.add_pipe(
    "xx_coref", config={"chunk_size": 2500, "chunk_overlap": 2, "device": 0}
)

doc = nlp(text)
print(doc._.coref_clusters)
# Output
#
# [[[4, 5], [7, 7], [27, 27], [36, 36]],
# [[12, 12], [15, 16]],
# [[9, 10], [27, 28]],
# [[22, 23], [31, 31]]]
print(doc._.resolved_text)
# Output
#
# Do not forget about Momofuku Ando!
# Momofuku Ando created instant noodles in Osaka.
# At Osaka, Nissin was founded.
# Many students survived by eating instant noodles,
# but Many students don't even know Momofuku Ando.
print(doc._.cluster_heads)
# Output
# 
# {Momofuku Ando: [5, 6], 
# instant noodles: [11, 12], 
# Osaka: [14, 14], 
# Nissin: [21, 21], 
# Many students: [26, 27]} 