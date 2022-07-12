import spacy
import sys
#import neuralcoref
import coreferee
from gpml.util.SemanticRoleLabel2 import SemanticRoleLabel
from spacy.tokens import Doc, Token, Span
from gpml.util.RestCaller import callAllenNlpApi
from gpml.ch12.text_processors import TextProcessor
from gpml.util.graphdb_base import GraphDBBase






class GraphBasedNLP(GraphDBBase):

    def __init__(self, argv):
        super().__init__(command=__file__, argv=argv)
        spacy.prefer_gpu()

        self.nlp = spacy.load('en_core_web_trf')
        #coref = neuralcoref.NeuralCoref(self.nlp.vocab)
        #self.nlp.add_pipe(coref, name='neuralcoref')
        self.nlp.add_pipe('coreferee')

        if "srl" in self.nlp.pipe_names:
            self.nlp.remove_pipe("srl")
            _ = self.nlp.add_pipe("srl")

        self.nlp.add_pipe("srl")


        print(self.nlp.pipe_names)

        self.__text_processor = TextProcessor(self.nlp, self._driver)
        self.create_constraints()

    def create_constraints(self):
        self.execute_without_exception("CREATE CONSTRAINT ON (u:Tag) ASSERT (u.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (i:TagOccurrence) ASSERT (i.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (t:Sentence) ASSERT (t.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (l:AnnotatedText) ASSERT (l.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (l:NamedEntity) ASSERT (l.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (l:Entity) ASSERT (l.type, l.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (l:Evidence) ASSERT (l.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (l:Relationship) ASSERT (l.id) IS NODE KEY")

    def tokenize_and_store(self, text, text_id, storeTag):
        docs = self.nlp.pipe([text])
        for doc in docs:
            annotated_text = self.__text_processor.create_annotated_text(doc, text_id)
            spans = self.__text_processor.process_sentences(annotated_text, doc, storeTag, text_id)
            nes = self.__text_processor.process_entities(spans, text_id)
            coref = self.__text_processor.process_coreference(doc, text_id)
            self.__text_processor.build_entities_inferred_graph(text_id)
            self.__text_processor.apply_pipeline_1(doc)
            rules = [
                {
                    'type': 'RECEIVE_PRIZE',
                    'verbs': ['receive'],
                    'subjectTypes': ['PERSON', 'NP'],
                    'objectTypes': ['WORK_OF_ART']
                }
            ]
            self.__text_processor.extract_relationships(text_id, rules)
            self.__text_processor.build_relationships_inferred_graph(text_id)


if __name__ == '__main__':
    basic_nlp = GraphBasedNLP(sys.argv[1:])
    basic_nlp.tokenize_and_store(
        """Apple Computer today introduced the new MacBook line, which includes the Macbook and Macbook Pro. It is the successor to the iBook line and contains Intel Core Duo processors and a host of features, and starting at a price of $1,099. The Macbook features a 13.3" widescreen display, while the Pro can be purchased with either 15" or 17" displays. It comes in two colors: Black (2 GHz model only) and White (1.83 and 2 GHz models). This release leaves only one PowerPC processor computer that has not made the transition to Intel chips, the PowerMac G5.""",
        3,
        False)

    basic_nlp.close()
