'''
@author madhumita
'''

class Annotation(object):
    
    def __init__(self, begin, end, pipeline = 'clamp'):
        '''
        @param start: Start offset of annotation
        @param end: End offset of annotation
        @param pipeline (clamp/ctakes): string with the name of pipeline the annotations have been generated from
        '''
        self.begin = begin
        self.end = end
        self.pipeline = pipeline

class Sentence(Annotation):
    
    def __init__(self, begin, end, section, pipeline = 'clamp'):
        '''
        @param section: Section header for the given sentence
        '''
        
        super(Sentence, self).__init__(begin=begin, end=end, pipeline = pipeline)
        
        self.section = section
        
class Token(Annotation):
    
    def __init__(self, begin, end, pos, pipeline = 'clamp'):
        '''
        @param pos: POS tag of the token
        '''
        super(Token, self).__init__(begin=begin, end=end, pipeline = pipeline)
        
        self.pos = pos
        

class Concept(Annotation):
    
    def __init__(self, begin, end, sem_type, cui, assertion, mention, pipeline = 'clamp'):
        '''
        @param sem_type: UMLS semantic type of the given concept
        @param cui: UMLS Concept Unique Identifier (CUI) for the concept
        @param assertion: whether the concept is negated or not
        @param mention: the mention of the concept in the text
        '''
        
        super(Concept, self).__init__(begin=begin, end=end, pipeline = pipeline)
        
        self.sem_type = sem_type
        self.cui = cui
        self.assertion = assertion
        self.mention = mention
    
class TextView(object):
    
    '''
    Provides the objects for the text annotations for the complete view of a text
    '''
    
    def __init__(self):
        self.sentences = list() #set of Sentence objects
        self.tokens = list() #set of Token objects
        self.concepts = list() #set of Concept objects
        
    def read_annotations(self, dir_name, file, read_sent = True, read_tokens = True, read_concepts = True, pipeline = 'clamp'):
        
        if pipeline == 'clamp':
            for line in open(dir_name+file):
                line = line.split('\t')
                if read_sent and line[0].lower() == 'sentence':
                    self.sentences.append( Sentence(int(line[1]), int(line[2]), line[3].split('=')[1]) )
                if read_tokens and line[0].lower() == 'token':
                    self.tokens.append( Token(int(line[1]), int(line[2]), line[3].split('=')[1]) )
                if read_concepts and line[0].lower() == 'namedentity':
                    assertion = cui = s_type = ne = None
                    for term in line:
                        if 'semantic' in term:
                            s_type = term.split('=')[1]
                        elif 'assertion' in term:
                            assertion = term.split('=')[1]
                        elif 'cui' in term:
                            cui = term.split('=')[1]
                        elif 'ne' in term:
                            ne = term.split('=')[1]
                    
                    if not cui:
                        print(ne)
                        
                    self.concepts.append( Concept(int(line[1]), int(line[2]), 
                                               sem_type =  s_type, 
                                               assertion = assertion, 
                                               cui = cui, 
                                               mention = ne 
                                               )
                                      )
    def get_covered_tokens(self, cur_sent):
        '''
        Return tokens covered by a sentence annotation. Assumes that sentence and token annotations are already present.
        @param cur_sent: a sentence annotation to find the covered tokens under.
        @return cov_tokens: list of tokens covered under the sentence.
        '''
        cov_tokens = list()
        for cur_token in self.tokens:
            if cur_token.begin >= cur_sent.begin and cur_token.end <= cur_sent.end:
                cov_tokens.append(cur_token)
            elif cur_token.begin >= cur_sent.end:
                break
        return cov_tokens