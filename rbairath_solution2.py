from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from argparse import ArgumentParser
import sys
import os.path


class Embedding:
    def __init__(self, embedding):
        """
        Initialize the instance with model & embedding
        """
        print('Initializing embedding from file ' + embedding + '. Please wait, this may take a while.')
        try:
            is_binary = os.path.splitext(embedding)[1] != '.txt'
            self.model = KeyedVectors.load_word2vec_format(embedding, binary=is_binary)
        except Exception as ex:
            print(ex)
            print('Unable to read file ' + embedding + '. Please check the error above', file=sys.stderr)
            sys.exit(1)
        print('Embedding loaded successfully!')


class Assignment5:
    def __init__(self, w2v_filename, glove_filename):
        """
        Initialize class with embeddings
        :param w2v_filename: Name of the w2v format embedding
        :param glove_filename: Name of the glove format embedding
        """
        
        if w2v_filename:
            # Initialize Word2Vec embedding (if provided)
            self.w2v_embedding = Embedding(w2v_filename)
        
        if glove_filename:
            """
            Convert GloVe embedding to Word2Vec format. The only difference between
            the two formats is that the first line of Word2Vec format contains
            number of lines and number of dimensions in file
            """
            parts = os.path.splitext(glove_filename)
            convert_filename = parts[0] + '.w2v' + parts[1]
            glove2word2vec(glove_filename, convert_filename)
            
            # Initialize GloVe embedding (if provided)
            self.glove_embedding = Embedding(convert_filename)
        
        # Initialize input file variable
        self.test_words = ['accept', 'combine', 'increase', 'give', 'open', 'scatter']
    
    def execute(self):
        """
        Executes the similar words test
        """
        # Execute test on Word2Vec embedding
        if hasattr(self, 'w2v_embedding'):
            for word in self.test_words:
                similar_words = self.w2v_embedding.model.similar_by_word(word=word, topn=10)
                similar_words = [word[0] for word in similar_words]
                print('Words similar to %s as per Word2Vec embeddings are:' % word)
                print(similar_words)

        # Execute test on GloVe embedding
        if hasattr(self, 'glove_embedding'):
            for word in self.test_words:
                similar_words = self.glove_embedding.model.similar_by_word(word=word, topn=10)
                similar_words = [word[0] for word in similar_words]
                print('Words similar to %s as per GloVe embeddings are:' % word)
                print(similar_words)


# Initialize argument parser
parser = ArgumentParser(description='Executes analogy test on Word2Vec and/or GloVe embeddings.')
# Add argument for word2vec format filename
parser.add_argument('--w2v', action='store', dest='w2v', default=None,
                    help='Filename of word2vec format pre-trained embedding. Extension must be .txt for text format')
# Add argument for GloVe format filename
parser.add_argument('--glove', action='store', dest='glove', default=None,
                    help='Filename of glove format pre-trained embedding. Extension must be .txt for text format')

# Read arguments
args = parser.parse_args()
# If none of the argument is provided, show error and help
if not args.w2v and not args.glove:
    print('Provide at least one embedding file')
    parser.print_help()
    sys.exit(0)

# Execute the program
assignment5 = Assignment5(args.w2v, args.glove)
assignment5.execute()