from gensim.models.keyedvectors import KeyedVectors
import sys


class Embedding:
    def __init__(self, embedding, isbinary):
        """
        Initialize the instance with model & embedding
        """
        print('Initializing embedding from file ' + embedding + '. Please wait, this may take a while.')
        try:
            self.model = KeyedVectors.load_word2vec_format(embedding, binary=isbinary)
        except:
            print('Unable to read file ' + embedding + '. Please verify that the file exists', file=sys.stderr)
            sys.exit(1)
        print('Embedding loaded successfully!')
    
    def test(self, words):
        """
        Executes the analogy test on given set of 4 words
        :param words: list of 4 words
        :return: accuracy of the model
        """
        try:
            # Get 10 word vectors which are similar to the analogy
            similar_words = self.model.most_similar(positive=[words[1], words[2]], negative=[words[0]])
            # Extract word from the word vectors
            similar_words = [vector[0] for vector in similar_words]
            # Check if fourth word belong to the extracted word list
            if words[3] in similar_words :
                return True
            return False
        except Exception as ex:
            print(ex)
            # Word not found, return False
            return False

class Analogies:
    """
    Iterable instance of Analogies
    """
    
    def __init__(self, filename):
        """
        Read the analogies file. The format of file is should be as follows:
        1. There should be a line starting with ': ' (colon space) and group name
        2. All subsequent lines must contain 4 case-sensitive words separated by space which belong
        to the group specified with the previous line starting with ': '
        
        :param filename: Text file with analogies
        """
        self.file = open(filename, 'r')
    
    def select_group(self, group):
        """
        Select analogy group from the given analogy file
        :param group: Group name to select
        :return: Nothing
        """
        self.file.seek(0)
        
        group = ': ' + group
        for line in self.file:
            line = line.strip()
            if line == group:
                return
        
        raise Exception('Group ' + group + ' not found.')
    
    def __iter__(self):
        """
        Iterator instance
        :return: Iterator instance
        """
        return self
    
    def __next__(self):
        """
        Iterate over analogies
        :return: Words array
        """
        line = next(self.file)
        if line[0] == ':':
            raise StopIteration
        
        words = line.strip().split()
        return words
    
class Assignment5 :
    # Initialize Word2Vec model with Google News pre-trained vectors
    w2v_embedding = Embedding('embeddings/GoogleNews-vectors-negative300.bin', True)
    
    # Initialize Word2Vec model with Glove pre-trained vectors
    glove_embedding = Embedding('embeddings/glove.6B.300d.w2v.txt', False)
    
    def __init__(self, filename):
        """
        Initialize class with analogies file
        :param filename: Analogies file
        """
        try:
            # Initialize Mikolov analogies
            self.analogies = Analogies('test/word-test.v1.txt')
        except:
            print('Unable to read file ' + filename + '. Please verify that the file exists', file=sys.stderr)
            sys.exit(1)
            
    def execute(self, embedding, group):
        try:
            # Select Group
            self.analogies.select_group(group)
        except:
            print('Group ' + group + ' is not found')
            return 0
        
        total = 0
        correct = 0
        for words in self.analogies:
            print(words)
            total += 1
            if embedding.test(words):
                correct +=1
                
        return correct * 100 / total
        
    def execute_word2vec(self, group):
        return self.execute(self.w2v_embedding, group)

    def execute_glove(self, group):
        return self.execute(self.glove_embedding, group)


"""
Execute Solution 1
Perform embeddings test on Mikolov Analogies
"""
assignment5 = Assignment5('test/word-test.v1.txt')

#accuracy = assignment5.execute_word2vec('capital-world')
#print('Accuracy of word2vec GoogleNews embeddings for "capital-world" analogies is %.2f%%' % accuracy)

accuracy = assignment5.execute_glove('capital-world')
print('Accuracy of GloVe embeddings for "capital-world" analogies is %.2f%%' % accuracy)

sys.exit()

accuracy = assignment5.execute_word2vec('currency')
print('Accuracy of word2vec GoogleNews embeddings for "capital-world" analogies is %.2f%%' % accuracy)

accuracy = assignment5.execute_glove('currency')
print('Accuracy of GloVe embeddings for "capital-world" analogies is %.2f%%' % accuracy)

accuracy = assignment5.execute_word2vec('city-in-state')
print('Accuracy of word2vec GoogleNews embeddings for "capital-world" analogies is %.2f%%' % accuracy)

accuracy = assignment5.execute_glove('city-in-state')
print('Accuracy of GloVe embeddings for "capital-world" analogies is %.2f%%' % accuracy)

accuracy = assignment5.execute_word2vec('family')
print('Accuracy of word2vec GoogleNews embeddings for "capital-world" analogies is %.2f%%' % accuracy)

accuracy = assignment5.execute_glove('family')
print('Accuracy of GloVe embeddings for "capital-world" analogies is %.2f%%' % accuracy)

accuracy = assignment5.execute_word2vec('gram1-adjective-to-adverb')
print('Accuracy of word2vec GoogleNews embeddings for "capital-world" analogies is %.2f%%' % accuracy)

accuracy = assignment5.execute_glove('gram1-adjective-to-adverb')
print('Accuracy of GloVe embeddings for "capital-world" analogies is %.2f%%' % accuracy)

accuracy = assignment5.execute_word2vec('gram2-opposite')
print('Accuracy of word2vec GoogleNews embeddings for "capital-world" analogies is %.2f%%' % accuracy)

accuracy = assignment5.execute_glove('gram2-opposite')
print('Accuracy of GloVe embeddings for "capital-world" analogies is %.2f%%' % accuracy)

accuracy = assignment5.execute_word2vec('gram3-comparative')
print('Accuracy of word2vec GoogleNews embeddings for "capital-world" analogies is %.2f%%' % accuracy)

accuracy = assignment5.execute_glove('gram3-comparative')
print('Accuracy of GloVe embeddings for "capital-world" analogies is %.2f%%' % accuracy)

accuracy = assignment5.execute_word2vec('gram6-nationality-adjective')
print('Accuracy of word2vec GoogleNews embeddings for "capital-world" analogies is %.2f%%' % accuracy)

accuracy = assignment5.execute_glove('gram6-nationality-adjective')
print('Accuracy of GloVe embeddings for "capital-world" analogies is %.2f%%' % accuracy)


"""
Execute Solution 3
Perform embeddings test on Custom Analogies
"""
assignment5 = Assignment5('test/word-test.v2.txt')
accuracy = assignment5.execute_word2vec('part-to-whole')
print('Accuracy of word2vec GoogleNews embeddings for "part-to-whole" analogies is %.2f%%' % accuracy)

accuracy = assignment5.execute_glove('part-to-whole')
print('Accuracy of word2vec GoogleNews embeddings for "month-season" analogies is %.2f%%' % accuracy)

accuracy = assignment5.execute_word2vec('month-season')
print('Accuracy of word2vec GoogleNews embeddings for "part-to-whole" analogies is %.2f%%' % accuracy)

accuracy = assignment5.execute_glove('month-season')
print('Accuracy of word2vec GoogleNews embeddings for "month-season" analogies is %.2f%%' % accuracy)