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
            is_binary = os.path.splitext(embedding)[1] == '.bin'
            self.model = KeyedVectors.load_word2vec_format(embedding, binary=is_binary)
        except Exception as ex:
            print(ex)
            print('Unable to read file ' + embedding + '. Please check the error above', file=sys.stderr)
            sys.exit(1)
        print('Embedding loaded successfully!')
    
    def test(self, words):
        """
        Executes the analogy test on given set of 4 words (case-sensitive).
        
        :param words: list of 4 words
        :return: accuracy of the model
        """
        try:
            # Skip input with invalid format
            if len(words) != 4:
                return False
            
            # Get the most similar word
            similar_words = self.model.most_similar(positive=[words[1], words[2]], negative=[words[0]], topn=1)
            
            # Extract word from the word vectors
            similar_words = [vector[0] for vector in similar_words]
            
            # Check if fourth word belong to the extracted word list
            if words[3] in similar_words:
                return True
            
            # Return false for incorrect prediction
            return False
        except Exception as ex:
            print(ex)
            # Word not found, return False
            return False
    
    
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
        self.input_file = None
            
    def execute(self, embedding):
        """
        Execute the embedding test
        
        :param embedding: Instance of Embedding class to test with
        :return: Accuracy percentage
        """
        total = 0       # total number of lines
        correct = 0     # number of correct predictions
        
        # Loop on input file, positioned already to the group
        for line in self.input_file:
            line = line.strip()
            
            # Skip empty line
            if not line:
                continue
            
            # If next group is found, break
            if line.startswith(':'):
                break
            
            # Split line to words
            words = line.split()
            total += 1
            
            # Test embedding
            if embedding.test(words):
                correct += 1
                
        if total > 0:
            return correct * 100 / total
        else:
            return 0
        
    def execute_word2vec(self, group):
        """
        Execute the test on Word2Vec embedding and print accuracy percentage
        :param group: Group name
        :return: Accuracy
        """
        if hasattr(self, 'w2v_embedding'):
            # Execute test with Word2Vec embedding
            accuracy = self.execute(self.w2v_embedding)
            print('Accuracy of Word2Vec embedding for %s group is %.2f%%' % (group, accuracy))
            return accuracy

    def execute_glove(self, group):
        """
        Execute the test on GloVe embedding and print accuracy percentage
        :param group: Group name
        :return: Accuracy
        """
        if hasattr(self, 'glove_embedding'):
            accuracy = self.execute(self.glove_embedding)
            print('Accuracy of GloVe embedding for %s group is %.2f%%' % (group, accuracy))
            return accuracy
    
    def prompt(self):
        """
        Prompt for input filename and displays the groups found in the input filename
        Once, the user enters the group index, execute accuracy calculation method
        :return:
        """
        while True:
            try:
                # Prompt user for input filename of analogies
                input_filename = input('Enter input filename (Press Ctrl+C to exit): ')
            except KeyboardInterrupt:
                print('\nCleaning up! please wait..')
                sys.exit(0)
                
            try:
                # Try to open input file
                self.input_file = open(input_filename, 'r')
            except IOError as ex:
                print(ex)
                print('Unable to read file ' + input_filename + '. Please verify that the file exists', file=sys.stderr)
                continue
                
            # Read groups in the input file
            group_names = list()    # Group names
            group_pos = dict()      # Group name with location in file
            while True:
                line = self.input_file.readline()
                if not line:
                    break;
                elif line.startswith(':'):
                    group_name = line[2:].strip()
                    group_names.append(group_name)
                    group_pos[group_name] = self.input_file.tell()
                    
            # If no group found, ask for another file
            if len(group_names) == 0:
                print('No groups found in the given file. Please check and try again!')
                continue

            # Prompt user for group selection
            while True:
                # Display list of groups in file
                os.system('cls' if os.name == 'nt' else 'clear')
                print('Following groups are found in the file.')
                for index, group_name in enumerate(group_names):
                    print('%d. %s' % (index + 1, group_name))

                break_outer = False
                    
                while True:
                    # Prompt for group index
                    index = input('Enter the group index to test for accuracy, or 0 to exit: ')
                    try:
                        # Non-numeric input provided
                        index = int(index)
                    except ValueError:
                        print('Invalid index, please try again')
                        continue
                        
                    if index == 0:
                        # 0 entered, break out and prompt for new input file
                        break_outer = True
                        break
                    elif index > len(group_names):
                        # Index out of range, try again
                        print('Invalid index, please try again')
                        continue
                        
                    # Set file pointer to the group's beginning line
                    self.input_file.seek(group_pos[group_names[index - 1]])
                    # Execute analogy tests
                    self.execute_word2vec(group_names[index - 1])
                    self.execute_glove(group_names[index - 1])
                    input('Press enter key to continue..')
                    break
                
                # Break out and prompt for new input file
                if break_outer:
                    self.input_file.close()
                    break


# Initialize argument parser
parser = ArgumentParser(description='Executes analogy test on Word2Vec and/or GloVe embeddings.')
# Add argument for word2vec format filename
parser.add_argument('--w2v', action='store', dest='w2v', default=None, help='Filename of word2vec format pre-trained embedding. Extension must be .txt for text format')
# Add argument for GloVe format filename
parser.add_argument('--glove', action='store', dest='glove', default=None, help='Filename of glove format pre-trained embedding. Extension must be .txt for text format')

# Read arguments
args = parser.parse_args()
# If none of the argument is provided, show error and help
if not args.w2v and not args.glove:
    print('Provide at least one embedding file')
    parser.print_help()
    sys.exit(0)
    
# Execute the program
assignment5 = Assignment5(args.w2v, args.glove)
assignment5.prompt()