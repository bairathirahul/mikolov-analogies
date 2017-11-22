# ITCS 5111 - ASSIGNMENT 5

The programs rbairath.py and rbairath_solution2.py are the solutions to the Assignment 5.

## rbairath.py:
This program processes Mikolov Analogies and report the accuracy of the given model. The program works only on pre-trained embeddings of either Word2Vec format or GloVe format. The embeddings used by me can be downloaded from the following URLs:
Google News Word2Vec binary format: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
Glove Common Crawls: http://nlp.stanford.edu/data/glove.840B.300d.zip
Both of the embeddings are case-sensitive. The program only works for the case-sensitive embeddings.

### Requirements:
* Python 3.5 or later
* Gensim package. For installation instructions, refer to: https://radimrehurek.com/gensim/install.html

### Execution:
* Before execution, make sure that you have downloaded and extracted pre-trained embeddings. To execute, run the following command:
`python3 rbairath.py --w2v <Word2Vec format embedding filename> --glove <Glove format embedding filename>`
* At least one of the two embedding file must be provided. 
* The program will take some time to initialize depending on the size of the given embedding file. 
* After initialization, it will ask for the input questions filename, which must be a text file. 
* It will list all the groups that are found in filename and will ask to enter the group index. The group at the given index will be processed and the accuracy will be reported.

##### Note:
Please be sure that you have extracted the embeddings file if it's archived. Also, if the file format is binary, then the extension should be ".bin". All other extensions are considered to be text file.

## rbairath_solution2.py
This program generates 10 most similar words of the given input words. This program also works only on pre-trained embeddings of either Word2Vec or GloVe format.

### Requirements:
* Python 3.5 or later
* Gensim package. For installation instructions, refer to: https://radimrehurek.com/gensim/install.html

### Execution:
* Before execution, make sure that you have downloaded and extracted pre-trained embeddings. To execute, run the following command:
`python3 rbairath_solution2.py --w2v <Word2Vec format embedding filename> --glove <Glove format embedding filename> --words word1 word2 word3` 
* At least one of the two embedding file must be provided. 
* The program will take some time to initialize depending on the size of the given embedding file. 
* As the words parameter, you can provide any number of words
* For each provided word, the program will display 10 most similar words as calculated by the provided embeddings.