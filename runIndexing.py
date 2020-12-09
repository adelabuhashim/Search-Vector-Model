import math
import glob
import os.path
import re

def tokenize(doc):
    ''' splits a given string into tokens around all non-alphabetical characters

    args:
        doc: a string representing an entire document (can contain linebreaks)
    returns:
        a list of alphabetical tokens (all non-empty)
    '''
    # TODO implement
    # replace any non-alphabetical characters with white space

    content = re.sub("[^a-zA-Z]+", " ", doc)
    # split content into tokens
    token_list = content.split()
    # TODO in a comment, give 5 examples (made up by yourself) of different
    #      types of character sequences that will be handled inappropriately by
    #      this simple tokenization algorithm
    # example1: 'ab45c' --> 'ab', 'c'
    # example5: 'I will do it' --> 'I', 'will', 'do', 'it'
    # example1: 'He has 4 sisters' --> 'He', 'has', 'sisters'
    # example1: 'my_mail@gmail.com' --> 'my', 'mail', 'gmail', 'com'
    # example1: 'I have 400 $' --> 'I', 'have'
    return token_list


def normalize(token_list):
    ''' puts all tokens in a given list in lower case and returns the list
    (changes can happen in place, i.e., the input itself may change) '''
    # TODO implement
    # make tokens in lower case
    token_list = [item.lower() for item in token_list]
    # TODO in a comment, give 5 examples (made up by yourself) of token pairs
    #      that might be normalized and treated as the same (5 different types
    #      of differences between the tokens in the pairs) but are treated as
    #      distinct by this simple normalization algorithm
    # example1: 'I', 'i' --> 'i'
    # example2: 'He', 'he' --> 'he'
    # example1: 'ABA', 'aba' --> 'aba'
    # example1: 'RNNs', 'rnns' --> 'rnns'
    # example1: 'NLP', 'nlp' --> 'nlp'
    return token_list


def getVocabulary(term_lists):
    ''' determines the list of distinct terms for a given list of term lists

    args:
        term_lists: a list of lists of normalized tokens / terms (i.e., strings)
    returns:
        a sorted list of all distinct terms in the input, i.e., the index terms
    '''
    # TODO
    # Make all lists as one vector
    flat_list = [item for sub_list in term_lists for item in sub_list]
    # store unique words
    vocab = list(set(flat_list))
    # sort unique words
    vocab.sort()
    return vocab


def getInverseVocabulary(vocab):
    ''' produces a mapping from index terms to indices in the vocabulary

    args:
        vocab: the list of index terms, the vocabulary
    results:
        a dictionary term2id such that vocab[term2id[term]] = term for all terms
    '''
    # TODO
    inverse_vocab = {k: v for v, k in enumerate(vocab)}
    return inverse_vocab


def getTermFrequencies(term_list, term2id):
    ''' determines the frequencies of all terms in a given term list

    able to handle terms in the list that are not in the vocabulary

    args:
        term_list: a list of normalized tokens produced from a document
        term2id: the inverse vocabulary produced by getInverseVocabulary
    returns:
        a vector (list) tfs of term frequencies, including zero entries
        tfs[i] refers to the term for which term2id[term] = i, for all i
    '''
    # TODO
    # initialize term frequencies list: to assure the right size
    tfs = []
    # fill frequencies list with zeros
    for i in range(len(term2id.keys())):
        tfs.append(0.0)
    # replace zeros with each term frequency
    for term in term_list:
        tfs[term2id[term]] += 1.0
    return tfs


def getInverseDocumentFrequencies(matrix):
    ''' determines the idf of all terms based on counts in given matrix

    args:
        matrix: the 2d weight matrix of the document collection (intermediate)
            matrix[i] returns a list of all weights for document i
            matrix[i][j] returns the weight for term j in document i
    returns:
        list of inverse document frequencies, one per term
    '''
    # TODO
    # initialize inverse document frequencies list
    idfs = []
    # fill inverse document frequencies list with zeros: to assure the right size
    for i in range(len(matrix[0])):
        idfs.append(0.0)
    # counter initialization
    index = 0
    # iterate over matrix: list of lists
    # the iterate on each element on the list
    # we the term is non zero we will give it a corresponding count
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] > 0.0:
                idfs[index] += 1
            else:
                pass
            # increment counter
            index += 1
        # reset the counter for the next loop
        index = 0
    # applying formula from lec19
    matrix_size = len(matrix)
    for i in range(len(idfs)):
        idfs[i] = math.log10(matrix_size / idfs[i])

    return idfs


def logTermFrequencies(tfs):
    ''' turns given list of term freq. into log term freq. and returns it
    (changes can happen in place, i.e., the input itself may change) '''
    # TODO
    for i in range(len(tfs)):
        if tfs[i] > 0:
            tfs[i] = float(math.log10(tfs[i]) + 1.0)
        else:
            tfs[i] = 0

    return tfs


def getTfIdf(tfs, idfs):
    ''' returns tf.idf weights for given document's term freq. and given idfs

    args:
        tfs: term frequencies of one document, i.e. one row in the matrix
        idfs: inverse document frequencies for the collection
    returns:
        list of tf.idf weights, i.e., elementwise product of the two input lists
    '''
    # TODO

    for i in range(len(tfs)):
        tfs[i] *= idfs[i]

    return tfs


def normalizeVector(vector):
    ''' normalizes a vector by dividing each element by the L2 norm
    (changes can happen in place, i.e., the input itself may change)

    args:
        vector: a list of numerical values, e.g. log term frequencies
    returns:
        the length-normalized vector
    '''
    # TODO
    l2_norm = 0
    index = 0
    # sum of vector's items
    for i in range(len(vector)):
        l2_norm += vector[i]

    l2_norm = float(math.sqrt(l2_norm))
    for i in range(len(vector)):
        vector[i] = vector[i] / l2_norm

    return vector


###############################################################
def dotProduct(v1, v2):
    ''' returns the dot product of two input vectors '''
    dot_product = 0.0
    for i in range(len(v1)):
        dot_product += v1[i] * v2[i]

    return dot_product


def runQuery(query, k, matrix, term2id):
    ''' executes a given query using a given weight matrix

    processes the query to obtain a vector of normalized log term frequencies,
    then returns the top k documents

    args:
        query: a string to process for document retrieval
        k: the (maximum) number of documents to return
        matrix: the 2d weight matrix of the document collection
            matrix[i] returns all weights for document i
            matrix[i][j] returns the weight for term j in document i
        term2id: a mapping from terms to indices in the second matrix dimension
    returns:
        up to k document indices ranked by the match score between the documents
        and the query; only documents with non-zero score should be returned
        (so it can be fewer than k)
    '''

    # Initialize dot product dict
    dot_product_value = {}
    # print the tokens after normalization in the output
    print(normalize(tokenize(query)))
    # get term frequency of the query tfs
    query_vector = getTermFrequencies(normalize(tokenize(query)), term2id)
    # log frequencies
    logTermFrequencies(query_vector)
    # L2 normalization
    normalizeVector(query_vector)

    # calculate the dot product of the query vector and matrix
    for i in range(len(matrix)):
        dot_product_value[i] = dotProduct(query_vector, matrix[i])
    # sort dot_product_value dict by value
    #dot_product_value = dict(sorted(dot_product_value.items(), key=operator.itemgetter(1), reverse=True))
    dot_product_value = {r: dot_product_value[r] for r in sorted(dot_product_value, key=dot_product_value.get, reverse=True)}
    # get key of each term in dot product
    keys = list(dot_product_value.keys())
    # list with top scored documents keys
    sub = keys[0:k]

    return sub


def main():
    # process all files (tokenization and token normalization)
    term_lists = []
    file_names = []
    for txtFile in glob.glob(os.path.join('data/', '*.txt')):
        with open(txtFile) as tf:
            term_lists.append(normalize(tokenize('\n'.join(tf.readlines()))))
            file_names.append(txtFile)
    # determine the vocabulary and the inverse mapping
    vocab = getVocabulary(term_lists)
    term2id = getInverseVocabulary(vocab)

    # size should be 9084 once the functions above are implemented
    print('vocabulary size:', len(vocab))

    # compute the weight matrix
    matrix = [[0.0 for i in range(len(vocab))] for j in range(len(term_lists))]
    for i, term_list in enumerate(term_lists):
        matrix[i] = getTermFrequencies(term_list, term2id)

    idfs = getInverseDocumentFrequencies(matrix)
    for i in range(len(matrix)):
        matrix[i] = logTermFrequencies(matrix[i])
        matrix[i] = getTfIdf(matrix[i], idfs)
        matrix[i] = normalizeVector(matrix[i])

    # run some test queries
    docs = runQuery('god', 3, matrix, term2id)
    print([file_names[i] for i in docs], end='\n\n\n')
    docs = runQuery('liberty freedom justice', 3, matrix, term2id)
    print([file_names[i] for i in docs], end='\n\n\n')
    docs = runQuery('Though passion may have strained it must not break our '
                    'bonds of affection', 3, matrix, term2id)
    print([file_names[i] for i in docs], end='\n\n\n')
    docs = runQuery('carnage', 1, matrix, term2id)
    print([file_names[i] for i in docs], end='\n\n\n')


if __name__ == '__main__':
    main()
