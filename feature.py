import csv
import numpy as np
import argparse

VECTOR_LEN = 300   # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = dict()
    with open(file) as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map


def feature(data, glove):
    result = []
    for label, review in data:
        words = review.split(' ')
        embeddings = [glove[word] for word in words if word in glove]
        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
        else:
            avg_embedding = np.zeros(VECTOR_LEN)
        result.append(np.hstack([label, avg_embedding]))
    return np.array(result)


def writefile(data, file):
    with open(file, 'w') as f:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]-1):
                f.write(format(data[i][j], '.6f') + '\t')
            f.write(format(data[i][data.shape[1]-1], '.6f') + '\n')


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("validation_input", type=str, help='path to validation input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the input .tsv file')
    parser.add_argument("feature_dictionary_in", type=str, 
                        help='path to the GloVe feature dictionary .txt file')
    parser.add_argument("train_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("validation_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the validation data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the test data should be written')
    args = parser.parse_args()
    
    feature_dictionary = load_feature_dictionary(args.feature_dictionary_in)
    train_feature_arrays = feature(load_tsv_dataset(args.train_input), feature_dictionary)
    test_feature_arrays = feature(load_tsv_dataset(args.test_input),feature_dictionary)
    validation_feature_arrays = feature(load_tsv_dataset(args.validation_input),feature_dictionary)

    writefile(train_feature_arrays,args.train_out)
    writefile(test_feature_arrays,args.test_out)
    writefile(validation_feature_arrays,args.validation_out)
