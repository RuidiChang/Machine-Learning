import argparse
import numpy as np


def get_inputs():
    """
    Collects all the inputs from the command line and returns the data. To use this function:

        train_data, words_to_index, tags_to_index, init_out, emit_out, trans_out = get_inputs()
    
    Where above the arguments have the following types:

        train_data --> A list of training examples, where each training example is a list
            of tuples train_data[i] = [(word1, tag1), (word2, tag2), (word3, tag3), ...]
        
        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

        init_out --> A file path to which you should write your initial probabilities

        emit_out --> A file path to which you should write your emission probabilities

        trans_out --> A file path to which you should write your transition probabilities
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str)
    parser.add_argument("index_to_word", type=str)
    parser.add_argument("index_to_tag", type=str)
    parser.add_argument("hmmprior", type=str)
    parser.add_argument("hmmemit", type=str)
    parser.add_argument("hmmtrans", type=str)

    args = parser.parse_args()

    train_data = list()
    with open(args.train_input, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            train_data.append(xi)
    
    with open(args.index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}
    
    with open(args.index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}
    
    return train_data, words_to_indices, tags_to_indices, args.hmmprior, args.hmmemit, args.hmmtrans


def learn(train_data, words_to_indices, tags_to_indices):
    num_tags = len(tags_to_indices)
    num_words = len(words_to_indices)

    pi = np.zeros(num_tags)
    B = np.zeros((num_tags, num_tags))
    A = np.zeros((num_tags, num_words))
        
    for example in train_data:
        words = [pair[0] for pair in example]
        tags = [pair[1] for pair in example]
        tag_ids = [tags_to_indices[tag] for tag in tags]
        word_ids = [words_to_indices[word] for word in words]

        pi[tag_ids[0]] += 1
        for i in range(len(example)-1):
            B[tag_ids[i], tag_ids[i+1]] += 1

        for tag_id, word_id in zip(tag_ids, word_ids):
            A[tag_id, word_id] += 1

    pi += 1
    B += 1
    A += 1

    pi /= np.sum(pi)
    B /= np.sum(B, axis=1, keepdims=True)
    A /= np.sum(A, axis=1, keepdims=True)

    return pi, B, A



if __name__ == "__main__":
    # Collect the input data
    train_data, words_to_index, tags_to_index, init_out, emit_out, trans_out = get_inputs()

    # Initialize the initial, emission, and transition matrices

    # Increment the matrices

    # Add a pseudocount
    prior, trans, emit = learn(train_data, words_to_index, tags_to_index)
    # Save your matrices to the output files --- the reference solution uses 
    # np.savetxt (specify delimiter=" " for the matrices)
    np.savetxt(init_out, prior) 
    np.savetxt(trans_out, trans) 
    np.savetxt(emit_out, emit) 
    
    pass
