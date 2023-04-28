import argparse
import numpy as np
import math

def get_inputs():
    """
    Collects all the inputs from the command line and returns the data. To use this function:

        validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, predicted_file, metric_file = parse_args()

    Where above the arguments have the following types:

        validation_data --> A list of validation examples, where each element is a list:
            validation_data[i] = [(word1, tag1), (word2, tag2), (word3, tag3), ...]
        
        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

        hmminit --> A np.ndarray matrix representing the initial probabilities

        hmmemit --> A np.ndarray matrix representing the emission probabilities

        hmmtrans --> A np.ndarray matrix representing the transition probabilities

        predicted_file --> A file path (string) to which you should write your predictions

        metric_file --> A file path (string) to which you should write your metrics
    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument("validation_data", type=str)
    parser.add_argument("index_to_word", type=str)
    parser.add_argument("index_to_tag", type=str)
    parser.add_argument("hmminit", type=str)
    parser.add_argument("hmmemit", type=str)
    parser.add_argument("hmmtrans", type=str)
    parser.add_argument("predicted_file", type=str)
    parser.add_argument("metric_file", type=str)

    args = parser.parse_args()

    validation_data = list()
    with open(args.validation_data, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            validation_data.append(xi)
    
    with open(args.index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}
    
    with open(args.index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}
    
    hmminit = np.loadtxt(args.hmminit, dtype=float, delimiter=" ")
    hmmemit = np.loadtxt(args.hmmemit, dtype=float, delimiter=" ")
    hmmtrans = np.loadtxt(args.hmmtrans, dtype=float, delimiter=" ")

    return validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, args.predicted_file, args.metric_file

# You should implement a logsumexp function that takes in either a vector or matrix
# and performs the log-sum-exp trick on the vector, or on the rows of the matrix

def forwardbackward(seq, loginit, logtrans, logemit, words_to_indices, tags_to_indices):
    """
    Your implementation of the forward-backward algorithm.

        seq is an input sequence, a list of words (represented as strings)

        loginit is a np.ndarray matrix containing the log of the initial matrix

        logtrans is a np.ndarray matrix containing the log of the transition matrix

        logemit is a np.ndarray matrix containing the log of the emission matrix

        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

    You should compute the log-alpha and log-beta values and predict the tags for this sequence.
    """
    L = len(seq) #3
    M = len(loginit)#2
    
    # Initialize log_alpha and fill it in - feel free to use words_to_indices to index the specific word
    log_alpha = np.zeros([M, L])
    log_alpha[:,0] = loginit + logemit[:, words_to_indices[seq[0]]]
    for t in range(1, L):
        for j in range(0, M):
            lse = logsumexp(log_alpha[:,t-1] + logtrans[:,j])    
            log_alpha[j][t] = logemit[j, words_to_indices[seq[t]]] + lse

    # Initialize log_beta and fill it in - feel free to use words_to_indices to index the specific word
    log_beta = np.zeros((M, L))
    for t in range(len(seq)-2, -1, -1):
        for j in range(0, M):
            log_beta[j][t] = logsumexp(log_beta[:,t+1] + logemit[:, words_to_indices[seq[t+1]]] + logtrans[j,:])
        
    # Compute the predicted tags for the sequence - tags_to_indices can be used to index to the rwquired tag
    prob = log_alpha + log_beta
    tags_index = np.argmax(prob, axis = 0)
    indices_to_tags = {v: k for k, v in tags_to_indices.items()}
    tags = [indices_to_tags[index] for index in tags_index]
    
    # Return the predicted tags and the log-probability
    ll = logsumexp(log_alpha[:,-1])
    return tags, ll

def logsumexp(matrix):
    # perform logsumexp on each row of matrix
    mmax = np.max(matrix)
    return mmax + np.log(np.sum(np.exp(matrix - mmax)))
    
    

    
    
if __name__ == "__main__":
    # Get the input data
    validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, predicted_file, metric_file = get_inputs()

    # For each sequence, run forward_backward to get the predicted tags and 
    # the log-probability of that sequence.

    # Compute the average log-likelihood and the accuracy. The average log-likelihood 
    # is just the average of the log-likelihood over all sequences. The accuracy is 
    # the total number of correct tags across all sequences divided by the total number 
    # of tags across all sequences.
    predictions = []
    total_likelihood = 0.0
    total_tags = 0
    correct = 0
    
    for sequence in validation_data:
        seq = [pair[0] for pair in sequence]
        tags = [pair[1] for pair in sequence]
        
        tags_pred, log_likelihood = forwardbackward(seq, np.log(hmminit), np.log(hmmtrans), np.log(hmmemit), words_to_indices, tags_to_indices)

        predicted_sentence = ["{}\t{}".format(seq, tags_pred) for seq, tags_pred in zip(seq, tags_pred)]
        
        predictions.append(predicted_sentence)
        
        total_likelihood += log_likelihood
        total_tags += len(tags_pred)
        
        correct += sum([tags_pred == tags for tags_pred, tags in zip(tags_pred, tags)])
    
    avg_likelihood = total_likelihood / float(len(validation_data))
    tag_accuracy = float(correct) / float(total_tags)
    
    with open(predicted_file, "w") as f:
        for sentence in predictions:
            line = "\n".join(sentence)
            f.write(line + "\n\n")
            
    with open(metric_file, "w") as f:
        f.write("Average Log-Likelihood: {}\n".format(avg_likelihood))
        f.write("Accuracy: {}".format(tag_accuracy))


    