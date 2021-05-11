import os
import glob
import sys
from operator import itemgetter # for sort

import plsa


def print_topic_word_distribution(corpus, number_of_topics, topk, filepath):
    """
    Print topic-word distribution to file and list @topk most probable words for each topic
    """
    print("Writing topic-word distribution to file: " + filepath)
    V = len(corpus.vocabulary)  # size of vocabulary
    assert (topk < V)
    f = open(filepath, "w")
    for k in range(number_of_topics):
        word_prob = corpus.topic_word_prob[k, :]
        word_index_prob = []
        for i in range(V):
            word_index_prob.append([i, word_prob[i]])
        word_index_prob = sorted(word_index_prob, key=itemgetter(1), reverse=True)  # sort by word count
        f.write("Topic #" + str(k) + ":\n")
        for i in range(topk):
            index = word_index_prob[i][0]
            f.write(corpus.vocabulary[index] + " ")
        f.write("\n")

    f.close()


def print_document_topic_distribution(corpus, number_of_topics, topk, filepath):
    """
    Print document-topic distribution to file and list @topk most probable topics for each document
    """
    print("Writing document-topic distribution to file: " + filepath)
    assert (topk <= number_of_topics)
    f = open(filepath, "w")
    D = len(corpus.documents)  # number of documents
    for d in range(D):
        topic_prob = corpus.document_topic_prob[d, :]
        topic_index_prob = []
        for i in range(number_of_topics):
            topic_index_prob.append([i, topic_prob[i]])
        # topic_index_prob = sorted(topic_index_prob, key=itemgetter(1), reverse=True) -> Sorting
        f.write("Document #" + str(d) + ":\n")
        for i in range(topk):
            # index = topic_index_prob[i][0]
            # f.write("topic" + str(index) + " ")
            prob = topic_index_prob[i][1]
            f.write(str(prob) + " ")
        f.write("\n")

    f.close()

def print_document_topic_distribution_csv(corpus, number_of_topics, topk, filepath):
    assert(topk <= number_of_topics)
    np.savetxt(filepath,corpus.document_topic_prob,delimiter=",")

def print_documents(corpus, filepath):
    f = open(filepath, "w")
    D = len(corpus.documents) # number of documents
    for d in corpus.documents:
      f.write(d + "\n")
    f.close()

def main(argv):
    A = Corpus('gdrive/MyDrive/ThesisDataset/223_motifs_without_x.txt')

    document_file = ('gdrive/MyDrive/ThesisDataset/training_data_0_952_1.txt')

    documents = Document(document_file)  # instantiate document
    documents.split()  # tokenize

    for i in documents.lines:
        A.add_document(i)
    A.build_vocabulary()

    print("Vocabulary size:" + str(len(A.vocabulary)))
    print("Number of documents:" + str(len(A.documents)))
    # print(len(documents.lines))


    number_of_topics = 50
    max_iterations = 10
    A.plsa(number_of_topics, max_iterations)
    tpks = number_of_topics  # Number of top words in a topic

# print corpus.document_topic_prob
# print corpus.topic_word_prob
# cPickle.dump(corpus, open('./models/corpus.pickle', 'w'))

    print_documents(A, 'gdrive/MyDrive/ThesisDataset/plsa2_withoutX/plsa2_result50_50topwords/documents.txt')
##print_topic_word_distribution(A, number_of_topics, k, 'gdrive/MyDrive/ThesisDataset/plsa2_result50/topic_word.txt') with k <= number_of_topics => k - most probable topics
    print_topic_word_distribution(A, number_of_topics, tpks,
                              'gdrive/MyDrive/ThesisDataset/plsa2_withoutX/plsa2_result50_50topwords/topic_word.txt')
    print_document_topic_distribution(A, number_of_topics, tpks,
                                  'gdrive/MyDrive/ThesisDataset/plsa2_withoutX/plsa2_result50_50topwords/document-topic.txt')
    print_document_topic_distribution_csv(A, number_of_topics, tpks,
                                      'gdrive/MyDrive/ThesisDataset/plsa2_withoutX/plsa2_result50_50topwords/document-topic.csv')

if __name__ == "__main__":
    main(sys.argv)