import sys
from operator import itemgetter  # for sort

import plsa
import numpy as np


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
    np.savetxt(filepath, corpus.document_topic_prob, delimiter=",")


def print_documents(corpus, filepath):
    f = open(filepath, "w")
    for d in corpus.documents:
      f.write(d + "\n")
    f.close()


def main(argv):
    A = plsa.Corpus('./171_motifs_without_x.txt')

    document_file = './training_data_0_951_1.txt'

    documents = plsa.Document(document_file)  # instantiate document
    documents.split()  # tokenize

    for i in documents.lines:
        A.add_document(i)
    A.build_vocabulary()

    print("Vocabulary size:" + str(len(A.vocabulary)))
    print("Number of documents:" + str(len(A.documents)))
    # print(len(documents.lines))

    number_of_topics = int(argv[1])
    tpks = int(argv[2])
    max_iterations = int(argv[3])

    A.plsa(number_of_topics, max_iterations)
    

    # print corpus.document_topic_prob
    # print corpus.topic_word_prob
    # cPickle.dump(corpus, open('./models/corpus.pickle', 'w'))

    print_documents(A, './documents.txt')
    # print_topic_word_distribution(A, number_of_topics,
    # k, 'gdrive/MyDrive/ThesisDataset/plsa2_result50/topic_word.txt')
    # with k <= number_of_topics => k - most probable topics
    print_topic_word_distribution(A, number_of_topics, tpks,
                                  './TopicMotifs/plsa2_{}topics_{}topwords.txt'.format(number_of_topics, tpks))

    # print_document_topic_distribution(A, number_of_topics, tpks,
    # 'plsa2_withoutX/plsa2_result{}_{}topwords/document-topic.txt'.format(number_of_topics))

    print_document_topic_distribution_csv(A, number_of_topics,
                           tpks, './csv/plsa2_{}topics_{}topwords.csv'.format(number_of_topics, tpks))


if __name__ == "__main__":
    main(sys.argv)