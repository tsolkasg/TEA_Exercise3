from helper import *

# Data paths
train_path = "Data/en_ewt-ud-train.conllu"
dev_path = "Data/en_ewt-ud-dev.conllu"
test_path = "Data/en_ewt-ud-test.conllu"

# Read data
words_count = count_words(train_path)
train_data, word2pos, pos2word, postag_set, postags_data = read_train_data(train_path, words_count)
most_probable_tag = find_most_probable_tag(pos2word)
word_2_most_probable_pos = keep_most_frequent_tags(word2pos)

dev_data = read_test_data(dev_path, words_count)

test_data = read_test_data(test_path, words_count)

# Create plots
HMMplotTrainTestLines(train_data, dev_data, pos2word, word_2_most_probable_pos, postag_set)

# Train on both train+dev data, report on test data
print_HMM_classification_report(train_data + dev_data, test_data)
