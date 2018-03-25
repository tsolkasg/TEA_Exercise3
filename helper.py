from collections import defaultdict
import numpy as np
import math
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn_crfsuite import metrics
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from nltk.util import ngrams
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

UNK = "*unkn*"

redundant_postags = ['PUNCT', 'SYM']

# Set to 1 to use the original word
# Set to 2 to use the lemma of the word
index_to_use = 2


# Returns a dictionary from word to occurrences
def count_words(path):
    # all words - to find rare words
    words_count = defaultdict(lambda: 0)
    with open(path, "r", encoding="utf8") as file:
        for line in file:
            # Ignore comments
            if line.startswith('#'):
                continue

            # Ignore empty lines
            if not line.strip():
                continue

            # Get information from line
            # print(line)
            tokens = line.split()
            word = tokens[index_to_use]
            postag = tokens[3]

            if postag in redundant_postags:
                continue

            # Fill dictionary
            words_count[word] += 1

    return words_count


# Returns a list with lists(sentences) of tuples(word, postag, initial_word) AND various counts for words/pos tags
# Initial_word is needed only by CRF classifier when we also have *unk* in train data, in all other cases it is not used
def read_train_data(path, words_count):
    # word to postag frequency - for baseline classifier
    word2pos = defaultdict(lambda: defaultdict(lambda: 0))
    # postag to word frequency - for viterbi
    pos2word = defaultdict(lambda: defaultdict(lambda: 0))  # remove rare??(unknown)
    # all postags - for viterbi
    postag_set = set()
    # sequence of postags per sentence
    postags_data = []
    # list of lists of tuples
    sentence_data = []
    with open(path, "r", encoding="utf8") as file:
        list_of_sentence_words = []
        # postag sequence per sentence(needed for postag bigrams)
        list_of_sentence_postags = []
        for line in file:
            # ignore comments
            if line.startswith('#'):
                continue

            # If line is empty
            if not line.strip():
                # Remove sentences with only redundant(useless) tags
                if len(list_of_sentence_words) == 0:
                    continue
                sentence_data.append(list_of_sentence_words)
                postags_data.append(list_of_sentence_postags)
                list_of_sentence_words = []
                list_of_sentence_postags = []
                continue

            # Get information from line
            # print(line)
            tokens = line.split()
            word = tokens[index_to_use]
            initial_word = word
            # Set threshold for unknown words
            if ((word not in words_count) or words_count[word] < 2):
                word = UNK

            postag = tokens[3]

            if postag in redundant_postags:
                continue

            # Fill dictionaries
            word2pos[initial_word][postag] += 1
            pos2word[postag][word] += 1
            postag_set.add(postag)
            word_info = (word, postag, initial_word)
            list_of_sentence_words.append(word_info)
            list_of_sentence_postags.append(postag)

    return sentence_data, word2pos, pos2word, postag_set, postags_data


# Returns a list with lists(sentences) of tuples(word, postag, initial_word)
def read_test_data(path, words_count):
    data, _, _, _, _ = read_train_data(path, words_count)
    return data


# Keep the most frequent tag for each word - Used by base classifier
def keep_most_frequent_tags(word2pos):
    result_dict = {}
    for word in word2pos:
        max_occurrences = 0
        best_tag = ""
        for tag, occurrences in word2pos[word].items():
            if (occurrences > max_occurrences):
                max_occurrences = occurrences
                best_tag = tag
        result_dict[word] = best_tag

    return result_dict


# Get most probable of all tags
def find_most_probable_tag(pos2word):
    most_occurrences = 0
    most_probable_tag = ""
    for tag in pos2word:
        tag_occurences = sum(pos2word[tag].values())
        if tag_occurences > most_occurrences:
            most_occurrences = tag_occurences
            most_probable_tag = tag
    return most_probable_tag


# Count pos to pos tag frequencies
def createPosTagBigrams(pos_tag_list_of_all_sentences):
    pos_model = defaultdict(lambda: defaultdict(lambda: 0))
    for postags_list_of_one_sentence in pos_tag_list_of_all_sentences:
        model_grams = [gram for gram in ngrams(postags_list_of_one_sentence, 2)]
        pos_model["*start*"][postags_list_of_one_sentence[0]] += 1

        for w1, w2 in model_grams:
            pos_model[w1][w2] += 1
        pos_model[postags_list_of_one_sentence[len(postags_list_of_one_sentence) - 1]]["*end*"] += 1
    return pos_model


# Data must be list of lists of tuples
# Used for benchmark(to calculate probabilities for 10%, 20% ,... of data)
def get_HMM_info(data):
    # Postag to word frequency - for viterbi
    pos2word = defaultdict(lambda: defaultdict(lambda: 0))
    # All postags - for viterbi
    postag_set = set()
    # All words - its size is used for smoothing
    words_set = set()
    # Sequence of postags per sentence
    postags_data = []
    for sentence in data:
        # Postag sequence per sentence(needed for postag bigrams)
        list_of_sentence_postags = []
        for word, postag, _ in sentence:
            # Fill dictionaries
            pos2word[postag][word] += 1
            postag_set.add(postag)
            words_set.add(word)
            list_of_sentence_postags.append(postag)

        postags_data.append(list_of_sentence_postags)

    pos2pos = createPosTagBigrams(postags_data)
    return pos2word, pos2pos, postag_set, words_set


# Data must be list of lists of tuples
# Return test label as 1 list
def get_HMM_y(data):
    test_y = []
    for sentence in data:
        for _, postag, _ in sentence:
            # Fill list
            test_y.append(postag)

    return test_y


# For CRF features (we add features for previous/next word when there is one)
# sent is : [(word, postag, initial_word), ....]
def word2features(sent, i):
    word = sent[i][0]
    initial_word = sent[0][2]
    features = {
        'word_lower': word.lower(),
        'word[-3:]': initial_word[-3:],
        'word[-2:]': initial_word[-2:],
        'word_isupper': initial_word.isupper(),
        'word_istitle': initial_word.istitle(),
        'word_isdigit': initial_word.isdigit(),
        'word_containsUpper': any(letter.isupper() for letter in initial_word),
        'word_containsDigit': any(letter.isdigit() for letter in initial_word),
        'word_isalpha': all(letter.isalpha() for letter in initial_word),
        'word[:3]': initial_word[:3],
        'word[:2]': initial_word[:2],
    }

    # If not the first word, add features regarding previous word
    if i > 0:
        previous_word = sent[i - 1][0]
        previous_initial_word = sent[i - 1][2]
        features.update({
            'previous_word_lower': previous_word.lower(),
            'previous_word[-3:]': previous_initial_word[-3:],
            'previous_word[-2:]': previous_initial_word[-2:],
            'previous_word_isupper': previous_initial_word.isupper(),
            'previous_word_istitle': previous_initial_word.istitle(),
            'previous_word_isdigit': previous_initial_word.isdigit(),
            'previous_word_containsUpper': any(letter.isupper() for letter in previous_initial_word),
            'previous_word_containsDigit': any(letter.isdigit() for letter in previous_initial_word),
            'previous_word_isalpha': all(letter.isalpha() for letter in previous_initial_word),
            'previous_word[:3]': previous_initial_word[:3],
            'previous_word[:2]': previous_initial_word[:2],
        })
    else:
        features['BOS'] = True

    # If not the last word, add features regarding next word
    if i < len(sent) - 1:
        next_word = sent[i + 1][0]
        next_initial_word = sent[i + 1][2]
        features.update({
            'next_word_lower': next_word.lower(),
            'next_word[-3:]': next_initial_word[-3:],
            'next_word[-2:]': next_initial_word[-2:],
            'next_word_isupper': next_initial_word.isupper(),
            'next_word_istitle': next_initial_word.istitle(),
            'next_word_isdigit': next_initial_word.isdigit(),
            'next_word_containsUpper': any(letter.isupper() for letter in next_initial_word),
            'next_word_containsDigit': any(letter.isdigit() for letter in next_initial_word),
            'next_word_isalpha': all(letter.isalpha() for letter in next_initial_word),
            'next_word[:3]': next_initial_word[:3],
            'next_word[:2]': next_initial_word[:2],
        })
    else:
        features['EOS'] = True

    return features


# Smoothed probabilities from dictionary of bigram frequencies
def getBiProbality(model, w1, w2, vocab):
    total_count = float(sum(model[w1].values()))
    if w2 not in model[w1]:
        return (1 / (total_count + len(vocab)))
    prob = (model[w1][w2] + 1) / (total_count + len(vocab))
    return prob


# Viterbi
def viterbi(sent, post_tag_vocab, word_vocab, pos_model, pos_word_model):
    viterbi_dict = defaultdict(lambda: defaultdict(lambda: 0))
    backpointer = defaultdict(lambda: defaultdict(lambda: 0))
    current_word, _, _ = sent[0]
    # First word
    for pos in post_tag_vocab:
        viterbi_dict[pos][1] = np.log(getBiProbality(pos_model, "*start*", pos, post_tag_vocab)) + np.log(
            getBiProbality(pos_word_model, pos, current_word, word_vocab))
        backpointer[pos][1] = "*start*"
    # Rest words
    for t, word_tuple in enumerate(sent[1:], 2):
        current_word, _, _ = word_tuple
        for pos in post_tag_vocab:
            best_pos = ""
            max_prob = -math.inf
            for pos2 in post_tag_vocab:
                prob = viterbi_dict[pos2][t - 1] + np.log(
                    getBiProbality(pos_model, pos2, pos, post_tag_vocab)) + np.log(
                    getBiProbality(pos_word_model, pos, current_word, word_vocab))
                if prob > max_prob:
                    max_prob = prob
                    best_pos = pos2
            viterbi_dict[pos][t] = max_prob
            backpointer[pos][t] = best_pos

    # *end*
    best_pos = ""
    max_prob = -math.inf
    for pos in post_tag_vocab:
        prob = viterbi_dict[pos][len(sent)] + np.log(getBiProbality(pos_model, pos, "*end*", post_tag_vocab))
        if prob > max_prob:
            max_prob = prob
            best_pos = pos
    viterbi_dict["*end*"][len(sent) + 1] = max_prob
    backpointer["*end*"][len(sent) + 1] = best_pos

    # Backtrack
    solution = list()
    current_state = "*end*"
    cnt = len(sent) + 1
    previous_state = backpointer[current_state][cnt]
    while previous_state != "*start*":
        solution.append(previous_state)
        current_state = previous_state
        cnt -= 1
        previous_state = backpointer[current_state][cnt]
    return list(reversed(solution))


# Returns metrics
def CRFbenchmark(clf, train_X, train_y, test_X, test_y, postags, word2pos, most_probable_tag, baseline=False):
    # Predicts always the most popular tag
    if (baseline):
        pred = []
        for sentence in test_X:
            sentence_pred = []
            for features in sentence:
                if features['word_lower'] in word2pos:
                    sentence_pred.append(word2pos[features['word_lower']])
                else:
                    sentence_pred.append(most_probable_tag)

            pred.append(sentence_pred)
    else:
        clf.fit(train_X, train_y)
        pred = clf.predict(test_X)

    result = {}
    for tag in postags:
        m = metrics.flat_classification_report(test_y, pred, labels=[tag], digits=3)
        result[tag] = float(m.split()[7])

    m = metrics.flat_classification_report(test_y, pred, digits=3)
    # For Overall results
    result["Total"] = float(m.split()[84])
    return result


# To change form of data - to help plotting
def get_list_from_dict(dictionary, tag):
    list_result = list()
    for i in range(1, 11):
        list_result.append(dictionary[i][tag])
    return list_result

# To change form of data - to help plotting
def fix_results(initial_results, postags):
    results = {}
    results['train_size'] = initial_results['train_size']
    results['base_classifier'] = {}
    results['on_test'] = {}
    results['on_train'] = {}

    for tag in postags:
        results['on_train'][tag] = get_list_from_dict(initial_results['on_train'], tag)
        results['on_test'][tag] = get_list_from_dict(initial_results['on_test'], tag)
        results['base_classifier'][tag] = get_list_from_dict(initial_results['base_classifier'], tag)

    tag = "Total"
    results['on_train'][tag] = get_list_from_dict(initial_results['on_train'], tag)
    results['on_test'][tag] = get_list_from_dict(initial_results['on_test'], tag)
    results['base_classifier'][tag] = get_list_from_dict(initial_results['base_classifier'], tag)

    return results


# CRF plots
# All data = list of lists
def CRFplotTrainTestLines(title, clf, X_train, y_train, X_test, y_test, postags, word2pos, most_probable_tag):
    train_x_s_s, train_y_s_s = shuffle(X_train, y_train)

    results = {}
    results['train_size'] = []
    results['base_classifier'] = {}
    results['on_test'] = {}
    results['on_train'] = {}

    for i in range(1, 11):
        print("iteration : %d" % i)
        if (i == 10):
            train_x_part = train_x_s_s
            train_y_part = train_y_s_s
        else:
            to = int(i * (len(train_x_s_s) / 10))
            train_x_part = train_x_s_s[0:to]
            train_y_part = train_y_s_s[0:to]

        # Train size
        results['train_size'].append(len(train_x_part))

        # Train
        result = CRFbenchmark(clf, train_x_part, train_y_part, train_x_part, train_y_part, postags, word2pos,
                              most_probable_tag)
        results['on_train'][i] = result

        # Test
        result = CRFbenchmark(clf, train_x_part, train_y_part, X_test, y_test, postags, word2pos, most_probable_tag)
        results['on_test'][i] = result

        # Base classifier
        result = CRFbenchmark(None, train_x_part, train_y_part, X_test, y_test, postags, word2pos, most_probable_tag,
                              True)
        results['base_classifier'][i] = result

    results = fix_results(results, postags)

    # Create the plots
    fontP = FontProperties()
    fontP.set_size('small')

    for tag in postags:
        fig = plt.figure()
        fig.suptitle('Learning Curves : ' + tag, fontsize=20)
        ax = fig.add_subplot(111)
        ax.axis([0, len(train_x_part) + 1000, 0, 1.1])
        line_up, = ax.plot(results['train_size'], results['on_train'][tag], 'o-', label='Train', color="blue")
        line_down, = ax.plot(results['train_size'], results['on_test'][tag], 'o-', label='Dev', color="orange")
        line_base, = ax.plot(results['train_size'], results['base_classifier'][tag], 'o-', label='Baseline',
                             color="green")

        plt.xlabel('N. of training instances', fontsize=18)
        plt.ylabel('F1 score', fontsize=16)
        plt.legend([line_up, line_down, line_base], ['Train', 'Dev', 'Baseline'], prop=fontP)
        plt.grid(True)

        plt.show()

        fig.savefig('crf_' + tag + '.png')

    tag = "Total"
    fig = plt.figure()
    fig.suptitle('Learning Curves : ' + tag, fontsize=20)
    ax = fig.add_subplot(111)
    ax.axis([0, len(train_x_part) + 1000, 0, 1.1])
    line_up, = ax.plot(results['train_size'], results['on_train'][tag], 'o-', label='Train', color="blue")
    line_down, = ax.plot(results['train_size'], results['on_test'][tag], 'o-', label='Dev', color="orange")
    line_base, = ax.plot(results['train_size'], results['base_classifier'][tag], 'o-', label='Baseline',
                         color="green")

    plt.xlabel('N. of training instances', fontsize=18)
    plt.ylabel('F1 score', fontsize=16)
    plt.legend([line_up, line_down, line_base], ['Train', 'Dev', 'Baseline'], prop=fontP)
    plt.grid(True)

    plt.show()

    fig.savefig('crf_' + tag + '.png')


def HMMbenchmark(train_X, test_X, most_probable_tag, word2pos, postags, baseline=False):
    # Predicts always the most popular tag
    if (baseline):
        pred = []
        for sentence in test_X:
            for _, postag, initial_word in sentence:
                if initial_word in word2pos:
                    pred.append(word2pos[initial_word])
                else:
                    pred.append(most_probable_tag)
    else:
        pos2word, pos_bigrams, all_pos, all_words = get_HMM_info(train_X)
        pred = list()
        for s in test_X:
            res = viterbi(s, all_pos, all_words, pos_bigrams, pos2word)
            pred += res

    test_y = get_HMM_y(test_X)
    result = {}
    for tag in postags:
        f1 = f1_score(test_y, pred, labels=[tag], average="micro")
        result[tag] = f1

    f1 = f1_score(test_y, pred, average="micro")
    result["Total"] = f1
    return result

# HMM plots
# All data = list of lists
def HMMplotTrainTestLines(X_train, X_test, pos2word, word2pos, postag_set):
    train_x_s_s = X_train
    test_x_s_s = X_test

    train_x_s_s = shuffle(train_x_s_s)
    postags = postag_set
    results = {}
    results['train_size'] = []
    results['base_classifier'] = {}
    results['on_test'] = {}
    results['on_train'] = {}
    most_probable_tag = find_most_probable_tag(pos2word)
    for i in range(1, 11):
        if (i == 10):
            train_x_part = train_x_s_s
        else:
            to = int(i * (len(train_x_s_s) / 10))
            train_x_part = train_x_s_s[0:to]
        print("iteration : %d" % i)

        # Train size
        results['train_size'].append(len(train_x_part))

        # Train
        result = HMMbenchmark(train_x_part, train_x_part, most_probable_tag, word2pos, postags)
        results['on_train'][i] = result

        # Test
        result = HMMbenchmark(train_x_part, test_x_s_s, most_probable_tag, word2pos, postags)
        results['on_test'][i] = result

        # Base classifier
        result = HMMbenchmark(train_x_part, test_x_s_s, most_probable_tag, word2pos, postags, True)
        results['base_classifier'][i] = result

    results = fix_results(results, postags)

    # Create the plot
    fontP = FontProperties()
    fontP.set_size('small')

    for tag in postags:
        fig = plt.figure()
        fig.suptitle('Learning Curves : ' + tag, fontsize=20)
        ax = fig.add_subplot(111)
        ax.axis([0, len(train_x_part) + 100, 0, 1.1])
        line_up, = ax.plot(results['train_size'], results['on_train'][tag], 'o-', label='Train', color="blue")
        line_down, = ax.plot(results['train_size'], results['on_test'][tag], 'o-', label='Dev', color="orange")
        line_base, = ax.plot(results['train_size'], results['base_classifier'][tag], 'o-', label='Baseline',
                             color="green")

        plt.xlabel('N. of training instances', fontsize=18)
        plt.ylabel('F1 score', fontsize=16)
        plt.legend([line_up, line_down, line_base], ['Train', 'Dev', 'Baseline'], prop=fontP)
        plt.grid(True)

        plt.show()

        fig.savefig('hmm_' + tag + '.png')

    tag = "Total"
    fig = plt.figure()
    fig.suptitle('Learning Curves : ' + tag, fontsize=20)
    ax = fig.add_subplot(111)
    ax.axis([0, len(train_x_part) + 100, 0, 1.1])
    line_up, = ax.plot(results['train_size'], results['on_train'][tag], 'o-', label='Train', color="blue")
    line_down, = ax.plot(results['train_size'], results['on_test'][tag], 'o-', label='Dev', color="orange")
    line_base, = ax.plot(results['train_size'], results['base_classifier'][tag], 'o-', label='Baseline',
                         color="green")

    plt.xlabel('N. of training instances', fontsize=18)
    plt.ylabel('F1 score', fontsize=16)
    plt.legend([line_up, line_down, line_base], ['Train', 'Dev', 'Baseline'], prop=fontP)
    plt.grid(True)

    plt.show()

    fig.savefig('hmm_' + tag + '.png')


def print_HMM_classification_report(X_train, X_test):
    pos2word, pos_bigrams, all_pos, all_words = get_HMM_info(X_train)
    pred = list()
    # counter = 0 # used for printing some results
    for s in X_test:
        res = viterbi(s, all_pos, all_words, pos_bigrams, pos2word)
        # Print some results(predictions)
        # if counter <10 :
        #     print(s)
        #     print(res)
        # else:
        #     pred += res
        #     break
        pred += res
        # counter += 1

    test_y = get_HMM_y(X_test)

    print(classification_report(test_y, pred))
    print(confusion_matrix(test_y, pred))
