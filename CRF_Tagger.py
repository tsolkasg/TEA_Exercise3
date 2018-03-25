import sklearn_crfsuite
from sklearn_crfsuite import metrics
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from helper import *


# Get features for sentence
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

# Get labels for sentence
def sent2labels(sent):
    return [postag for _, postag, _ in sent]


# Data paths
train_path = "Data/en_ewt-ud-train.conllu"
dev_path = "Data/en_ewt-ud-dev.conllu"
test_path = "Data/en_ewt-ud-test.conllu"

# Read data
words_count = count_words(train_path)
train_data, word2pos, pos2word, postag_set, postags_data = read_train_data(train_path, words_count)
most_probable_tag = find_most_probable_tag(pos2word)
word2pos = keep_most_frequent_tags(word2pos)
print("train data : %d" % len(train_data))

# print("distinct words : %d" % len(words_count))

dev_data = read_test_data(dev_path, words_count)
print("dev data : %d" % len(dev_data))

test_data = read_test_data(test_path, words_count)
print("test data : %d" % len(test_data))

# Create features (as list of lists)
X_train = [sent2features(s) for s in train_data]
y_train = [sent2labels(s) for s in train_data]

X_dev = [sent2features(s) for s in dev_data]
y_dev = [sent2labels(s) for s in dev_data]

X_test = [sent2features(s) for s in test_data]
y_test = [sent2labels(s) for s in test_data]

##########################################################################################################################################
##########################################################################################################################################
# CV

# create classifier
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    max_iterations=100,
    all_possible_transitions=True
)

# Create search parameters
params = {
    'c1': [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10],
    'c2': [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10],
}

# Create scorer objects
f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted')

# Perform cross validation
rs = RandomizedSearchCV(crf, params,
                        cv=5,
                        verbose=1,
                        scoring=f1_scorer)
rs.fit(X_train, y_train)

print('best params:', rs.best_params_)
# print('best CV score:', rs.best_score_)

# Best estimator
crf = rs.best_estimator_

# Create plots
CRFplotTrainTestLines("crf", crf, X_train, y_train, X_dev, y_dev, postag_set, word2pos, most_probable_tag)

# Calculate metrics on test data
y_pred = crf.predict(X_test)
print(metrics.flat_classification_report(y_test, y_pred, digits=3))

# Confusion matrix
flat_y_test = [item for sublist in y_test for item in sublist]
flat_y_pred = [item for sublist in y_pred for item in sublist]
print(confusion_matrix(flat_y_test, flat_y_pred))


# Print some results
# for i in range(10):
#     print("real tags : ",y_test[i])
#     print("predicted tags : ",y_pred[i])
