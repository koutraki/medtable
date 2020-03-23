import argparse

import gensim.models
import numpy as np
import json
import pickle

import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support, \
    accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

'''
    Set up the arguments and parse them.
'''


def get_arguments():
    parser = argparse.ArgumentParser(description='Use this script to classify tables.')
    parser.add_argument('-t', '--tables', help='The tables input file.', required=False)
    parser.add_argument('-i', '--features', help='The input feature file.', required=False)
    parser.add_argument('-l', '--labels', help='The table labels.', required=False)
    parser.add_argument('-e', '--emb', help='The embedding file.', required=False)
    parser.add_argument('-o', '--out', help='The output file where we store the results', required=False)
    parser.add_argument('-f', '--operation_flag', help='The operation to carry out', required=True)
    return parser.parse_args()


'''
    Return the dictionary of entities, section and the corresponding labels.
'''


def load_gt(labels_file):
    fin = open(labels_file, 'rt')

    labels = {}
    for line in fin:
        # this table
        if line.strip().endswith('--'):
            continue
        data = line.split(',')

        if len(data) < 2:
            print(line)
            continue

        entity = data[0].strip().lower()
        section = data[1].strip().lower()
        label = 'table' in data[-1].strip().lower()

        if entity not in labels:
            labels[entity] = {}
        labels[entity][section] = label
    return labels


'''
    Generate the hash value for a table. We need to remove the table markup in order for the same extracted table
    to have the same hash value. In some cases the tables have conflicting markup, even though they have the same
    table content.
'''


def generate_table_hash(table_str):
    tbl = json.loads(table_str)
    tbl['markup'] = ''
    tbl['id'] = ''

    return hash(json.dumps(tbl))


'''
    Annotate the table instances with their corresponding labels.
'''


def process_instance_labels(tables_file, labels_file, out_file):
    # get the labels
    labels = load_gt(labels_file)

    fin = open(tables_file, 'rt')
    fout = open(out_file, 'wt')

    out_str = ''
    table_instances = {}
    table_instance_labels = {}
    out_val = ''
    for idx, line in enumerate(fin):
        if idx == 0:
            out_str = line.strip() + '\tlabel\n'
            continue

        data = line.strip().split('\t')
        entity = data[0].strip().lower()
        section = data[2].strip().lower()

        tbl_hash = generate_table_hash(data[4].strip())

        out_val += 'entity: %s\t section: %s\t hash: %s\n' % (entity, section, str(tbl_hash))
        label = entity in labels and section in labels[entity]

        if tbl_hash not in table_instance_labels:
            table_instance_labels[tbl_hash] = label
        elif label:
            table_instance_labels[tbl_hash] = label

        table_instances[tbl_hash] = line.strip()

    for tbl_hash in table_instances:
        # for val in table_instances[tbl_hash]:
        out_str += table_instances[tbl_hash] + '\t' + str(table_instance_labels[tbl_hash]) + '\n'

        if len(out_str) > 100000:
            fout.write(out_str)
            out_str = ''
    fout.write(out_str)

    open('data/test_debug.txt', 'wt').write(out_val)


'''
    Compute the average word embedding of this section.
'''


def get_average_emb(text, emb, is_w2v=True, dims=100):
    if is_w2v:
        data = re.sub('[^0-9a-zA-Z]+', ' ', text).lower().strip().split(' ')
    else:
        data = text
    # just in case we have a higher dimensionality than the embedding or vice versa.
    # We cut the embeddings to our desired dimensions, otherwise we cut the embeddings
    emb_0 = emb[list(emb.vocab.keys())[0]]
    emb_dim = dims if len(emb_0) >= dims else len(emb_0)

    # keep the average embeddings here
    avg_emb = np.zeros(shape=(emb_dim), dtype=float)
    for word in data:
        if word in emb.vocab:
            avg_emb += emb.get_vector(word)[:emb_dim]
        else:
            if is_w2v:
                avg_emb += emb.get_vector('unk')[:emb_dim]
            else:
                avg_emb += np.random.rand(emb_dim)

    avg_emb /= len(data)
    return avg_emb


'''
    Get all the column names from this table.
'''


def get_columns(table):
    cols = set()
    try:
        table = json.loads(table)

        for header in table['header']:
            columns = header['columns']

            for column in columns:
                col_name = column['name'].lower().strip()
                cols.add(col_name)
    except Exception as e:
        print(e.message)
    return cols


'''
    Generate the feature files.
'''


def construct_instance_features(tables_file, emb_file, out_dir):
    # load word embeddings first
    glove = gensim.models.KeyedVectors.load_word2vec_format(emb_file, binary=False)

    instances = []
    labels = []
    fin = open(tables_file, 'rt')
    for idx, line in enumerate(fin):
        if idx == 0:
            continue
        data = line.strip().split('\t')

        entity = data[0]
        #  section_level = data[1]
        section_label = data[2].strip().lower()
        section_avg_emb = get_average_emb(data[3], emb=glove)
        table_cols = get_columns(data[4])
        label = data[5]
        section_label_tokens = 'type' in section_label  # or 'classification' in section_label

        # write the features
        inst = {}
        inst['entity'] = entity
        inst['section'] = section_label
        #   inst['section_level'] = section_level
        inst['section_label_tokens'] = section_label_tokens

        for e_idx, emb_val in enumerate(section_avg_emb):
            inst['emb_%d' % e_idx] = emb_val

        for c_idx, col in enumerate(table_cols):
            inst['col_%d' % c_idx] = col

        instances.append(inst)
        labels.append(label)

    # store the section and column dicts
    pickle.dump(instances, open(out_dir + '/features.pckl', 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(labels, open(out_dir + '/inst_labels.pckl', 'wb'), pickle.HIGHEST_PROTOCOL)

    print('Finished constructing the feature file.')


'''
    Perform a k-fold cross validation on the table quality classification problem.
'''


def perform_cross_fold_validation(k, X, y, clf, algoname):
    kf = KFold(n_splits=k)
    avg = lambda scores: sum(scores) / len(scores)
    avgp, avgr, avgf1, avgsum, avg_balance_accuracy, avg_accuracy = [], [], [], [], [], []
    labels = []
    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = [y[i] for i in train_index]
        y_test = [y[i] for i in test_index]

        # evaluate for the current split
        clf.fit(X_train, y_train)
        labels = clf.classes_
        y_pred = clf.predict(X_test)

        p, r, f1, true_sum = precision_recall_fscore_support(y_test, y_pred)
        balance_accuracy = balanced_accuracy_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        avgp.append(p)
        avgr.append(r)
        avgf1.append(f1)
        avg_balance_accuracy.append(balance_accuracy)
        avg_accuracy.append(accuracy)
        avgsum.append(true_sum)

    # the remainder is for printing the evaluation report
    eval_report = {}
    support_scores = []
    for idx, label in enumerate(labels):
        avgp_score, avgr_scores, avg1_scores, avgsum_scores, avg_accuracy_scores, avg_baccuracy_scores \
            = avg(avgp), avg(avgr), avg(avgf1), sum(avgsum), avg(avg_accuracy), avg(avg_balance_accuracy)

        eval_report[label] = {}
        eval_report[label]['precision'] = avgp_score[idx]
        eval_report[label]['recall'] = avgr_scores[idx]
        eval_report[label]['f1-score'] = avg1_scores[idx]
        eval_report[label]['support'] = avgsum_scores[idx]
        eval_report[label]['accuracy'] = avg_accuracy_scores
        eval_report[label]['balance_accuracy'] = avg_balance_accuracy
        support_scores.append(avgsum_scores[idx])

    last_line_heading = 'avg / total'
    name_width = max(len(cn) for cn in labels)
    headers = ["precision", "recall", "f1-score", "support", "accuracy", "balance_accuracy"]
    width = max(name_width, len(last_line_heading), 2)

    head_fmt = u'Classification Report for classifier based on %s\n' % algoname
    head_fmt += u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'
    for label in eval_report:
        row = [label, eval_report[label]['precision'], eval_report[label]['recall'], eval_report[label]['f1-score'],
               eval_report[label]['support']]
        report += row_fmt.format(*row, width=width, digits=2)
    report += u'\n'
    report += row_fmt.format(last_line_heading, np.average(p, weights=support_scores), np.average(r, weights=support_scores), np.average(f1, weights=support_scores), np.sum(support_scores), width=width, digits=2)

    report += 'Average Acc:%.2f\tBalanced Acc:%.2f' % (np.average(accuracy), np.average(balance_accuracy))
    return report


'''
    Return the feature indices for the entities and sections from the DictVectorizer object.
'''


def get_entity_section_names(dict_vectorizer):
    features = dict_vectorizer.get_feature_names()

    f_entity = {}
    f_section = {}
    for idx, fname in enumerate(features):
        if 'entity=' in fname:
            f_entity[idx] = fname
        elif 'section=' in fname:
            f_section[idx] = fname
    return f_entity, f_section


'''
    Get the entity and section name of an instance.
'''


def get_instance_entity_section(inst, entity_dict, section_dict):
    entity = ''
    section = ''
    for idx, val in enumerate(inst):
        if idx in entity_dict and val != 0:
            entity = entity_dict[idx]
        elif idx in section_dict and val != 0:
            section = section_dict[idx]

    entity = re.sub('entity=', '', entity)
    section = re.sub('section=', '', section)
    return entity, section


'''
    Train a RF model.
'''


def train(in_dir):
    # load the features
    v = DictVectorizer(sparse=False)
    features = pickle.load(open(in_dir + '/features.pckl', 'rb'))
    y = pickle.load(open(in_dir + '/inst_labels.pckl', 'rb'))

    X = v.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    clf_lr = LogisticRegression()
    clf_lr.fit(X_train, y_train)

    # evaluate on the test data
    y_pred_lr = clf_lr.predict(X_test)

    print('Logistic Regression\n' + classification_report(y_test, y_pred_lr))

    # cross fold validation
    clf = LogisticRegression()
    print(perform_cross_fold_validation(3, X, y, clf, 'LR'))


if __name__ == '__main__':
    p = get_arguments()

    if p.operation_flag == 'parse_instances':
        process_instance_labels(p.tables, p.labels, p.out)
    elif p.operation_flag == 'features':
        construct_instance_features(p.tables, p.emb, p.out)
    elif p.operation_flag == 'train':
        train(p.features)
