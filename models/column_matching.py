import json
import argparse
from collections import defaultdict

import gensim
import re
import numpy as np
from scipy import spatial
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from table_classifier import get_average_emb
from models.table_classifier import perform_cross_fold_validation
'''
    Check if the column contains structured values
'''


def has_structured_vals(col_idx, table):
    for row in table['rows']:
        for col_idx_cmp, cell_vals in enumerate(row['values']):
            if 'structured_values' in cell_vals and len(cell_vals['structured_values']) != 0 and col_idx_cmp == col_idx:
                return True
    return False


'''
    Set up the arguments and parse them.
'''


def get_arguments():
    parser = argparse.ArgumentParser(description='Use this script to extract the table data.')
    parser.add_argument('-t', '--tables', help='The tables input file.', required=False)
    parser.add_argument('-o', '--out_dir', help='The output file where we store the results', required=False)
    parser.add_argument('-e', '--emb', help='The embedding file.', required=False)
    parser.add_argument('-g', '--graph_emb', help='The graph embedding file.', required=False)
    parser.add_argument('-d', '--dim', help='The desired dimensions we want to analyze from the embeddings.', required=False, type=int)
    parser.add_argument('-f', '--flag', help='The operation that we will conduct in this class.')
    parser.add_argument('-i', '--input', help='The path to the input file that we will use to process.')
    parser.add_argument('-gt', '--ground_truth', help='The path to the ground-truth.')
    parser.add_argument('-p', '--pairs', help='The path to all the ground-truth pairs.')
    parser.add_argument('-cs', '--col_pair_sim', help='The path to the column pair similarity.')
    return parser.parse_args()


def get_table_columns(table_file):
    fin = open(table_file, 'rt')

    # keep the information about the columns in each table
    table_dist = {}
    table_dist['headers'] = []
    table_dist['cols'] = []
    all_columns = {}

    for idx, line in enumerate(fin):
        if idx == 0:
            continue
        data = line.strip().split('\t')

        entity = data[0]
        section = data[2]
        table_json = json.loads(data[4])
        table_id = table_json['id']

        all_columns[table_id] = {}

        headers = table_json['header']
        if data[5] == 'True':
            # Only in the positive tables
            for header in headers:
                columns = header['columns']
                for i, column in enumerate(columns):
                    col_name = column['name'].lower().strip()

                    # check if the column contains structured values
                    has_struct_vals = has_structured_vals(i, table_json)

                    all_columns[table_id][col_name] = (i, column, has_struct_vals, entity, section)

    return all_columns


'''
    This gets the column header word embedding representation. The representation is computed as the average of the word 
     embeddings in the column header.
'''


def compute_column_embeddings(all_columns, emb_dim, emb_file, out_dir):
    if os.path.exists(out_dir + '/col_header_w2v_rep.pck'):
        return pickle.load(open(out_dir + '/col_header_w2v_rep.pck', 'rb'))
    # load word embeddings first
    glove = gensim.models.KeyedVectors.load_word2vec_format(emb_file, binary=False)

    col_header_rep = {}
    for table_id in all_columns:
        col_header_rep[table_id] = {}
        for col_name in all_columns[table_id]:
            col_name = re.sub('[^0-9a-zA-Z]+', ' ', col_name).lower().strip()
            col_header_rep[table_id][col_name] = get_average_emb(text=col_name, dims=emb_dim, emb=glove, is_w2v=True)

    # store the rep
    pickle.dump(col_header_rep, open(out_dir + '/col_header_w2v_rep.pck', 'wb'), pickle.HIGHEST_PROTOCOL)
    return col_header_rep


'''
    This gets the column header word embedding representation. The representation is computed as the average of the word 
     embeddings in the column header.
'''


def get_column_instance_value_embeddings(all_columns, emb_file, emb_dim=100, out_dir='.'):
    if os.path.exists(out_dir + '/col_n2v_rep.pck'):
        return pickle.load(open(out_dir + '/col_n2v_rep.pck', 'rb'))
    # load word embeddings first
    n2v = gensim.models.KeyedVectors.load_word2vec_format(emb_file, binary=False)

    col_header_rep = {}
    for table_id in all_columns:
        col_header_rep[table_id] = {}
        for col_name in all_columns[table_id]:
            column = all_columns[table_id][col_name]

            # in case the column doesnt have any structured values.
            if not column[2]:
                col_header_rep[table_id][col_name] = np.random.rand(emb_dim)

            col_values = set()
            for value in column[1]['value_dist']:
                val = re.sub(' ', '_', value['value']).strip().lower()
                if len(val):
                    col_values.add(val)
            if len(col_values):
                col_header_rep[table_id][col_name] = get_average_emb(text=col_values, emb=n2v, is_w2v=False, dims=emb_dim)
            else:
                col_header_rep[table_id][col_name] = np.random.rand(emb_dim)

    # store the rep
    pickle.dump(col_header_rep, open(out_dir + '/col_n2v_rep.pck', 'wb'), pickle.HIGHEST_PROTOCOL)
    return col_header_rep


'''
    Compute a the probability distribution of each value occurring in a column. In many cases the probability  will
    be uniformly distributed. In such cases we can just use jaccard similarity over column values.
'''


def get_unstructured_col_val_rep(all_columns, out_dir):
    if os.path.exists(out_dir + '/col_unstructured_rep.pck'):
        return pickle.load(open(out_dir + '/col_unstructured_rep.pck', 'rb'))
    col_header_rep = {}
    for table_id in all_columns:
        col_header_rep[table_id] = {}
        for col_name in all_columns[table_id]:
            column = all_columns[table_id][col_name]
            col_val_dist = {}

            # in case the column doesnt have any structured values.
            if col_name not in col_header_rep[table_id]:
                col_header_rep[table_id][col_name] = col_val_dist

            sum_ = 0
            for value in column[1]['value_dist']:
                val = value['value'].strip().lower()
                if len(val):
                    sum_ += value['count']
                    col_val_dist[val] = value['count']
            col_val_dist = {k: v / float(sum_) for k, v in col_val_dist.items()}

            col_header_rep[table_id][col_name] = col_val_dist
    # store the rep
    pickle.dump(col_header_rep, open(out_dir + '/col_unstructured_rep.pck', 'wb'), pickle.HIGHEST_PROTOCOL)
    return col_header_rep


'''
    Compute the KL Divergence between two probability distributions
'''


def compute_kl_div(a_dict, b_dict):
    kl_div = 0
    for k, v in a_dict.items():
        if k not in b_dict:
            continue
        q_v = b_dict[k]

        kl_div += v * np.math.log(v / q_v)
    return kl_div


'''
    Compute the jaccard similarity.
'''


def compute_jaccard_sim(set_a, set_b):
    overlap = set_a & set_b
    all_vals = set_a | set_b

    if len(all_vals) == 0:
        return 0
    return len(overlap) / float(len(all_vals))


'''
    Compute the different similarities between a pair of columns. We can weigh the similarities based on some
    parameter set \lambda_1 \lambda_2 \lambda_1, e.g. sim = \lambda_1 * w2v_sim + \lambda_2 * n2v + \lambda_3 * unstructured
'''


def compute_column_similarity(col_a, col_b):
    w2v_sim = 1 - spatial.distance.cosine(col_a['w2v'], col_b['w2v'])
    n2v_sim = 1 - spatial.distance.cosine(col_a['n2v'], col_b['n2v'])

    # for the unstructured value representation, check if the values are uniformly distributed, if yes, use jaccard
    # is_uniform_prob_dist = set(col_a['unstructured'].values()) == 1 or set(col_b['unstructured'].values()) == 1
    jaccard_unstruct_sim = compute_jaccard_sim(col_a['unstructured'].keys(), col_b['unstructured'].keys())
    kl_unstruct_sim = compute_kl_div(col_a['unstructured'], col_b['unstructured'])

    return w2v_sim, n2v_sim, jaccard_unstruct_sim, kl_unstruct_sim


'''
    Merge the different column representations for easier access.
'''


def merge_all_col_reps(col_w2v, col_n2v, col_unstruct, all_cols):
    all_reps = {}
    for tbl_id in all_cols:
        all_reps[tbl_id] = {}
        for col_name in all_cols[tbl_id]:
            all_reps[tbl_id][col_name] = {}
            all_reps[tbl_id][col_name]['w2v'] = col_w2v[tbl_id][col_name]
            all_reps[tbl_id][col_name]['n2v'] = col_n2v[tbl_id][col_name]
            all_reps[tbl_id][col_name]['unstructured'] = col_unstruct[tbl_id][col_name]

    return all_reps


def output_column_similarity(all_reps):
    similiarities = open('data/column_similarities.txt', 'wt')
    out_str = ''
    for tbl_id in all_reps:
        for col_name in all_reps[tbl_id]:
            col_a = all_reps[tbl_id][col_name]
            for tbl_id_b in all_reps:
                if tbl_id == tbl_id_b:
                    continue
                for col_name_b in all_reps[tbl_id_b]:
                    col_b = all_reps[tbl_id_b][col_name_b]
                    w2v_sim, n2v_sim, jaccard_unstruct_sim, kl_unstruct_sim = compute_column_similarity(col_a, col_b)
                    out_str += '%s\t%s\t%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\n' % (str(tbl_id), str(tbl_id_b), col_name, col_name_b, w2v_sim, n2v_sim, jaccard_unstruct_sim, kl_unstruct_sim)

                    if len(out_str) > 100000:
                        # similiarities.write(out_str.encode('utf-8'))
                        similiarities.write(out_str)
                        out_str = ''
    # similiarities.write(out_str.encode('utf-8'))
    similiarities.write(out_str)


'''
    Get the column pair similarities.
'''


def get_col_pair_sims(in_file):
    # read the column pairs
    fin = open(in_file, 'rt')

    # column pairs
    col_pairs = {}
    col_pairs['w2v'] = {}
    col_pairs['n2v'] = {}
    col_pairs['kl'] = {}
    col_pairs['jacc'] = {}
    col_pairs['combined'] = {}
    col_pairs['combined_n2v_w2v'] = {}

    for line in fin:
        data = line.strip().split('\t')

        w2v_sim = float(data[4])
        n2v_sim = float(data[5])
        jacc = float(data[6])
        kl = float(data[7])

        combined = (w2v_sim + n2v_sim + jacc + kl) / 4.0
        combined_n2v_w2v = (w2v_sim + n2v_sim) / 2.0
        if w2v_sim not in col_pairs['w2v']:
            col_pairs['w2v'][w2v_sim] = []
        if n2v_sim not in col_pairs['n2v']:
            col_pairs['n2v'][n2v_sim] = []
        if jacc not in col_pairs['jacc']:
            col_pairs['jacc'][jacc] = []
        if kl not in col_pairs['kl']:
            col_pairs['kl'][kl] = []
        if combined not in col_pairs['combined']:
            col_pairs['combined'][combined] = []
        if combined_n2v_w2v not in col_pairs['combined_n2v_w2v']:
            col_pairs['combined_n2v_w2v'][combined_n2v_w2v] = []

        col_pairs['w2v'][w2v_sim].append(line)
        col_pairs['n2v'][n2v_sim].append(line)
        col_pairs['jacc'][jacc].append(line)
        col_pairs['kl'][kl].append(line)
        col_pairs['combined'][combined].append(line)
        col_pairs['combined_n2v_w2v'][combined_n2v_w2v].append(line)
    return col_pairs


'''
    Get the column pair similarities.
'''


def load_col_pair_sims(in_file):
    # read the column pairs
    fin = open(in_file, 'rt')

    # column pairs
    col_pairs = defaultdict(list)
    for line in fin:
        data = line.strip().split('\t')

        tbl_id_a = data[0]
        tbl_id_b = data[1]

        col_a = data[2]
        col_b = data[3]

        # we need to construct an invariant ID for the pairs
        str_id = tbl_id_a + tbl_id_b + col_a + col_b
        id_ = sum(ord(c) for c in str_id)

        col_pair_sim = {}
        col_pair_sim['tables'] = [tbl_id_a, tbl_id_b]
        col_pair_sim['cols'] = [col_a, col_b]
        col_pair_sim['sim'] = list(map(float, data[4:]))
        col_pair_sim['sim'].extend([np.mean(col_pair_sim['sim']), np.mean(col_pair_sim['sim'][:2])])

        col_pairs[id_].append(col_pair_sim)

    return col_pairs


'''
    Generate the samples per each of the similarity buckets. 
'''


def sample_columns_for_labelling(in_file, num_samples, sim_tag):
    bins = np.array([0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])

    # column pairs
    col_pairs = get_col_pair_sims(in_file)

    # get the samples per bin
    if sim_tag not in col_pairs:
        return None

    col_pairs_sub = col_pairs[sim_tag]
    col_pairs_sub_vals = list(col_pairs_sub.keys())
    bins_val = np.digitize(col_pairs_sub_vals, bins)

    # merge the lines into the buckets
    merged_lines = {k: [] for k in bins}
    for bin_idx, bin in enumerate(bins):
        bins_val_sub = [idx for idx, val in enumerate(bins_val) if val == bin_idx]
        for indice in bins_val_sub:
            key_val = col_pairs_sub_vals[indice]
            merged_lines[bin].extend(col_pairs[sim_tag][key_val])

    samples = {}
    for bin in merged_lines:
        num_samples = num_samples if len(merged_lines[bin]) > num_samples else len(merged_lines[bin])
        sampled_vals = np.random.choice(len(merged_lines[bin]), num_samples, replace=False)

        samples[bin] = []
        for val in sampled_vals:
            samples[bin].append(merged_lines[bin][val])

    return samples


def load_table_information(table_file):
    tables = {}
    fin = open(table_file, 'rt')
    for idx, line in enumerate(fin):
        if idx == 0:
            continue
        data = line.strip().split('\t')
        # entity	section_level	section	section_text	table_json	label
        tbl_json = json.loads(data[4])

        tbl_id = tbl_json['id']
        tables[tbl_id] = (data[0], data[1], data[2], data[3], tbl_json)
    return tables


'''
    Write the column pair data for GT evaluation.
'''


def write_column_pair_gt_data(table_file, col_samples, out_dir):
    tables = load_table_information(table_file)

    # str(tbl_id), str(tbl_id_b), col_name, col_name_b, w2v_sim, n2v_sim,
    fout = open(out_dir + '/col_pairs_gt.html', 'wt')
    out_str = '<html><style>\ntable {font-family: arial, sans-serif; border-collapse: collapse; width: 50%;}\ntr, td, th {border: 2px solid; padding:0 5px 0 5px; vertical-align:top}\ntr:nth-child(even) {background-color: #dddddd;}</style>'
    out_str += '<script src="https://code.jquery.com/jquery-1.12.4.min.js"></script><script>$(document).ready(function() {$("button").click(function(){var favorite = [];$.each($("input[name=\'gt_val\']:checked"), function(){favorite.push($(this).val());});$("textarea#ExampleMessage").val(favorite.join("\\n"));});});</script>'

    count = 0
    for bucket in col_samples:
        for sample in col_samples[bucket]:
            count += 1
            data = sample.strip().split('\t')
            tbl_a = int(data[0])
            tbl_b = int(data[1])

            col_a = data[2]
            col_b = data[3]

            table_a = tables[tbl_a]
            table_b = tables[tbl_b]

            table_out = '<table><caption>Similarity %.2f</caption><tr><td>Table A: %s</td><td>Table B: <strong>%s</strong></td><td><strong>Relevant</strong></td></tr>' % (
                bucket, table_a[0], table_b[0])
            table_out += '<tr style="vertical-align:top"><td align="left">Section A: <strong>%s</strong></td><td align="right">Section B: <strong>%s</strong></td></tr>' % (table_a[2], table_b[2])
            table_out += '<tr style="vertical-align:top"><td align="left">Column A: <strong>%s</strong></td><td align="right">Column B: <strong>%s</strong></td></tr>' % (col_a, col_b)
            table_out += '<tr style="vertical-align:top"><td style="text-align:left; vertical-align:top; border="1px">%s</td><td style="text-align:right; vertical-align:top; border="1px">%s</td>' % (
                re.sub('\n+', ' ', table_a[4]['markup']).strip(), re.sub('\n+', ' ', table_b[4]['markup']).strip())

            tbl_id_a_b = str(tbl_a) + ';' + str(tbl_b) + ';' + table_a[0] + ';' + table_b[0] + ';' + table_a[2] + ';' + table_b[2] + ';' + col_a + ';' + col_b
            table_out += '<td><input type="checkbox" id="%s" name="gt_val" value="%s"><label for="%s">Align</label></td></tr>' % (tbl_id_a_b, tbl_id_a_b, tbl_id_a_b)
            table_out += '</table>\n'

            out_str += re.sub('\n+', '<br/> ', table_out)

            fout.write(out_str)
            out_str = ''
    out_str = '<button type="button">Store Values</button>'
    out_str += '<textarea disabled id="ExampleMessage" rows="4" cols="50"></textarea>'
    out_str += '</body></html>'
    fout.write(out_str)

    # store the samples too
    pickle.dump(col_samples, open(out_dir + '/col_gt_pair_samples.pck', 'wb'), pickle.HIGHEST_PROTOCOL)

    print('Sampled %d instances ' % count)


def __load_column_align_gt(ground_truth):
    f = open(ground_truth, 'rt', encoding='utf8')
    gt = defaultdict(list)
    for line in f:
        data = line.strip().split(';')
        tbl_a = data[0]
        tbl_b = data[1]

        col_a = data[-2]
        col_b = data[-1]

        # we need to construct an invariant ID for the pairs
        str_id = tbl_a + tbl_b + col_a + col_b
        id_ = sum(ord(c) for c in str_id)

        # we add the pairs in both directions <col_a, col_b> and <col_b, col_a> for easier access
        gt[id_].append((tbl_a, tbl_b, col_a, col_b))
    return gt


'''
    Construct a classification model for evaluation purposes.
'''


def get_model(model='rf', random_state=0):
    if model == 'rf':
        clf = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=random_state, n_jobs=20)
    elif model == 'lr':
        clf = LogisticRegression(random_state=random_state)
    elif model == 'svm':
        clf = LinearSVC(random_state=0, tol=1e-5)
    elif model == 'mlp':
        clf = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100), activation='relu', solver='adam', learning_rate=0.001, max_iter=1000)
    return clf


'''
    Train and test the model.
'''


def train_column_alignment_model(column_data, column_labels, model_type='rf', train_ratio=0.5, random_seed=0):
    # determine the train and test split
    X_train, X_test, y_train, y_test = train_test_split(column_data, column_labels, test_size=(1 - train_ratio), random_state=random_seed)
    clf = get_model(model_type, random_seed)

    clf.fit(X_train, y_train)

    y_pred_rf = clf.predict(X_test)
    report = classification_report(y_test, y_pred_rf, digits=3)
    print(report)

    # cross fold validation
    clf = get_model(model_type, random_seed)
    print(perform_cross_fold_validation(5, X, y, clf, 'LR'))


'''
    Load only the column similarities for the columns that are in our ground-truth.
'''


def _load_col_pairs_similarity(col_similarity_path, col_pairs_eval):
    col_pairs = load_col_pair_sims(col_similarity_path)

    # return the subset that matches our ground-truth
    gt_pairs = __load_column_align_gt(col_pairs_eval)
    col_pairs_sub = defaultdict(list)
    count = 0
    for pair_id in gt_pairs:
        if pair_id in col_pairs:
            pairs_sub = col_pairs[pair_id]
            for pair_gt_sub in gt_pairs[pair_id]:
                tbl_a_gt, tbl_b_gt, col_a_gt, col_b_gt = pair_gt_sub
                # match now the column similarities
                for pair_sim_sub in pairs_sub:
                    tbl_ids = pair_sim_sub['tables']
                    col_ids = pair_sim_sub['cols']

                    if tbl_a_gt == tbl_ids[0] and tbl_b_gt == tbl_ids[1] and col_a_gt == col_ids[0] and col_b_gt == col_ids[1]:
                        col_pairs_sub[pair_id].append(pair_sim_sub)
                        count += 1

    print(count)
    return col_pairs_sub


'''
    Construct training data for the column alignment task.
'''


def construct_column_alignment_train_data(input, ground_truth, col_similarity, col_pairs):
    # load the ground-truth
    gt = __load_column_align_gt(ground_truth)
    samples = _load_col_pairs_similarity(col_similarity, col_pairs)

    # load the tables
    tables = load_table_information(input)

    # for each column pair generate the training data
    num_instances = sum(len(samples[id_]) for id_ in samples)
    X = np.zeros(shape=(num_instances, 6))
    y = []

    counter = 0
    for col_pair_id in samples:
        for sample in samples[col_pair_id]:
            # extract the column features
            X[counter] = sample['sim']  # w2v; n2v; jaccard; kl; all_mean; w2v+n2v mean

            label = True if col_pair_id in gt else False
            y.append(label)
            counter += 1

    return X, y


if __name__ == '__main__':
    p = get_arguments()

    if p.flag == 'column_similarities':
        all_columns = get_table_columns(p.tables)

        col_header_rep_w2v = compute_column_embeddings(all_columns=all_columns, emb_dim=int(p.dim), out_dir=p.out_dir, emb_file=p.emb)
        col_value_rep_n2v = get_column_instance_value_embeddings(all_columns=all_columns, emb_file=p.graph_emb, out_dir=p.out_dir, emb_dim=int(p.dim))
        col_unstruct_rep = get_unstructured_col_val_rep(all_columns, p.out_dir)

        # the merged representations.
        all_reps = merge_all_col_reps(col_w2v=col_header_rep_w2v, col_n2v=col_value_rep_n2v, col_unstruct=col_unstruct_rep, all_cols=all_columns)

        output_column_similarity(all_reps)

    elif p.flag == 'column_gt':
        col_pair_samples = sample_columns_for_labelling(p.input, 50, 'combined_n2v_w2v')
        # output the data for evaluation
        write_column_pair_gt_data(p.tables, col_pair_samples, p.out_dir)

    elif p.flag == 'align':
        X, y = construct_column_alignment_train_data(p.input, p.ground_truth, p.col_pair_sim, p.pairs)
        train_column_alignment_model(X, y, 'lr')
