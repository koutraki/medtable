import argparse
import json
from collections import Counter

'''
    Set up the arguments and parse them.
'''


def get_arguments():
    parser = argparse.ArgumentParser(description='Use this script to extract the table data.')
    parser.add_argument('-t', '--tables', help='The tables input file.', required=False)
    parser.add_argument('-o', '--out_file', help='The output file where we store the results', required=True)
    return parser.parse_args()


'''
    Extract the table stats.
'''


def extract_table_stats(table_file):
    fin = open(table_file, 'rt')

    # keep the information about the columns in each table
    col_dist = {}
    table_dist = {}
    table_dist['headers'] = []
    table_dist['cols'] = []

    num_rows = 0
    for idx, line in enumerate(fin):
        if idx == 0:
            continue
        data = line.strip().split('\t')

        entity = data[0]
        section = data[2]

        table_json = json.loads(data[-1])
        num_rows += len(table_json['rows'])

        headers = table_json['header']
        table_dist['headers'].append(len(headers))

        for header in headers:
            columns = header['columns']

            table_dist['cols'].append(len(columns))

            for column in columns:
                col_name = column['name'].lower().strip()
                col_values = len(column['value_dist'])

                if col_name not in col_dist:
                    col_dist[col_name] = {}
                    col_dist[col_name]['entities'] = []
                    col_dist[col_name]['sections'] = []
                    col_dist[col_name]['num_values'] = []
                    col_dist[col_name]['values'] = []

                col_dist[col_name]['entities'].append(entity)
                col_dist[col_name]['sections'].append(section)
                col_dist[col_name]['num_values'].append(col_values)

                for val in column['value_dist']:
                    col_dist[col_name]['values'].append(val['value'].lower().strip() * val['count'])

    return table_dist, col_dist, num_rows


'''
    Aggregate the table stats.
'''


def aggregate_table_stats(table_file, out_file):
    table_dist, col_dist, num_rows = extract_table_stats(table_file)

    col_num_avg = Counter(table_dist['cols'])
    headers = Counter(table_dist['headers'])

    out_str = 'column\tnum_entities\tnum_sections\tavg_num_values\tdistinct_values\tvalue_dist\n'
    for col_label in col_dist:
        col = col_dist[col_label]
        num_entities = len(set(col['entities']))
        num_sections = len(set(col['sections']))
        avg_values = sum(col['num_values']) / len(col['num_values'])

        val_counter = Counter(col['values'])
        val_dist = ';'.join([k+'='+str(v) for k, v in val_counter.items()])
        total_val = len(val_counter.items())
        out_str += '%s\t%d\t%d\t%.3f\t%d\t%s\n' % (col_label, num_entities, num_sections, avg_values, total_val, str(val_dist))
    open(out_file, 'wt').write(out_str)

    print ('The average number of columns is %s' % str(col_num_avg))
    print ('The average number of headers is %s' % str(headers))
    print ('The total number of rows is %d' % num_rows)


if __name__ == '__main__':
    p = get_arguments()

    aggregate_table_stats(p.tables, p.out_file)