from csv import DictReader

import pandas as pd
import csv

def parse_csv_file(filepath, column_name, isURLFormat=False):
    df = pd.read_csv(filepath)
    x = df[column_name]
    z = x.get_values()
    articles = set()
    for y in z:
        if isURLFormat:
            y = y.split("/")[-1]
        articles.add(y.lower())
    return articles


def parse_tsv_file(filepath):
    with open(filepath) as tsvfile:
        reader = csv.DictReader(tsvfile, dialect='excel-tab')  # type: DictReader
        diseases = set()
        for row in reader:
            diseases.add(row['Name'].lower())
    return diseases


def overlap_tables_wikidata_diseases():
    articles = parse_csv_file(column_name='itemLabel', filepath='/home/mary/Documents/bioNLP2019/wikidata_diseases.csv')
    tablenet_entities = set([entity.strip().lower() for entity in open('/home/mary/Documents/bioNLP2019/table_entities.txt', 'rt').readlines()])

    overlap = tablenet_entities & articles
    print (len(overlap))
    overlap_str = '\n'.join(overlap)
    open('/home/mary/Documents/bioNLP2019/wikidata_tablenet_entities_overlap.txt', 'wt').write(overlap_str)

def overlap_tables_DO_diseases():
    diseases_DO = parse_tsv_file("/home/mary/Documents/bioNLP2019/DO_diseases/D-DoMiner_miner-diseaseDOID.tsv")
    tablenet_entities = set([entity.strip().lower() for entity in open('/home/mary/Documents/bioNLP2019/table_entities.txt', 'rt').readlines()])
    overlap = tablenet_entities & diseases_DO
    print (len(overlap))
    overlap_str = '\n'.join(overlap)
    open('/home/mary/Documents/bioNLP2019/DO_tablenet_entities_overlap.txt', 'wt').write(overlap_str)
