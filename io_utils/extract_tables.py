import argparse
import gzip
import json
import re

from bs4 import BeautifulSoup

'''
    Set up the arguments and parse them.
'''


def get_arguments():
    parser = argparse.ArgumentParser(description='Use this script to extract the table data.')
    parser.add_argument('-t', '--tables', help='The tables input file.', required=False)
    parser.add_argument('-s', '--seeds', help='The entity seeds.', required=False)
    parser.add_argument('-o', '--out_file', help='The output file where we store the results', required=True)
    parser.add_argument('-f', '--operation_flag', help='The operation that we will carry out in this script', required=True)
    parser.add_argument('-w', '--wiki_content', help='The wikipedia content of the tables.')
    return parser.parse_args()


'''
    Extract the tables.
'''


def extract_disease_tables(table_file, out_file, seeds_path):
    # read the entity list that contain tables and are of type 'Disease'
    entities = set([re.sub('_', ' ', e.lower()).strip() for e in open(seeds_path, 'rt').readlines()])

    fin = gzip.open(table_file, 'rt')
    fout = open(out_file, 'wt')

    out_str = ''
    counter = 0
    for line in fin:
        if len(entities) == 0:
            break
        table_json = json.loads(line)
        entity_label = re.sub('_', ' ', table_json['entity']).strip().lower()

        if entity_label in entities:
            print ('Extracting table data from entity %s' % entity_label)
            counter += 1
            out_str += line

            # remove the entity that we've matched
            entities.remove(entity_label)

        if len(out_str) > 100000:
            fout.write(out_str)
            out_str = ''
    fout.write(out_str)

    print ('Finished extracting tables. We have %d entities of type Disease that have a table' % counter)


'''
    Extract the section text from the entities.
'''


def get_wiki_content(wiki_content):
    entities = {}

    fin = gzip.open(wiki_content, 'rt')
    for line in fin:
        data = line.split('\t')

        entity_label = data[0].lower().strip()
        html_content = data[1]

        html_data = BeautifulSoup(html_content, features='lxml')
        sections = html_data.find_all('section')

        entities[entity_label] = {}

        for idx, section in enumerate(sections):
            if section.find('h2') or section.find('h3') or section.find('h4') or section.find('h5'):
                section_label = ''
                section_text = ''
                section_level = 0
                if section.find('h2'):
                    section_label = section.find('h2').text.lower().strip()
                    section_text = section.text
                    section_level = 1
                elif section.find('h3'):
                    section_label = section.find('h3').text.lower().strip()
                    section_text = section.text
                    section_level = 2
                elif section.find('h4'):
                    section_label = section.find('h4').text.lower().strip()
                    section_text = section.text
                    section_level = 3
                elif section.find('h5'):
                    section_label = section.find('h5').text.lower().strip()
                    section_text = section.text
                    section_level = 4

                entities[entity_label][section_label] = (section_level, section_text)
            elif idx == 0:
                entities[entity_label]['MAIN'] = (0, section.text)

    return entities


'''
    Get a specific section from an entity.
'''


def get_section_text(entity_sections, entity, section):
    if entity not in entity_sections or section not in entity_sections[entity]:
        return 0, 'N/A'

    section_level = entity_sections[entity][section][0]
    section_text = entity_sections[entity][section][1]
    section_text = re.sub('\t', ' ', section_text).strip()
    section_text = re.sub('\n', ' ', section_text).strip()
    return section_level, section_text


'''
    Flatten the table information for each entity, by extracting each table in a single row
    with the section information, section text, and the table information in json.
'''


def flatten_tables(table_file, out_file, wiki_content):
    entity_sections = get_wiki_content(wiki_content)

    fin = open(table_file, 'rt')
    fout = open(out_file, 'wt')

    out_str = 'entity\tsection_level\tsection\tsection_text\ttable_json\n'
    counter = 0
    for line in fin:
        table_json = json.loads(line)
        entity_label = re.sub('_', ' ', table_json['entity']).lower().strip()

        if len(table_json['sections']) == 0:
            continue

        for section in table_json['sections']:
            section_label = section['section'].lower().strip()

            if len(section['tables']) == 0:
                continue

            for table in section['tables']:
                section_level, section_text = get_section_text(entity_sections, entity_label, section_label)
                out_str += entity_label + '\t' + str(section_level) + '\t' + section_label + '\t' +  section_text + '\t' + json.dumps(table) + '\n'

                counter += 1

        # flush the output
        if len(out_str) > 100000:
            fout.write(out_str)
            out_str = ''

    fout.write(out_str)

    print ('We have written %d tables from the entities of type Diesease.' % counter)


def extract_html_content(seeds_path, tables_file, out_file):
    # read the entity list that contain tables and are of type 'Disease'
    entities = set([re.sub('_', ' ', e.lower()).strip() for e in open(seeds_path, 'rt').readlines()])

    fin = gzip.open(tables_file, 'rt')
    fout = open(out_file, 'wt')
    outstr = ''
    for line in fin:
        data = line.split('\t')[0]
        if data.lower().strip() in entities:
            outstr += line

        if len(outstr) > 100000:
            fout.write(outstr)
            outstr = ''
    fout.write(outstr)


if __name__ == '__main__':
    p = get_arguments()

    if p.operation_flag == 'extract_tables':
        extract_disease_tables(p.tables, p.out_file, p.seeds)

    elif p.operation_flag == 'flatten_tables':
        flatten_tables(p.tables, p.out_file, p.wiki_content)

    elif p.operation_flag == 'extract_content':
        extract_html_content(p.seeds, p.tables, p.out_file)
