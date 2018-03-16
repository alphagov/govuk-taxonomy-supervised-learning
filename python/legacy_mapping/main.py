import ijson
import gzip
import progressbar
import csv
import sys

from io import StringIO
from itertools import chain, groupby, islice
from data import document_types_excluded_from_the_topic_taxonomy

EXCLUDED_DOCS = document_types_excluded_from_the_topic_taxonomy()
LEGACY_TAGS = ['policy_areas', 'topics', 'policies', 'mainstream_browse_pages']

def legacy_taxon_to_topic_taxons(items):
	mapping = {}
	progress_bar = progressbar.ProgressBar()

	for item in progress_bar(items):
		if item['document_type'] in EXCLUDED_DOCS:
			continue

		for legacy_taxon in __extract_legacy_taxons(item):
			if len(legacy_taxon) == 0: continue

			if not legacy_taxon in mapping: 
				mapping[legacy_taxon] = []

			topic_taxon_set = __extract_topic_taxons(item)
			topic_taxon_set = __filter_topic_taxons(topic_taxon_set)
			if len(topic_taxon_set) == 0: continue
			
			mapping[legacy_taxon].append(topic_taxon_set)

	return mapping

def convert_to_scored_hashes(mapping):
	scores = []

	for legacy_taxon, topic_taxon_sets in mapping.items():
		topic_taxons = __flatten(topic_taxon_sets)
		topic_taxons.sort()

		if len(topic_taxons) == 0:
			scores.append({'legacy_taxon_path': legacy_taxon,
				           'topic_taxon_path': 'null',
				           'topic_taxon_level': 0,
				           'mapping_count': 0,
				           'mapping_share': 0})

		for topic_taxon, dups in groupby(topic_taxons):
			mapping_count = len(list(dups))
			mapping_share = mapping_count / len(topic_taxon_sets)

			scores.append({'legacy_taxon_path': legacy_taxon,
				 		   'topic_taxon_path': topic_taxon[1],
				 		   'topic_taxon_level': topic_taxon[0] + 1,
				 		   'mapping_count': mapping_count,
				 		   'mapping_share': mapping_share})

	return scores

def __extract_legacy_taxons(item):
	tags = []
	links = item['links']

	for area in LEGACY_TAGS:
		if area in item['links']:
			base_paths = map(lambda link: link['base_path'], links[area])
			tags.extend(map(lambda path: area + '[' + path + ']', base_paths))

	return tags

def __extract_topic_taxons(item):
	links = item['links']

	if 'taxons' in item['links']:
		traces = map(__trace_topic_taxon, links['taxons'])
		return list(set(__flatten(traces)))

	return []

def __trace_topic_taxon(link):
	taxons = [link]

	while 'parent_taxons' in link['links']:
		taxons.append(link['links']['parent_taxons'][0])
		link = link['links']['parent_taxons'][0]

	return enumerate(map(lambda taxon: taxon['base_path'], taxons[::-1]))

def __filter_topic_taxons(taxons):
	return [taxon for taxon in taxons if not taxon[1].startswith('/imported')]

def __flatten(items):
	return list(chain.from_iterable(items))

def __convert_to_csv(hashes):
	output = StringIO()
	writer = csv.DictWriter(output, fieldnames = hashes[0].keys())
	writer.writeheader()
	[writer.writerow(hash) for hash in hashes]
	return output.getvalue()

def __mapping_quality(item):
	return item['mapping_count'] * item['mapping_share'] * item['topic_taxon_level']

file = gzip.open(sys.argv[1])
gen = ijson.items(file, prefix='item')
gen = islice(gen, 10)

mapping = legacy_taxon_to_topic_taxons(gen)
scores = convert_to_scored_hashes(mapping)
scores.sort(key = __mapping_quality)
print(__convert_to_csv(scores))
