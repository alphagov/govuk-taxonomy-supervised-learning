import ijson
import json
import gzip
import os
import progressbar
import csv

from io import StringIO
from itertools import chain, groupby, islice
from data import document_types_excluded_from_the_topic_taxonomy

DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/content.json.gz')
EXCLUDED_DOCS = document_types_excluded_from_the_topic_taxonomy()
LEGACY_TAGS = ['policy_areas', 'topics', 'policies', 'mainstream_browse_pages']

def legacy_taxon_to_topic_taxons(items):
	mapping = {}
	progress_bar = progressbar.ProgressBar()

	for item in progress_bar(items):
		if item["document_type"] in EXCLUDED_DOCS:
			continue

		for legacy_taxon in extract_legacy_taxons(item):
			if len(legacy_taxon) == 0: continue

			if not legacy_taxon in mapping: 
				mapping[legacy_taxon] = []

			topic_taxon_set = _filter_imported(extract_topic_taxons(item))
			if len(topic_taxon_set) == 0: continue
			
			mapping[legacy_taxon].append(topic_taxon_set)

	return mapping

def convert_to_scored_hashes(mapping):
	scores = []

	for legacy_taxon, topic_taxon_sets in mapping.items():
		topic_taxons = _flatmap(lambda x: x, topic_taxon_sets)
		topic_taxons.sort()

		if len(topic_taxons) == 0:
			scores.append({"legacy_taxon": legacy_taxon,
				           "topic_taxon": "null",
				           "mapping_count": 0,
				           "mapping_share": 0})

		for topic_taxon, dups in groupby(topic_taxons):
			dups_len = len(list(dups))
			scores.append({"legacy_taxon": legacy_taxon,
				 		   "topic_taxon": topic_taxon,
				 		   "mapping_count": dups_len,
				 		   "mapping_share": dups_len / len(topic_taxon_sets)})

	return scores

def extract_legacy_taxons(item):
	tags = []
	links = item['links']

	for area in LEGACY_TAGS:
		if area in item['links']:
			base_paths = map(lambda link: link['base_path'], links[area])
			tags.extend(map(lambda path: area + '[' + path + ']', base_paths))

	return tags

def extract_topic_taxons(item):
	links = item['links']
	base_paths = []

	if 'taxons' in item['links']:
		base_paths.extend(map(lambda link: link['base_path'], links['taxons']))
		return list(set(_flatmap(_explode_base_path, base_paths)))

	return []

def _explode_base_path(base_path):
	parts = base_path.split("/")
	exploded_paths = ["/".join(parts[0:end]) for end in range(1, len(parts)+1)]
	return exploded_paths[1:]

def _filter_imported(base_paths):
	return [path for path in base_paths if not path.startswith("/imported")]

def _flatmap(f, items):
	return list(chain.from_iterable(map(f, items)))

def _convert_to_csv(hashes):
	output = StringIO()
	writer = csv.DictWriter(output, fieldnames = hashes[0].keys())
	writer.writeheader()
	[writer.writerow(hash) for hash in hashes]
	return output.getvalue()

def _mapping_quality(mapping):
	return mapping["mapping_count"] * mapping["mapping_share"]

file = gzip.open(DATA_PATH)
gen = ijson.items(file, prefix='item')
gen = islice(gen, 10)

mapping = legacy_taxon_to_topic_taxons(gen)
print(json.dumps(mapping, indent=4))
scores = convert_to_scored_hashes(mapping)
scores.sort(key = _mapping_quality)
# print(_convert_to_csv(scores))
