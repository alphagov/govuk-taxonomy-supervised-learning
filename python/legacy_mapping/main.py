import ijson
import json
import gzip
import os
import progressbar

from itertools import chain, groupby, islice
from data import document_types_excluded_from_the_topic_taxonomy

DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/content.json.gz')
EXCLUDED_DOCS = document_types_excluded_from_the_topic_taxonomy()
LEGACY_TAGS = ['policy_areas', 'topics', 'policies']
progress_bar = progressbar.ProgressBar()

def tag_to_path_sets_dict(items):
	mapping = {}

	for item in progress_bar(items):
		if item["document_type"] in EXCLUDED_DOCS:
			continue

		for key in legacy_base_paths(item):
			path_set = _filter_imported(taxon_paths(item))
			if len(path_set) == 0 or len(key) == 0: continue
			if not key in mapping: mapping[key] = []
			mapping[key].append(path_set)

	return mapping

def score_common_paths(path_sets):
	scores = {}
	all_paths = _flatmap(lambda x: x, path_sets)
	all_paths.sort()

	for path, dups in groupby(all_paths):
		dups_len = len(list(dups))
		scores[path] = [dups_len, dups_len / len(path_sets)]

	return scores

def legacy_base_paths(item):
	tags = []
	links = item['links']

	for area in LEGACY_TAGS:
		if area in item['links']:
			base_paths = map(lambda link: link['base_path'], links[area])
			tags.extend(map(lambda path: area + '[' + path + ']', base_paths))

	return tags

def taxon_paths(item):
	links = item['links']
	base_paths = []

	if 'taxons' in item['links']:
		base_paths.extend(map(lambda link: link['base_path'], links['taxons']))
		return list(set(_flatmap(_explode_base_path, base_paths)))

	return []

def _explode_base_path(base_path):
	parts = base_path.split("/")
	exploded_paths = ["/".join(parts[0:end]) for end in range(1, len(parts)+1)]
	return exploded_paths[1:-1]

def _filter_imported(base_paths):
	return [path for path in base_paths if not path.startswith("/imported")]

def _flatmap(f, items):
	return list(chain.from_iterable(map(f, items)))

file = gzip.open(DATA_PATH)
gen = ijson.items(file, prefix='item')
gen = islice(gen, 1000)

mapping = tag_to_path_sets_dict(gen)

for tag, path_sets in mapping.items():
	mapping[tag] = score_common_paths(path_sets)

print(json.dumps(mapping, indent=4))
