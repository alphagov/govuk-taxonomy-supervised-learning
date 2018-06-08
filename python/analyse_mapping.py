#!/usr/bin/env python3

import csv
from itertools import chain
from data import content_store

from data.policies import Policy
from data.policy_areas import PolicyArea
from data.specialist_topics import SpecialistTopic

from mapping import map_tag_to_taxon

class Taxon():
    def __init__(self, base_path, content_id, title, parent):
        self.base_path = base_path
        self.content_id = content_id
        self.title = title
        self.parent = parent
        self.legacy_taxons = []
        self.children = []

    def descendants(self):
        for child in self.children:
            yield child
            yield from child.descendants()

    def has_ancestor(self, ancestor):
        if self.parent is None:
            return False

        if self.parent == ancestor:
            return True
        else:
            return self.parent.has_ancestor(ancestor)

    def depth(self):
        if self.parent is None:
            return 0
        else:
            return self.parent.depth() + 1

    def __repr__(self):
        return "<Taxon {}>".format(self.title)

def build_legacy_taxon(legacy_taxon_content_item):
    legacy_class = {
        'policy': Policy,
        'policy_area': PolicyArea,
        'placeholder_policy_area': PolicyArea,
        'topic': SpecialistTopic
    }[legacy_taxon_content_item['document_type']]

    return legacy_class(
        legacy_taxon_content_item['title'],
        legacy_taxon_content_item['base_path'], # Close enough to the link
        content_id=legacy_taxon_content_item['content_id']
    )

def build_taxon_from_content_item(content_item, parent):
    taxon = Taxon(
        content_item["base_path"],
        content_item["content_id"],
        content_item["title"],
        parent
    )

    taxon.children = [
        build_taxon_from_content_item(child_content_item, taxon)
        for child_content_item
        in content_item["links"].get("child_taxons", [])
    ]

    taxon.legacy_taxons = [
        build_legacy_taxon(x)
        for x in content_item["links"].get("legacy_taxons", [])
    ]

    return taxon

def get_topic_taxonomy():
    homepage_content_item = content_store.fetch_content_item("/")

    homepage_taxon = build_taxon_from_content_item(
        homepage_content_item,
        parent=None
    )

    homepage_taxon.children = [
        build_taxon_from_content_item(
            content_store.fetch_content_item(level_one_taxon["base_path"]),
            parent=homepage_taxon
        )
        for level_one_taxon in homepage_content_item["links"]["level_one_taxons"]
    ]

    return homepage_taxon

def taxons_which_were_imported(homepage):
    all_taxons = list(homepage.descendants())

    taxons_to_exclude = [
        x for x in all_taxons
        if x.base_path in ('/education', '/transport/all', '/childcare-parenting', '/entering-staying-uk')
    ]

    return [
        x for x in all_taxons
        if (
                (x not in taxons_to_exclude) and
                all(
                    (not x.has_ancestor(taxon_to_exclude))
                    for taxon_to_exclude in taxons_to_exclude
                )
        )
    ]

def print_all_content_ids_for_legacy_things_in_the_mapping(mapping):
    legacy_content_ids_with_mapping = set(
        legacy_thing.content_id
        for taxon in taxons
        for legacy_thing in map_tag_to_taxon(mapping, taxon)
    )

    for content_id in legacy_content_ids_with_mapping:
        print("'" + content_id + "',")

def print_all_level_one_topics_in_the_mapping(mapping):
    print("Level one topics")
    for legacy_thing in chain(*mapping.values()):
        if legacy_thing.link.startswith("/topic") and legacy_thing.link.count("/") <= 2:
            print(legacy_thing)
    
def write_analysis_csv(homepage, mapping, filename='mapping_analysis_actual.csv'):
    taxons = taxons_which_were_imported(homepage)

    taxon_lines = [
        "{}{}".format(taxon.depth() * "    ", taxon.title)
        for taxon in taxons
    ]

    max_taxon_line_length = max(len(x) for x in taxon_lines)

    for line, taxon in zip(taxon_lines, taxons):
        padding = " " * (max_taxon_line_length - len(line))
        #print(line + "  " + padding + str(map_tag_to_taxon(mapping, taxon)))

    max_taxon_depth = max(x.depth() for x in taxons)

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for taxon in taxons:
            writer.writerow(
                ([''] * taxon.depth()) +
                [taxon.title] +
                ([''] * (max_taxon_depth - taxon.depth())) +
                [''] +
                [
                    ' '.join([
                        x.link for x in map_tag_to_taxon(mapping, taxon)
                        if isinstance(x, legacy_class)
                    ])
                    for legacy_class in (PolicyArea, Policy, SpecialistTopic)
                ]
            )

if __name__ == "__main__":
    topic_taxonomy = get_topic_taxonomy()

    mapping = {
        taxon: taxon.legacy_taxons
        for taxon in topic_taxonomy.descendants()
        if taxon.legacy_taxons
    }

    write_analysis_csv(topic_taxonomy, mapping)
