#!/usr/bin/env python3

import csv
from itertools import chain
from collections import defaultdict

from data import items_from_taxons_file
from data.policies import get_all_policies_by_link, Policy
from data.policy_areas import get_all_policy_areas_by_link, PolicyArea
from data.specialist_topics import get_all_specialist_topics_by_link, SpecialistTopic

class Taxon():
    def __init__(self, base_path, content_id, title, parent):
        self.base_path = base_path
        self.content_id = content_id
        self.title = title
        self.parent = parent
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

def get_topic_taxonomy():
    homepage = Taxon(
        "/",
        "f3bbdec2-0e62-4520-a7fd-6ffd5d36e03a",
        "GOV.UK homepage",
        parent=None
    )

    taxons_by_content_id = {
        homepage.content_id: homepage
    }

    taxon_parents_by_content_id = {}

    for taxon_data in items_from_taxons_file():
        taxon = Taxon(
            taxon_data["base_path"],
            taxon_data["content_id"],
            taxon_data["title"],
            parent=None # Fill in the parents later
        )
        
        taxons_by_content_id[taxon.content_id] = taxon
        taxon_parents_by_content_id[taxon.content_id] = taxon_data.get(
            "parent_content_id", homepage.content_id
        )

    for content_id, taxon in taxons_by_content_id.items():
        if content_id == homepage.content_id:
            continue
        
        taxon.parent = taxons_by_content_id[
            taxon_parents_by_content_id[content_id]
        ]

    taxon_children_by_content_id = defaultdict(list)

    for child_content_id, parent_content_id in taxon_parents_by_content_id.items():
        taxon_children_by_content_id[parent_content_id].append(child_content_id)

    for content_id, children in taxon_children_by_content_id.items():
        taxons_by_content_id[content_id].children = [
            taxons_by_content_id[child_content_id]
            for child_content_id in children
        ]

    return homepage

def map_tag_to_taxon(mapping, taxon):
    def get_first_occurance_of(legacy_class, taxon):
        legacy_things = [
            x for x in mapping.get(taxon, [])
            if isinstance(x, legacy_class)
        ]

        if legacy_things:
            return legacy_things
        else:
            if taxon.parent is None:
                return []
            else:
                return get_first_occurance_of(legacy_class, taxon.parent)

    return list(
        chain(*[
            get_first_occurance_of(x, taxon)
            for x in (Policy, PolicyArea, SpecialistTopic)
        ])
    )

def generate_mapping(taxons):
    policies = get_all_policies_by_link()
    policy_areas = get_all_policy_areas_by_link()
    specialist_topics = get_all_specialist_topics_by_link()

    mapping = {}

    def add_to_mapping(taxon, x):
        if taxon in mapping:
            mapping[taxon].append(x)
        else:
            mapping[taxon] = [x]

    def last_segment_matches(a, b):
        return a.split("/")[-1] == b.split("/")[-1]

    for link, policy in policies.items():
        for taxon in taxons:
            if last_segment_matches(taxon.base_path, link):
                add_to_mapping(taxon, policy)

    for link, policy_area in policy_areas.items():
        for taxon in taxons:
            if last_segment_matches(taxon.base_path, link):
                add_to_mapping(taxon, policy_area)

    for link, specialist_topic in specialist_topics.items():
        for taxon in taxons:
            if last_segment_matches(taxon.base_path, link):
                add_to_mapping(taxon, specialist_topic)

    return mapping


def write_analysis_csv(filename='mapping_analysis.csv'):
    homepage = get_topic_taxonomy()
    all_taxons = list(homepage.descendants())

    taxons_to_exclude = [
        x for x in all_taxons
        if x.base_path in ('/education', '/transport/all', '/childcare-parenting', '/entering-staying-uk')
    ]

    taxons = [
        x for x in all_taxons
        if (
                (x not in taxons_to_exclude) and
                all(
                    (not x.has_ancestor(taxon_to_exclude))
                    for taxon_to_exclude in taxons_to_exclude
                )
        )
    ]

    mapping = generate_mapping(taxons)

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

def write_content_tagger_csv(filename='mapping.csv'):
    homepage = get_topic_taxonomy()
    all_taxons = list(homepage.descendants())

    taxons_to_exclude = [
        x for x in all_taxons
        if x.base_path in ('/education', '/transport/all', '/childcare-parenting', '/entering-staying-uk')
    ]

    taxons = [
        x for x in all_taxons
        if (
                (x not in taxons_to_exclude) and
                all(
                    (not x.has_ancestor(taxon_to_exclude))
                    for taxon_to_exclude in taxons_to_exclude
                )
        )
    ]

    mapping = generate_mapping(taxons)

    taxon_content_id_and_mappings = [
        (taxon.content_id, [x.link for x in mapping[taxon]])
        for taxon in taxons
        if taxon in mapping
    ]

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for taxon_content_id, legacy_base_paths in taxon_content_id_and_mappings:
            writer.writerow((taxon_content_id, '|'.join(legacy_base_paths)))

if __name__ == '__main__':
    write_content_tagger_csv()
