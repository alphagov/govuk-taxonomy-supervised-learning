#!/usr/bin/env python3

import os
import data
import data.organisations
import progressbar
from collections import defaultdict

from lib.helpers import dig

import pathlib


class Taxon:
    def __init__(self, content_id, title):
        self.content_id = content_id
        self.title = title
        self.children = {}
        self.content_count = defaultdict(int)
        self.content_count_for_descendants = defaultdict(int)

    def record_content_item_tagged_to_descendants(self, item, children_path, tagging_keys):
        child_content_id = children_path[0]["content_id"]

        if child_content_id in self.children:
            child = self.children[child_content_id]
        else:
            child = self.children[child_content_id] = Taxon(
                child_content_id,
                children_path[0]["title"],
            )

        for key in tagging_keys(item):
            self.content_count_for_descendants[key] += 1

        if len(children_path) == 1:
            child.record_tagged_content_item(item, tagging_keys)
        else:
            child.record_content_item_tagged_to_descendants(item, children_path[1:], tagging_keys)

    def record_tagged_content_item(self, item, tagging_keys):
        for key in tagging_keys(item):
            self.content_count[key] += 1

    def content_count_for_self_and_children(self, keys):
        return (
            sum(self.content_count[key] for key in keys) +
            sum(
                child.content_count_for_self_and_children(keys)
                for child in self.children.values()
            )
        )

    def __repr__(self):
        return "<Taxon {}>".format(self.title)


def get_taxons_and_parents_from_links(content_item):
    def get_taxon_parents(content_item):
        parent = (
            dig(content_item, "links", "parent_taxons", 0) or
            dig(content_item, "links", "root_taxon", 0)
        )

        if parent is None:
            return []

        return [parent] + get_taxon_parents(parent)

    taxons = dig(content_item, "links", "taxons") or []

    return [
        [taxon] + get_taxon_parents(taxon)
        for taxon in taxons
    ]


def extract_related_organisations(content_item):
    organisations = set()

    primary_publishing_organisation = dig(
        content_item,
        "links",
        "primary_publishing_organisation",
        0,
    )

    if primary_publishing_organisation:
        organisations.add(primary_publishing_organisation["content_id"])

    organisation_items = dig(
        content_item,
        "links",
        "organisations",
    )

    if organisation_items:
        organisations.update(
            organisation_item["content_id"]
            for organisation_item in organisation_items
        )

    return organisations


def gather_data(content_items):
    homepage = Taxon(
        "f3bbdec2-0e62-4520-a7fd-6ffd5d36e03a",
        "GOV.UK homepage",
    )

    for item in content_items:
        taxons_and_parents = get_taxons_and_parents_from_links(item)

        for taxons in taxons_and_parents:
            if taxons[-1]["content_id"] != homepage.content_id:
                # Skip any taxons which are not part of the topic
                # taxonomy
                continue

            homepage.record_content_item_tagged_to_descendants(
                item,
                list(reversed(taxons))[1:],
                extract_related_organisations,
            )

    return homepage


def write_csv_file_for_organisations(
    output_file,
    taxon_tree,
    organisations,
):
    def process_taxons(taxons, level):
        if level >= 3:
            return

        for taxon in sorted(taxons, key=lambda x: x.title.lower()):
            tagged_content_count = sum(
                taxon.content_count[
                    organisation.content_id
                ]
                for organisation in organisations
            )

            descendants_tagged_content_count = sum(
                taxon.content_count_for_descendants[
                    organisation.content_id
                ]
                for organisation in organisations
            )

            tagged_content_count_for_taxon_and_children = \
                taxon.content_count_for_self_and_children(
                    tuple(
                        organisation.content_id
                        for organisation in organisations
                    )
                )

            if tagged_content_count or descendants_tagged_content_count:
                output_file.write(
                    "{},,{}\"{} ({})\"\n".format(
                        tagged_content_count_for_taxon_and_children,
                        "," * level,
                        taxon.title,
                        tagged_content_count,
                    )
                )

                if descendants_tagged_content_count != 0:
                    process_taxons(taxon.children.values(), level + 1)

    process_taxons([taxon_tree], level=0)


if __name__ == "__main__":
    ORGANISATION_NAMES = [
        "Attorney General's Office",
        "Cabinet Office",
        "Department for Business, Energy & Industrial Strategy",
        "Department for Digital, Culture, Media & Sport",
        "Department for Education",
        "Department for Environment, Food & Rural Affairs",
        "Department for Exiting the European Union",
        "Department for International Development",
        "Department for International Trade",
        "Department for Transport",
        "Department for Work and Pensions",
        "Department of Health and Social Care",
        "Foreign & Commonwealth Office",
        "HM Treasury",
        "Home Office",
        "Ministry of Defence",
        "Ministry of Housing, Communities & Local Government",
        "Ministry of Justice",
        "Northern Ireland Office",
        "Office of the Advocate General for Scotland",
        "Office of the Leader of the House of Commons",
        "Office of the Leader of the House of Lords",
        "Office of the Secretary of State for Scotland",
        "Office of the Secretary of State for Wales",
        "UK Export Finance",

        "The Charity Commission",
        "Competition and Markets Authority",
        "Crown Prosecution Service",
        "Food Standards Agency",
        "Forestry Commission",
        "Government Actuary's Department",
        "Government Legal Department",
        "HM Land Registry",
        "HM Revenue & Customs",
        "NS&I",
        "The National Archives",
        "National Crime Agency",
        "Office of Rail and Road",
        "Ofgem",
        "Ofqual",
        "Ofsted",
        "Serious Fraud Office",
        "Supreme Court of the United Kingdom",
        "UK Statistics Authority",
        "The Water Services Regulation Authority ",
    ]

    organisations_by_content_id = data.organisations.get_organisations_by_content_id()

    def find_organisation(title):
        for organisation in organisations_by_content_id.values():
            if organisation.title == title:
                return organisation
        else:
            raise KeyError(title)

    organisations = [find_organisation(name) for name in ORGANISATION_NAMES]

    progress_bar = progressbar.ProgressBar(redirect_stdout=True)
    homepage = gather_data(progress_bar(data.items_from_content_file()))

    output_prefix = "data/tagged_content_by_organisation_in_csv/"
    pathlib.Path(output_prefix).mkdir(exist_ok=True)

    for organisation in organisations:
        print("Writing data for ", organisation.title)

        descendants = organisation.descendants()

        with open(
            os.path.join(
                output_prefix,
                "{}_and_descendants.csv".format(organisation.slug),
            ),
            mode="w",
            encoding="utf8",
        ) as f:
            if descendants:
                f.write("\"{} including:\"\n".format(organisation.title))
                for descendant in descendants:
                    f.write(",\"{}\"\n".format(descendant.title))
            else:
                f.write("\"{} with no descendants\"\n".format(organisation.title))
            f.write("\n")

            write_csv_file_for_organisations(
                f,
                homepage,
                [organisation] + descendants,
            )

        for individual_organisation in ([organisation] + descendants):
            pathlib.Path(
                os.path.join(
                    output_prefix,
                    organisation.slug,
                )
            ).mkdir(exist_ok=True)

            with open(
                os.path.join(
                    output_prefix,
                    "{}/{}.csv".format(
                        organisation.slug,
                        individual_organisation.slug
                    )
                ),
                mode="w",
                encoding="utf8",
            ) as f:
                f.write("\"{}\"".format(individual_organisation.title))
                f.write("\n\n")

                write_csv_file_for_organisations(
                    f,
                    homepage,
                    [individual_organisation],
                )
