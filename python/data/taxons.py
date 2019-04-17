import itertools

def content_item_taxons(content_item):
    if "taxons" not in content_item["links"]:
        return tuple()

    def recurse_up_the_taxonomy(taxon):
        yield taxon

        if "parent_taxons" in taxon["links"]:
            yield from recurse_up_the_taxonomy(
                taxon["links"]["parent_taxons"][0]
            )

        if "root_taxon" in taxon["links"]:
            yield taxon["links"]["root_taxon"][0]

    return itertools.chain(*[
        recurse_up_the_taxonomy(taxon)
        for taxon in content_item["links"]["taxons"]
    ])

def content_item_tagged_to_topic_taxonomy(content_item):
    return any(
        filter(
            lambda taxon: taxon["base_path"] == "/",
            content_item_taxons(content_item)
        )
    )

def content_item_within_part_of_taxonomy(content_item, taxonomy_part_content_id):
    return any(
        filter(
            lambda taxon: taxon["content_id"] == taxonomy_part_content_id,
            content_item_taxons(content_item)
        )
    )

def content_item_directly_tagged_to_taxon(content_item, taxon_content_id):
    return any(
        map(
            lambda taxon: taxon["content_id"] == taxon_content_id,
            content_item["links"].get("taxons", [])
        )
    )
