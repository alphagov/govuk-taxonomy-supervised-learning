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
