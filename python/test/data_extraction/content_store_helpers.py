import responses, uuid
from data_extraction import plek


def content_store_has_item(path, json):
    responses.add(responses.GET, plek.find("draft-content-store") + "/content" + path,
                  json=json, status=200)


level_one_taxons = {
    "links": {
        "level_one_taxons":
            [
                {"base_path": "/taxons/root_taxon",
                 "content_id": "rrrr"}
            ]
        }
}

multi_level_child_taxons = {
    "base_path": "/taxons/root_taxon",
    "content_id": "rrrr",
    "links": {
        "child_taxons": [
            {
                "base_path": "/root_taxon/taxon_a",
                "content_id": "aaaa",
                "links": {
                    "child_taxons": [
                        {
                            "base_path": "/root_taxon/taxon_1",
                            "content_id": "aaaa_1111",
                            "links": {}
                        },
                        {
                            "base_path": "/root_taxon/taxon_2",
                            "content_id": "aaaa_2222",
                            "links": {}
                        }
                    ]
                }
            }
        ]
    }
}


def single_level_child_taxons(root, child_1, child_2):
    return {
        "base_path": "/taxons/#{root}",
        "content_id": root,
        "links": {
            "child_taxons": [
                {
                    "base_path": "/taxons/%s" % child_1,
                    "content_id": child_1,
                    "links": {}
                },
                {
                    "base_path": "/taxons/%s" % child_2,
                    "content_id": child_2,
                    "links": {}
                }
            ]
        }
    }

content_with_taxons = {
    "base_path": "/base_path",
    "content_id": "d282d35a-2bd2-4e14-a7a6-a04e6b10520f",
    "links": {
        "taxons": [{"content_id": "237b2e72-c465-42fe-9293-8b6af21713c0"},
                   {"content_id": "8da62d85-47c0-42df-94c4-eaaeac329671"}]
    }
}

content_with_ppo = {
    "base_path": "/base_path",
    "content_id": "d282d35a-2bd2-4e14-a7a6-a04e6b10520f",
    "links": {
        "primary_publishing_organisation": [
            {"title": "title1"}
        ]
    },
}

content_without_taxons = {
    "base_path": "/base_path",
    "content_id": "d282d35a-2bd2-4e14-a7a6-a04e6b10520f",
    "links": {
    }
}

content_links = {
    'results': [{'link': '/first/path'}, {'link': '/second/path'}]
}

content_first = {
    "base_path": '/first/path',
    "content_id": str(uuid.uuid4()),
    "links": {
    }
}

content_second = {
    "base_path": '/second/path',
    "content_id": str(uuid.uuid4()),
    "links": {
    }
}
