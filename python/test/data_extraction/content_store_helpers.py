import responses, uuid
from lib import plek


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
    "title": "title1",
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

content_with_direct_link_to_root_taxon = {
    "base_path": "/base_path",
    "content_id": "d282d35a-2bd2-4e14-a7a6-a04e6b10520f",
    "links": {
        "root_taxon": [{}]
    }
}

content_with_multiple_unconnected_taxons = {
    "base_path": "/base_path",
    "content_id": "d282d35a-2bd2-4e14-a7a6-a04e6b10520f",
    "links": {
        "taxons": [
            {
                "content_id": "377d3d35-4bc8-45a5-99be-d0ac4f25437a",
                "links": {
                }
            },
            {
                "content_id": "377d3d35-4bc8-45a5-99be-d0ac4f25437f",
                "links": {
                }
            }
        ],
    },
}

content_with_multiple_taxons = {
    "base_path": "/base_path",
    "content_id": "d282d35a-2bd2-4e14-a7a6-a04e6b10520f",
    "links": {
        "taxons": [
            {
                "content_id": "377d3d35-4bc8-45a5-99be-d0ac4f25437a",
                "links": {
                }
            },
            {
                "content_id": "377d3d35-4bc8-45a5-99be-d0ac4f25437f",
                "links": {
                    "parent_taxons": [
                        {
                            "content_id": "84a394d2-b388-4e4e-904e-136ca3f5dd7d",
                            "links": {
                                "parent_taxons": [
                                    {
                                        "links": {
                                            "root_taxon": [
                                                {
                                                    "content_id": "f3bbdec2-0e62-4520-a7fd-6ffd5d36e03a",
                                                }
                                            ]
                                        },
                                    }
                                ]
                            },
                        }
                    ]
                },
            }
        ],
    },
}
