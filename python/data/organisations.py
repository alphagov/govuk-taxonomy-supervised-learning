import requests
import itertools

class Organisation():
    def __init__(
        self,
        organisation_id,
        content_id,
        title,
        slug,
        child_organisations=[],
        superseding_organisations=[],
    ):
        self.content_id = content_id
        self.title = title
        self.slug = slug
        self.child_organisations = child_organisations
        self.superseding_organisations = superseding_organisations

    def __repr__(self):
        return "<Organisation {}>".format(self.title)

    def descendants(self):
        descendants = self.child_organisations[:]

        for child in self.child_organisations:
            descendants.extend(child.descendants())

        return descendants

def get_all_organisations_by_id():
    url = "https://www.gov.uk/api/organisations?page=1"
    results = {}

    while url is not None:
        response = requests.get(url).json()

        for result in response["results"]:
            results[result["id"]] = result

        url = response.get("next_page_url")

    return results

def get_organisations_by_content_id():
    organisation_data = get_all_organisations_by_id()

    organisations_by_id = {
        organisation_id: Organisation(
            organisation_id,
            data["details"]["content_id"],
            data["title"],
            data["details"]["slug"],
        )
        for organisation_id, data in organisation_data.items()
    }

    for organisation_id, data in organisation_data.items():
        organisation = organisations_by_id[organisation_id]

        organisation.child_organisations = list(
            map(
                lambda x: organisations_by_id[x],
                map(lambda x: x["id"], data["child_organisations"])
            )
        )

        organisation.superseding_organisations = list(
            map(
                lambda x: organisations_by_id[x],
                map(lambda x: x["id"], data["superseding_organisations"])
            )
        )

    return {
        organisation.content_id: organisation
        for organisation in organisations_by_id.values()
    }
