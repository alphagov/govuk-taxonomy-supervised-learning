import requests

class PolicyArea():
    def __init__(
        self,
        title,
        link,
        content_id=None
    ):
        self.title = title
        self.link = link
        self.content_id = content_id

    def __repr__(self):
        return "<PolicyArea {}>".format(self.link)

def get_all_policy_areas_by_link():
    url = "https://www.gov.uk/government/topics.json"

    response = requests.get(url).json()

    return {
        document["document"]["link"]: PolicyArea(
            document["document"]["title"],
            document["document"]["link"],
        )
        for document in response["documents"]
    }
