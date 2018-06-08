import requests

class Policy():
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
        return "<Policy {}>".format(self.link)

def get_all_policies_by_link():
    url = "https://www.gov.uk/government/policies.json"

    response = requests.get(url).json()

    return {
        document["document"]["link"]: Policy(
            document["document"]["title"],
            document["document"]["link"],
        )
        for document in response["documents"]
    }
