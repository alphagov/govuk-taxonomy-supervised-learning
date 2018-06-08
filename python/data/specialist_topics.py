import requests

class SpecialistTopic():
    def __init__(
            self,
            title,
            link,
            slug=None,
            content_id=None,
    ):
        self.title = title
        self.slug = slug
        self.link = link
        self.content_id = content_id

    def __repr__(self):
        return "<SpecialistTopic {}>".format(self.link)

def get_all_specialist_topics_by_link():
    url = "https://www.gov.uk/api/search.json?filter_content_store_document_type=topic&count=999"

    response = requests.get(url).json()

    return {
        document["link"]: SpecialistTopic(
            document["title"],
            document["link"],
            document["slug"],
        )
        for document in response["results"]
    }
