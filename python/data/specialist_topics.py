import requests

class SpecialistTopic():
    def __init__(
            self,
            title,
            slug,
            link
    ):
        self.title = title
        self.slug = slug
        self.link = link

    def __repr__(self):
        return "<SpecialistTopic {}>".format(self.link)

def get_all_specialist_topics_by_link():
    url = "https://www.gov.uk/api/search.json?filter_content_store_document_type=topic&count=999"

    response = requests.get(url).json()

    return {
        document["link"]: SpecialistTopic(
            document["title"],
            document["slug"],
            document["link"],
        )
        for document in response["results"]
    }
