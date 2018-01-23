import os


# Naive implementation of Plek

def find(service):
    defined_uri = __defined_service_uri_for(service)
    if defined_uri:
        return defined_uri

    domain = os.getenv('GOVUK_APP_DOMAIN') or os.getenv('GOVUK_APP_DOMAIN_EXTERNAL')

    if domain:
        scheme = 'https'
    else:
        domain = "dev.gov.uk"
        scheme = 'http'

    return "{scheme}://{service}.{domain}".format(scheme=scheme, service=service, domain=domain)


def __defined_service_uri_for(service):
    var_name = "PLEK_SERVICE_{service}_URI".format(service=service.upper().replace("-", "_"))
    return os.getenv(var_name)
