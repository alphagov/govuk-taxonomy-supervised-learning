import gzip
import ijson
import progressbar
from collections import defaultdict
from lib.helpers import dig
from lib import services
from data_extraction.export_data import jenkins_compatible_progress_bar
from data import items_from_content_file

def contextual_sidebar_metrics():
    progress_bar = jenkins_compatible_progress_bar()
    content_items = items_from_content_file()

    nav_type_count = defaultdict(int)

    for content_item in progress_bar(content_items):
        nav_type_count[navigation_type(content_item)] += 1

    for nav_type, count in nav_type_count.items():
        services.statsd.gauge('contextual_navigation.nav_type.' + nav_type, count)
        print("{}: {}".format(nav_type, count))

    related_navigation_count = sum(nav_type_count[type] for type in ('mainstream', 'curated', 'default'))
    services.statsd.gauge('contextual_navigation.related_navigation_count', related_navigation_count)

def navigation_type(content_item):
    mainstream_tags = dig(content_item, 'links', 'mainstream_browse_pages') or []
    curated_items = dig(content_item, 'links', 'ordered_related_items') or []

    step_by_step = dig(content_item, 'links', 'part_of_step_navs') or []
    if len(step_by_step) != 1:
        step_by_step = None

    live_taxons = any(
        taxon.get('phase') == 'live'
        for taxon in dig(content_item, 'links', 'taxons') or []
    )

    if step_by_step:
        return 'step_by_step'
    elif mainstream_tags:
        return 'mainstream'
    elif curated_items:
        return 'curated'
    elif live_taxons:
        return 'taxon'
    else:
        return 'default'
