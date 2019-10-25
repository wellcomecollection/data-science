import requests
from urllib.parse import quote


def search(query):
    base_url = 'https://en.wikipedia.org/w/api.php?action=query&format=json&list=search&srsearch='
    complete_json = requests.get(base_url + query).json()
    try:
        result = complete_json['query']['search'][0]['title'].replace(' ', '_')
    except (KeyError, IndexError):
        result = ''
    return result


def get_wikipedia_data(entity):
    best_result = search(entity)
    base_url = 'https://en.wikipedia.org/w/api.php?action=query&format=json&prop=pageprops|pageterms&titles='
    complete_json = requests.get(base_url + best_result).json()
    try:
        data = list(complete_json['query']['pages'].values())[0]
    except KeyError:
        data = []

    if 'pageprops' in data and 'terms' in data:
        return data
    else:
        raise ValueError(
            'looks like {} isn\'t a valid wikidata entity'.format(entity)
        )


def get_title_and_description(wikipedia_data):
    try:
        title = wikipedia_data['terms']['label'][0]
        url = 'https://en.wikipedia.org/wiki/' + title.replace(' ', '_')
    except KeyError:
        title = None
        url = None
    try:
        description = wikipedia_data['terms']['description'][0]
    except KeyError:
        description = None
    return title, url, description


def get_wikidata_id(wikipedia_data):
    return wikipedia_data['pageprops']['wikibase_item']


def get_wikidata_json(wikipedia_data):
    wikidata_id = get_wikidata_id(wikipedia_data)
    base_url = 'https://www.wikidata.org/wiki/Special:EntityData/{}.json'
    url = base_url.format(wikidata_id)
    wikidata_json = requests.get(url).json()
    return wikidata_json['entities'][wikidata_id]


def get_lcsh_id(wikidata_json):
    try:
        lcsh_id = wikidata_json['claims']['P244'][0]['mainsnak']['datavalue']['value']
    except (KeyError, TypeError):
        lcsh_id = None
    return lcsh_id


def get_mesh_id(wikidata_json):
    try:
        mesh_id = wikidata_json['claims']['P486'][0]['mainsnak']['datavalue']['value']
    except (KeyError, TypeError):
        mesh_id = None
    return mesh_id


def get_identifiers(entity):
    try:
        wikipedia_data = get_wikipedia_data(entity)
        title, url, description = get_title_and_description(wikipedia_data)
        wikidata_json = get_wikidata_json(wikipedia_data)
        wikidata_id = get_wikidata_id(wikipedia_data)
        lcsh_id = get_lcsh_id(wikidata_json)
        mesh_id = get_mesh_id(wikidata_json)
        identifiers = {
            'title': title,
            'description': description,
            'wikipedia_url': url,
            'wikidata_id': wikidata_id,
            'lcsh_id': lcsh_id,
            'mesh_id': mesh_id
        }
        return identifiers
    except ValueError:
        return None
