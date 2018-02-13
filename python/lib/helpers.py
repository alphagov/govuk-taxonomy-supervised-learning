def slice(dict_in, key_list):
    return {key: value for (key, value) in dict_in.items() if key in key_list}


def dig(dict_in, *key_list):
    (head, *tail) = key_list
    if isinstance(head, int) and isinstance(dict_in, list):
        value = dict_in[head]
    elif isinstance(head, str) and isinstance(dict_in, dict):
        value = dict_in.get(head)
    else:
        return dict_in
    if len(tail) == 0:
        return value
    else:
        return dig(value, *tail)


def merge(dict_one, dict_two):
    return dict(dict_one, **dict_two)
