def slice(hash_in, key_list):
    return {key: value for (key, value) in hash_in.items() if key in key_list}


def dig(hash_in, *key_list):
    (head, *tail) = key_list
    value = hash_in.get(head)
    if len(tail) == 0 or (value is None) or not isinstance(value, dict):
        return value
    else:
        return dig(value, *tail)


def merge(hash_one, hash_two):
    return dict(hash_one, **hash_two)