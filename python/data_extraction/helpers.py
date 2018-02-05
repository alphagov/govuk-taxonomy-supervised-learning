def slice(hash_in, key_list):
    return {key: value for (key, value) in hash_in.items() if key in key_list}


def dig(hash_in, *key_list):
    (head, *tail) = key_list
    if isinstance(head, int) and isinstance(hash_in, list):
        value = hash_in[head]
    elif isinstance(head, str) and isinstance(hash_in, dict):
        value = hash_in.get(head)
    else:
        return hash_in
    if len(tail) == 0:
        return value
    else:
        return dig(value, *tail)


def merge(hash_one, hash_two):
    return dict(hash_one, **hash_two)
