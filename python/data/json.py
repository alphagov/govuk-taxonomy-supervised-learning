import json

def stream_json(output_file, iterator):
    # The json package in the stdlib doesn't support dumping a
    # generator, but it can handle lists, so this class acts as a
    # go between, making the generator look like a list.
    class StreamContent(list):
        def __bool__(self):
            # The json class tests the truthyness of this object,
            # so this needs to be overridden to True
            return True

        def __iter__(self):
            return iterator

    json.dump(
        StreamContent(),
        output_file,
        indent=4,
        check_circular=False,
        sort_keys=True,
    )
