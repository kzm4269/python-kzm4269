from importlib.util import spec_from_file_location
from pathlib import Path


def find_source_files(path):
    return (
        str(path_)
        for path_ in Path(path).glob('**/*')
        if spec_from_file_location(name='', location=path_) is not None
    )


def read_source_files(path):
    def _read():
        for path_ in find_source_files(path):
            with open(path_, 'rb') as fp:
                data = fp.read()
            yield path_, data

    return dict(_read())
