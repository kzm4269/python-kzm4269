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


def dump_tree(
        root, label=None, children=None, has_children=None, include_root=True):
    """Convert a object which have a tree structure to a string.

    Parameters
    ----------
    root : any object
        Root node of the tree.
    label : callable, optional
        Function converts the given node to a string.
    children : callable, optional
        Function returns children of the given node.
    has_children : callable, optional
        Function returns true if the given node has children.
    include_root : bool, optional
        If true is given, include information of the root node into the output.

    Returns
    -------
    out : str

    Examples
    --------
    >>> print(dump_tree(
    ...     ['Alpha',
    ...      ['Bravo',
    ...       ['Charlie']],
    ...      ['Delta',
    ...       ['Echo'],
    ...       ['Foxtrot']],
    ...      ['Golf']]))
    Alpha
    |-- Bravo
    |   `-- Charlie
    |-- Delta
    |   |-- Echo
    |   `-- Foxtrot
    `-- Golf
    """
    if not callable(label):
        def label(node):
            return next(iter(node))
    if not callable(children):
        def children(node):
            return list(node)[1:]
    if callable(has_children):
        _has_children = has_children

        def _children(node):
            return children(node) if _has_children(node) else ()
    else:
        _children = children

        def _has_children(node):
            _children_iter = iter(children(node))
            try:
                next(_children_iter)
            except StopIteration:
                return False
            return True

    def _dump_tree(node):
        tree = dump_tree(
            root=node, label=label, children=_children,
            has_children=_has_children, include_root=False)
        return str(label(node)) + ('\n' + tree if tree else '')

    if include_root:
        return _dump_tree(root)

    children_iter = iter(_children(root))
    try:
        child = next(children_iter)
    except StopIteration:
        return ''

    result = ''
    while True:
        child_tree = _dump_tree(child)
        try:
            child = next(children_iter)
        except StopIteration:
            return result + '`-- ' + child_tree.replace('\n', '\n    ')
        result += '|-- ' + child_tree.replace('\n', '\n|   ') + '\n'
