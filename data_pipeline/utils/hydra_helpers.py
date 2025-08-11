import importlib


def load_attr(path: str):
    """
    Import and return the attribute at `path`.
    E.g. path="foo.bar.BAZ" will import foo.bar and return foo.bar.BAZ.
    """
    module_path, name = path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, name)
