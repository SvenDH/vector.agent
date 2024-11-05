import inspect
import re

type_map = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}

def func2json(func) -> dict:
    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )
    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}
    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }


def split(text: str, separators: list[str], chunk_size: int = 1000, chunk_overlap: int = 0) -> list[str]:
    chunks = []
    separator = separators[-1]
    newseps = []
    for i, s in enumerate(separators):
        sep = re.escape(s)
        if s == "":
            separator = s
            break
        if re.search(sep, text):
            separator = s
            newseps = separators[i + 1 :]
            break

    sep = re.escape(separator)
    if sep:
        _splits = re.split(f"({sep})", text)
        splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
        if len(_splits) % 2 == 0:
            splits += _splits[-1:]
        splits = [_splits[0]] + splits
    else:
        splits = list(text)
    splits = [s for s in splits if s != ""]

    merged = []
    sep = ""
    for s in splits:
        if len(s) < chunk_size:
            merged.append(s)
        else:
            if merged:
                chunks.extend(merge(merged, sep, chunk_size, chunk_overlap))
                merged = []
            if not newseps:
                chunks.append(s)
            else:
                chunks.extend(split(s, newseps, chunk_size, chunk_overlap))
    if merged:
        chunks.extend(merge(merged, sep, chunk_size, chunk_overlap))
    return chunks


def merge(splits: list[str], separator: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    sep_len = len(separator)
    docs = []
    current_doc = []
    total = 0
    for d in splits:
        n = len(d)
        if total + n + (sep_len if len(current_doc) > 0 else 0) > chunk_size:
            if len(current_doc) > 0:
                doc = separator.join(current_doc).strip()
                if doc != "":
                    docs.append(doc)
                while total > chunk_overlap or (
                    total + n + (sep_len if len(current_doc) > 0 else 0) > chunk_size
                    and total > 0
                ):
                    total -= len(current_doc[0]) + (sep_len if len(current_doc) > 1 else 0)
                    current_doc = current_doc[1:]
        current_doc.append(d)
        total += n + (sep_len if len(current_doc) > 1 else 0)
    doc = separator.join(current_doc).strip()
    if doc != "":
        docs.append(doc)
    return docs
