"""Warm request dispatch for prompts pre-encoded before the DiT is loaded."""

import json


def request_key(data):
    return json.dumps([data["prompt"], data.get("edit_prompt", {})],
                      sort_keys=True, ensure_ascii=False)


def iter_requests(initial_items, contexts, *, serve_stdin, stream, on_error=None):
    if not serve_stdin:
        for item in initial_items:
            yield item, contexts[request_key(item)]
        return

    for line in stream:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            for required in ("prompt", "cond_image", "cond_audio"):
                if not item.get(required):
                    raise ValueError(f"request is missing {required}")
            key = request_key(item)
            if key not in contexts:
                raise ValueError("request prompt was not pre-encoded at startup")
        except (ValueError, TypeError, AttributeError) as error:
            if on_error is None:
                raise
            on_error(error)
            continue
        yield item, contexts[key]


def reset_kv_caches(caches):
    """Clear prior request history while keeping cache allocation resident."""
    if caches is None:
        return
    for step in caches.values():
        for layer in step.values():
            for key in ("k", "v"):
                layer[key].zero_()
            for key in ("k_scale", "v_scale"):
                if layer[key] is not None:
                    layer[key].fill_(1.0)
