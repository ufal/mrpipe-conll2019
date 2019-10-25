#!/usr/bin/env python3
import json
import fileinput

for line in fileinput.input():
    d = json.loads(line)
    if d["framework"] == "eds":
        for node in d["nodes"]:
            if "anchors" in node and len(node["anchors"]) > 1:
                node["anchors"] = node["anchors"][:1]
        for node in d["nodes"]:
            if "anchors" not in node:
                anchor = None
                for edge in d["edges"]:
                    if edge["source"] == node["id"]:
                        target = [n for n in d["nodes"] if n["id"] == edge["target"] and "anchors" in n]
                        if len(target):
                            anchor = target[0]["anchors"]
                            break
                if not anchor:
                    for edge in d["edges"]:
                        if edge["target"] == node["id"]:
                            source = [n for n in d["nodes"] if n["id"] == edge["source"] and "anchors" in n]
                            if len(source):
                                anchor = source[0]["anchors"]
                                break
                if not anchor:
                    raise ValueError("Anchor for node {} in sentence {} not found".format(node, d["id"]))
                node["anchors"] = anchor
    print(json.dumps(d, ensure_ascii=False))
