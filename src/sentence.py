#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2018 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Sentence."""

import collections
import copy
import datetime
import json
import sys

import numpy as np

from lemmatizer import Lemmatizer
from mapping import Mapping


class Sentence:
    PROP_SEP = "\n"
    FORM_SEP = Lemmatizer.SEPARATOR

    TOKENS = 0
    TOKEN_VALUES = 1
    TOKEN_EDGE_PARENTS = 2
    TOKEN_EDGE_CHILDREN = 3
    TOKEN_EDGE_VALUES = 4
    EDGE_PARENTS = 5
    EDGE_CHILDREN = 6
    NODE_VALUES = 7
    EDGE_VALUES = 8
    FACTORS = 9
    FACTORS_1D = [ TOKENS, TOKEN_EDGE_PARENTS, TOKEN_EDGE_CHILDREN, EDGE_PARENTS, EDGE_CHILDREN ]
    FACTORS_2D = [ TOKEN_VALUES, TOKEN_EDGE_VALUES, NODE_VALUES, EDGE_VALUES ]
    MAPPING_1D = [ TOKENS ]
    MAPPING_2D = [ NODE_VALUES, EDGE_VALUES ]
    FACTORS_CHARS = [ TOKENS ] 
    FACTORS_NOT_ENCODED = [ TOKENS, EDGE_VALUES ]


    def __init__(self, dataset, value_dict):
        """Initializes sentence.
        Arguments:
            value_dict: Dictionary with key-value pairs:
                id: sentence id (string),
                flavor: flavor (int),
                framework: framework (string),
                version: version (float),
                time: time (time),
                input: original sentence (string),
                mode: dm|eds|psd|ucca|amr,
        """

        self.dataset = dataset

        # Sentence specific information
        self.id = value_dict["id"]
        self.flavor = value_dict["flavor"]
        self.framework = value_dict["framework"]
        self.version = value_dict["version"]
        self.time = value_dict["time"]
        self.input = value_dict["input"]
        self.mode = value_dict["mode"]

        # Initialize factors
        self.factors = [[] for _ in range(self.FACTORS)]
        self.factor_strings = [[] for _ in range(self.FACTORS)]

        self.children, self.parents = [], []


    def init_tokens(self, value_dict):
        """Initializes tokens in empty sentence.
        Arguments:
            value_dict: Dictionary with key-value pairs:
                tokens: Python array of tokens (strings),
                token_ids: Python array of token ids (ints),
                token_values
                token_ranges: Dictionary of token ranges, with keys "start-end",
                charseq_ids: Python array of token charseq ids (ints),
        """

        # Tokens
        self.factor_strings[self.TOKENS] = value_dict["tokens"]
        self.factors[self.TOKENS] = np.array(value_dict["token_ids"], np.int32)
        self.factors[self.TOKEN_VALUES] = np.array(value_dict["token_values"], np.int32)
        self.token_ranges = value_dict["token_ranges"]
        self.charseq_ids = value_dict["charseq_ids"]

        # Count token character starts and ends
        self.token_starts = dict()
        self.token_ends = dict()
        self.token_to_token_ranges = ["" for x in range(len(self.factor_strings[self.TOKENS]))]
        for token_range in self.token_ranges:
            (start, end) = token_range.split("-")
            self.token_starts[start] = self.token_ranges[token_range]
            self.token_ends[end] = self.token_ranges[token_range]
            self.token_to_token_ranges[self.token_ranges[token_range]] = token_range

        self.children = [[] for _ in range(self.n_tokens())]
        self.parents = [[] for _ in range(self.n_tokens())]

    def init_token_edges(self, value_dict):
        """Initializes edges in empty sentence.
        Arguments:
            value_dict: Dictionary with key-value pairs:
                token_edge_parents: Python array of edge parents (ints),
                token_edge_children: Python array of edge children (ints),
                token_edge_values: Python 2D array of edge property ids x edges (ints),
        """
        self.factors[self.TOKEN_EDGE_PARENTS] = np.array(value_dict["token_edge_parents"], np.int32)
        self.factors[self.TOKEN_EDGE_CHILDREN] = np.array(value_dict["token_edge_children"], np.int32)
        self.factors[self.TOKEN_EDGE_VALUES] = np.array(value_dict["token_edge_values"], np.int32)

    def init_nodes(self, node_values):
        """Initializes nodes in empty sentence.
        """

        self.factors[self.NODE_VALUES] = np.array(node_values, np.int32)
        self.children.extend([[] for _ in range(self.n_nodes())])
        self.parents.extend([[] for _ in range(self.n_nodes())])

    def add_node(self, values):
        self.factors[self.NODE_VALUES].append(values)

        self.children.append([])
        self.parents.append([])


    def init_edges(self, value_dict):
        """Initializes edges in empty sentence.
        Arguments:
            value_dict: Dictionary with key-value pairs:
                edge_parents: Python array of edge parents (ints),
                edge_children: Python array of edge children (ints),
                edge_values: Python 2D array of edge property ids x edges (ints),
        """
        self.factors[self.EDGE_PARENTS] = np.array(value_dict["edge_parents"], np.int32)
        self.factors[self.EDGE_CHILDREN] = np.array(value_dict["edge_children"], np.int32)
        self.factors[self.EDGE_VALUES] = np.array(value_dict["edge_values"], np.int32)

        for e in range(self.n_edges()):
            self.children[self.factors[self.EDGE_PARENTS][e]].append(e)
            self.parents[self.factors[self.EDGE_CHILDREN][e]].append(e)

    def add_edge(self, parent, child, values):
        self.children[parent].append(self.n_edges())
        self.parents[child].append(self.n_edges())

        self.factors[self.EDGE_PARENTS].append(parent)
        self.factors[self.EDGE_CHILDREN].append(child)
        self.factors[self.EDGE_VALUES].append(values)

    def n_tokens(self):
        """Returns number of tokens."""
        return len(self.factor_strings[self.TOKENS])

    def n_token_edges(self):
        return len(self.factor_strings[self.TOKEN_EDGE_VALUES])

    def n_nodes(self):
        """Returns number of nodes."""
        return len(self.factors[self.NODE_VALUES])


    def n_edges(self):
        """Returns number of edges."""
        return len(self.factors[self.EDGE_VALUES])

    def empty_copy(self):
        sentence = copy.copy(self)

        sentence.factors = copy.copy(self.factors)
        for factor in [self.NODE_VALUES, self.EDGE_PARENTS, self.EDGE_CHILDREN, self.EDGE_VALUES]:
            sentence.factors[factor] = []

        sentence.children = [[] for _ in range(self.n_tokens())]
        sentence.parents = [[] for _ in range(self.n_tokens())]

        return sentence

    def write(self, mappings):
        """Returns all values in a Python dictionary."""

        # Sentence specific information, with current time
        now = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d (%H:%M)")
        sentence_dict = collections.OrderedDict([
            ("id", self.id),
            ("flavor", self.flavor),
            ("framework", self.framework),
            ("version", self.version),
            ("time", now),
            ("input", self.input),
        ])

        # Print nodes and tops
        nodes = []
        tops = []

        # Find top nodes
        for e in self.children[0]:
            if self.factors[self.EDGE_VALUES][e][0] == Mapping.ROOT:
                tops.append(int(self.factors[self.EDGE_CHILDREN][e] - self.n_tokens()))

        for i in range(self.n_nodes()):
            node = dict()

            # Node id
            node["id"] = i

            # Node anchors
            node_anchored_tokens, anchors = [], []
            for e in self.children[i + self.n_tokens()]:
                # Invisible edge - anchor
                if self.factors[self.EDGE_VALUES][e][0] == Mapping.ANCHOR:
                    child = self.factors[self.EDGE_CHILDREN][e]
                    if child >= self.n_tokens() or not self.token_to_token_ranges[child]: continue

                    node_anchored_tokens.append(self.factor_strings[self.TOKENS][child])
                    (start, end) = self.token_to_token_ranges[child].split("-")
                    anchors.append((int(start), int(end)))
            if self.mode != "amr" and anchors:
                node["anchors"] = []
                for start, end in sorted(anchors):
                    if len(node["anchors"]) and node["anchors"][-1]["to"] in (start - 1, start):
                        node["anchors"][-1]["to"] = end
                    else:
                        node["anchors"].append({"from": start, "to": end})
            node_anchored_tokens = self.FORM_SEP.join(node_anchored_tokens)

            # Get node values (and label)
            for p, prop in enumerate(self.dataset.node_properties):
                value = mappings[self.NODE_VALUES][prop].id_to_string(self.factors[self.NODE_VALUES][i][p], encoded_from=node_anchored_tokens)
                if prop == "label" and value != "<none>":
                    node["label"] = value
                    # Revert the Sentence.FORM_SEP in eds labels back to +
                    if self.framework == "eds":
                        node["label"] = value.replace(Sentence.FORM_SEP, "+")
                    continue

                if value == "<none>": continue
                
                if not "properties" in node: node["properties"] = []
                node["properties"].append(prop)
                if not "values" in node: node["values"] = []
                for repeated_value in value.split(self.PROP_SEP):
                    if self.dataset.node_property_types[p] == bool:
                        node["values"].append(bool(repeated_value))
                    elif self.dataset.node_property_types[p] == int:
                        node["values"].append(int(repeated_value))
                    else:
                        node["values"].append(repeated_value)

            assert "id" in node, "Mandatory \"id\" not in node."
            nodes.append(node)

        # Print edges
        edges = []
        for e in range(self.n_edges()):
            if self.factors[self.EDGE_VALUES][e][0] in [Mapping.ROOT, Mapping.ANCHOR]: continue

            edge = dict()

            edge["source"] = int(self.factors[self.EDGE_PARENTS][e] - self.n_tokens())
            edge["target"] = int(self.factors[self.EDGE_CHILDREN][e] - self.n_tokens())
            if edge["source"] < 0 or edge["target"] < 0: continue
           
            for p, prop in enumerate(self.dataset.edge_properties):
                value = mappings[self.EDGE_VALUES][prop].id_to_string(self.factors[self.EDGE_VALUES][e][p])
                
                if prop == "label":
                    edge["label"] = value
                    continue

                if value == "<none>": continue
                
                if "properties" not in edge: edge["properties"] = []
                edge["properties"].append(prop)
                if not "values" in edge: edge["values"] = []
                for repeated_value in value.split(self.PROP_SEP):
                    if self.dataset.edge_property_types[p] == bool:
                        edge["values"].append(bool(repeated_value))
                    elif self.dataset.edge_property_types[p] == int:
                        edge["values"].append(int(repeated_value))
                    else:
                        edge["values"].append(repeated_value)
                        
            # Generate normals from edge label
            if self.mode == "amr" and "label" in edge:
                if edge["label"] == "mod":
                    edge["normal"] = "domain"
                elif edge["label"].endswith("-of-of") \
                     or edge["label"].endswith("-of") \
                       and edge["label"] not in {"consist-of" "subset-of"} \
                       and not edge["label"].startswith("prep-"):
                    edge["normal"] = edge["label"][:-3]

            assert "label" in edge, "Mandatory \"label\" not in edge."
            assert "source" in edge, "Mandatory \"source\" not in edge."
            assert "target" in edge, "Mandatory \"target\" not in edge."
            edges.append(edge)

        sentence_dict["tops"] = tops
        sentence_dict["nodes"] = nodes
        sentence_dict["edges"] = edges

        return json.dumps(sentence_dict, ensure_ascii=False)
