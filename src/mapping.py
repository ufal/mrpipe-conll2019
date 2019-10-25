#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2018 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Mapping from/to strings and ints."""


import numpy as np

from lemmatizer import Lemmatizer


class Mapping:
    """Mapping from/to strings and ints.
        strings_map: String -> string_id map.
        strings: String_id -> string list.

        Optionally:
            alphabet_map: Character -> char_id map.
            alphabet: Char_id -> character list.
            charseqs_map: String -> character_sequence_id map.
            charseqs: Character_sequence_id -> [characters], where character is an index
          to the dataset alphabet.

        Optionally:
            is_encoded: If True or None (undecided), strings are encoded as
            lemma rules using Lemmatizer.

        """

    # Special strings
    PAD = 0
    UNK = 1
    NONE = 2
    ROOT = 3
    ANCHOR = 4

    def __init__(self, train=None, include_characters=False, is_encoded=None):
        """Instantiates empty mapping.
        Arguments:
            train: Train mapping. If given, the words and alphabets are reused
                from the train mapping.
            include_characters: If True, character-level mappings are constructed.
            is_encoded: If True or none (undecided), strings are encoded as
                lemma rules using Lemmatizer.
        """

        # Is encoded as lemma rule? (train.is_encoded overrides argument)
        self.is_encoded = train.is_encoded if train else is_encoded
        if self.is_encoded is None:
            self.strings_original = set()

        # String to id map
        self.strings_map = train.strings_map if train else {
            "<pad>": self.PAD, "<unk>": self.UNK, "<none>": self.NONE, "<root>": self.ROOT, "<anchor>": self.ANCHOR}

        # Id to string map
        self.strings = train.strings if train else ["<pad>", "<unk>", "<none>", "<root>", "<anchor>"]

        # Characters
        self._include_characters = include_characters
        if self._include_characters:
            self.alphabet_map = train.alphabet_map if train else {"<pad>": self.PAD, "<unk>": self.UNK, "<none>": self.NONE, "<root>": self.ROOT}
            self.alphabet = train.alphabet if train else ["<pad>", "<unk>", "<none>", "<root>"]
            self.charseqs_map = {"<pad>": self.PAD, "<unk>": self.UNK, "<none>": self.NONE, "<root>": self.ROOT}
            self.charseqs = [[self.PAD], [self.UNK], [self.NONE], [self.ROOT]]
            self.charseq_ids = []

    @staticmethod
    def from_absolute_encodings(mapping, is_encoded):
        assert mapping.is_encoded is None or mapping.is_encoded == True
        assert is_encoded == True

        new = Mapping(include_characters=mapping._include_characters, is_encoded=True, train=None)
        for key in mapping.strings_map:
            if not key in ["<pad>", "<unk>", "<none>", "<root>", "<anchor>"]:
                if Lemmatizer.is_absolute_lemma_rule(key):
                    new.strings_map[key] = len(new.strings)
                    new.strings.append(key)
        return new

    def add_string(self, string, encoded_from=None, train=None): 
        """Add string to mapping and return id and optionally character id.
        Arguments:
            string: string.
            encoded_from: string, only used if is_encoded == True or
                is_encoded == None (undecided).
            train: Train mapping. If given, the words and alphabets are reused
                from the train mapping.
        Returns:
            If characters are allowed, a tuple (string id, character id).
            Otherwise only string id.
        """

        # Store strings when is_encoded == None
        if self.is_encoded is None:
            self.strings_original.add(string)

        # Encode string with lemma rule
        if self.is_encoded == None or self.is_encoded == True:
            # Do not encode special labels
            if not string in ["<pad>", "<unk>", "<none>", "<root>", "<anchor>"]:
                encoded_string = Lemmatizer.gen_absolute_lemma_rule(encoded_from, string)
                if encoded_string in self.strings_map:
                    string = encoded_string
                else:
                    string = Lemmatizer.gen_lemma_rule(encoded_from, string)

        # Word-level information
        if string not in self.strings_map:
            if train:
                string = '<unk>'
            else:
                self.strings_map[string] = len(self.strings)
                self.strings.append(string)
        
        if self._include_characters:
            # Character-level information
            if string not in self.charseqs_map:
                self.charseqs_map[string] = len(self.charseqs)
                self.charseqs.append([])
                for c in string:
                    if c not in self.alphabet_map:
                        if train:
                            c = '<unk>'
                        else:
                            self.alphabet_map[c] = len(self.alphabet)
                            self.alphabet.append(c)
                    self.charseqs[-1].append(self.alphabet_map[c])

        return (self.strings_map[string], self.charseqs_map[string]) if self._include_characters else self.strings_map[string]


    def id_to_encoded_string(self, string_id, train=None):
        """Return encoded string from int id."""

        return self.strings[string_id]


    def id_to_string(self, string_id, encoded_from=None, train=None):
        """Returns decoded string from int id."""

        string = self.strings[string_id]

        # Special strings are not encoded
        if string in ["<pad>", "<unk>", "<none>", "<root>", "<anchor>"]:
            return string
        else:
            # Decode string with lemma rule
            if self.is_encoded == None or self.is_encoded == True:
                return Lemmatizer.apply_lemma_rule(encoded_from, string)
            else:
                return string
