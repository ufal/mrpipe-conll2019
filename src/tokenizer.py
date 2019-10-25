#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2018 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Tokenizer."""


import re

class Tokenizer:
    def tokenize(self, sentence):
        return NotImplementedError()

class MorphoditaTokenizer(Tokenizer):
    def __init__(self):
        import ufal.morphodita
        self._tokenizer = ufal.morphodita.Tokenizer.newEnglishTokenizer()
        self._token_ranges = ufal.morphodita.TokenRanges()

    def tokenize(self, sentence):
        result = []

        self._tokenizer.setText(sentence)
        while self._tokenizer.nextSentence(None, self._token_ranges):
            for token_range in self._token_ranges:
                result.append((token_range.start, token_range.start + token_range.length))

        return result

class RuleBasedTokenizer(Tokenizer):
    def __init__(self, finegrained):
        self._finegrained = finegrained

        self._token = re.compile(
            re.sub(r"(?<!\\)[a-z]", lambda l: "[{}{}]".format(l.group(0).lower(), l.group(0).upper()),
                   r"(would|could|ca|is|are|ai|was|were|do|does|did|should|have|has|had|wo|might|need)(?=n'?t\b) | n'?t\b | " +
                   r"can(?=not\b) | wan(?=na\b) | got(?=ta\b) | '(s|d|m|re|ve|ll)\b | ") +
            (r"\d+,\d+,\d+ | \d+[,-]\d+ | " if not finegrained else "") +
            r" ---* | `+ | '+ | [.]+ | !+ | " +
            r"\w(" + (r"\w-[^-\s]|&|/|'S\w|'[A-RT-Z]|[.](?=.*\w)|" if not finegrained else "") + r"\w|\d)*(\$|) | \S ",
            re.X)

    def tokenize(self, sentence):
        result = []
        i = 0
        while i < len(sentence):
            match = self._token.match(sentence[i:])
            if match:
                result.append((i, i + match.end()))
                i += match.end()
            else:
                i += 1
        return result
