#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2018 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Lemmatizer."""


class Lemmatizer:
    SEPARATOR = " "

    @staticmethod
    def _min_edit_script(source, target, allow_copy=True):
        a = [[(len(source) + len(target) + 1, None)] * (len(target) + 1) for _ in range(len(source) + 1)]
        for i in range(0, len(source) + 1):
            for j in range(0, len(target) + 1):
                if i == 0 and j == 0:
                    a[i][j] = (0, "")
                else:
                    if allow_copy and i and j and source[i - 1] == target[j - 1] and a[i-1][j-1][0] < a[i][j][0]:
                        a[i][j] = (a[i-1][j-1][0], a[i-1][j-1][1] + "→")
                    if i and a[i-1][j][0] < a[i][j][0]:
                        a[i][j] = (a[i-1][j][0] + 1, a[i-1][j][1] + "-")
                    if j and a[i][j-1][0] < a[i][j][0]:
                        a[i][j] = (a[i][j-1][0] + 1, a[i][j-1][1] + "+" + target[j - 1])
        return a[-1][-1][1]

    @staticmethod
    def is_absolute_lemma_rule(lemma_rule):
        _, rule = lemma_rule.split(";", 1)
        return rule.startswith("a")

    @staticmethod
    def gen_absolute_lemma_rule(form, lemma, allow_copy=True):
        form = form.lower()

        previous_case = -1
        lemma_casing = ""
        for i, c in enumerate(lemma):
            case = "↑" if c.lower() != c else "↓"
            if case != previous_case:
                lemma_casing += "{}{}{}".format("¦" if lemma_casing else "", case, i if i <= len(lemma) // 2 else i - len(lemma))
            previous_case = case

        return lemma_casing + ";a" + lemma.lower()

    @staticmethod
    def gen_lemma_rule(form, lemma, allow_copy=True):
        form = form.lower()

        previous_case = -1
        lemma_casing = ""
        for i, c in enumerate(lemma):
            case = "↑" if c.lower() != c else "↓"
            if case != previous_case:
                lemma_casing += "{}{}{}".format("¦" if lemma_casing else "", case, i if i <= len(lemma) // 2 else i - len(lemma))
            previous_case = case
        lemma = lemma.lower()

        diff_rule = "d"
        if Lemmatizer.SEPARATOR in form and Lemmatizer.SEPARATOR not in lemma:
            tokens = form.split(Lemmatizer.SEPARATOR)
            best, start, end = 0, 0, len(tokens)
            for i in range(len(tokens)):
                for j in range(len(tokens), i, -1):
                    source = "".join(tokens[i:j])
                    ratio = 1 if source == lemma else 2 * Lemmatizer._min_edit_script(source, lemma).count("→") / (len(lemma) + len(source))
                    if ratio >= best: best, start, end = ratio, i, j
                    if best == 1: break
                if best == 1: break
            if best:
                diff_rule = "t{}¦{}¦".format(start, end)
                ori_form = form
                form = "".join(tokens[start:end])

        best, best_form, best_lemma = 0, 0, 0
        for l in range(len(lemma)):
            for f in range(len(form)):
                cpl = 0
                while f + cpl < len(form) and l + cpl < len(lemma) and form[f + cpl] == lemma[l + cpl]: cpl += 1
                if cpl > best:
                    best = cpl
                    best_form = f
                    best_lemma = l

        rule = lemma_casing + ";"
        if not best:
            rule += "a" + lemma
        else:
            rule += diff_rule + "{}¦{}".format(
                Lemmatizer._min_edit_script(form[:best_form], lemma[:best_lemma], allow_copy),
                Lemmatizer._min_edit_script(form[best_form + best:], lemma[best_lemma + best:], allow_copy),
            )
        return rule

    @staticmethod
    def apply_lemma_rule(form, lemma_rule):
        casing, rule = lemma_rule.split(";", 1)
        if rule.startswith("a"):
            lemma = rule[1:]
        else:
            form = form.lower()
            if rule.startswith("t"):
                start, end, rule = rule[1:].split("¦", maxsplit=2)
                start, end = int(start), int(end)
                tokens = form.split(Lemmatizer.SEPARATOR)
                if start >= len(tokens): start = len(tokens) - 1
                form = "".join(tokens[start : end])
            else:
                rule = rule[1:]

            rules, rule_sources = rule.split("¦"), []
            assert len(rules) == 2
            for rule in rules:
                source, i = 0, 0
                while i < len(rule):
                    if rule[i] == "→" or rule[i] == "-":
                        source += 1
                    else:
                        assert rule[i] == "+"
                        i += 1
                    i += 1
                rule_sources.append(source)

            try:
                lemma, form_offset = "", 0
                for i in range(2):
                    j, offset = 0, (0 if i == 0 else len(form) - rule_sources[1])
                    while j < len(rules[i]):
                        if rules[i][j] == "→":
                            lemma += form[offset]
                            offset += 1
                        elif rules[i][j] == "-":
                            offset += 1
                        else:
                            assert(rules[i][j] == "+")
                            lemma += rules[i][j + 1]
                            j += 1
                        j += 1
                    if i == 0:
                        lemma += form[rule_sources[0] : len(form) - rule_sources[1]]
            except:
                lemma = form

        for rule in casing.split("¦"):
            if rule == "↓0": continue # The lemma is lowercased initially
            case, offset = rule[0], int(rule[1:])
            lemma = lemma[:offset] + (lemma[offset:].upper() if case == "↑" else lemma[offset:].lower())

        return lemma
