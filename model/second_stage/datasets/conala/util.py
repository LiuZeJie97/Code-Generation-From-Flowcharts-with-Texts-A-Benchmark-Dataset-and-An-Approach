# coding=utf-8

from __future__ import print_function

import itertools
import re
import ast
import astor
import nltk


# QUOTED_TOKEN_RE = re.compile(r"(?P<quote>''|[`'\"])(?P<string>.*?)(?P=quote)")

def compare_ast(node1, node2):
    if not isinstance(node1, str):
        if type(node1) is not type(node2):
            return False
    if isinstance(node1, ast.AST):
        for k, v in list(vars(node1).items()):
            if k in ('lineno', 'col_offset', 'ctx'):
                continue
            if not compare_ast(v, getattr(node2, k)):
                return False
        return True
    elif isinstance(node1, list):
        return all(itertools.starmap(compare_ast, zip(node1, node2)))
    else:
        return node1 == node2
import string
PUNC = string.punctuation

def split_code(line):
    nl_buffer = []
    if line[0] in PUNC:
        old_type = "punc"
    else:
        old_type = "value"

    line = line.lstrip()
    word_buff = ""
    for c in line:
        if c in PUNC:
            new_type = "punc"
        elif c == ' ':
            new_type = "space"
        elif c.isalpha():
            if ord(c) in range(65, 91) or ord(c) in range(97, 123):
                new_type = "value"
            else:
                new_type = "chinese"
        elif c.isdigit():
            new_type = "value"
        # elif c == "\n":
        #     new_type = "enter"
        else:
            #print("strange word:" + c + ".")
            continue
        if old_type == "chinese":
            nl_buffer.append(word_buff)
            word_buff = c
            old_type = new_type
        elif old_type == new_type:
            if (old_type == "punc"):
                temp = word_buff + c
                if temp in ["==", ">=", "<=", "->", "!=", "<<", ">>", "-=", "+=", "**"]:
                    word_buff += c
                else:
                    if word_buff != "":
                        nl_buffer.append(word_buff)
                    word_buff = c
            else:
                word_buff += c
        else:
            if old_type != "space":
                if word_buff != "":
                    nl_buffer.append(word_buff)

            word_buff = c
            old_type = new_type
    if word_buff != "":
        nl_buffer.append(word_buff)
    return nl_buffer

def tokenize_intent(intent):
    lower_intent = intent.lower()
    tokens = split_code(lower_intent)

    return tokens


def infer_slot_type(quote, value):
    if quote == '`' and value.isidentifier():
        return 'var'
    return 'str'


KEYWORD = ["while","for","if","elif","else","break","return","continue","def",'False', 'None', 'True', 'and', 'as', 'assert', 'class', 'del',
'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal',
'not', 'or', 'pass', 'raise', 'return', 'try', 'with', 'yield',"inputoutput","input","output","self","start","incomming","outgoing"]
def canonicalize_intent(intent_tokens_list):
    # handle the following special case: quote is `''`

    slot_map = dict()
    slot_map_rev  = {}
    var_id = 0
    str_id = 0
    canonicalize_intent_tokens_list = []

    f = lambda x='ddd': sum([1 if u'\u4e00' <= i <= u'\u9fff' else 0 for i in x]) > 0
    for intent_tokens in intent_tokens_list:
        canonicalize_intent_tokens = []
        for token in intent_tokens:
            if f(token) or not token.isalnum() or token in KEYWORD:
                canonicalize_intent_tokens.append(token)

            else:
                if token in slot_map_rev:
                    canonicalize_intent_tokens.append(slot_map_rev[token])
                else:
                    if token.isdigit():
                        slot_name = 'var_%d' % var_id
                        var_id += 1
                        slot_map[slot_name] = {'value': token.encode().decode('unicode_escape', 'ignore'),
                                               'quote': "",
                                               'type': 'var'}
                        slot_map_rev[token] = slot_name

                    else:
                        slot_name = 'str_%d' % str_id
                        str_id += 1
                        slot_map[slot_name] = {'value': token.encode().decode('unicode_escape', 'ignore'),
                                               'quote': "",
                                               'type': 'str'}
                        slot_map_rev[token] = slot_name

                    canonicalize_intent_tokens.append(slot_name)

        canonicalize_intent_tokens_list.append(canonicalize_intent_tokens)

            # slot_id = len(slot_map)
            # slot_name = 'slot_%d' % slot_id
            # # make sure slot_name is also unicode
            # slot_name = unicode(slot_name)



    return canonicalize_intent_tokens_list, slot_map


def replace_identifiers_in_ast(py_ast, identifier2slot):
    for node in ast.walk(py_ast):
        for k, v in list(vars(node).items()):
            if k in ('lineno', 'col_offset', 'ctx'):
                continue
            # Python 3
            # if isinstance(v, str) or isinstance(v, unicode):
            if isinstance(v, str):
                if v in identifier2slot:
                    slot_name = identifier2slot[v]
                    # Python 3
                    # if isinstance(slot_name, unicode):
                    #     try: slot_name = slot_name.encode('ascii')
                    #     except: pass

                    setattr(node, k, slot_name)


def is_enumerable_str(identifier_value):
    """
    Test if the quoted identifier value is a list
    """

    return len(identifier_value) > 2 and identifier_value[0] in ('{', '(', '[') and identifier_value[-1] in ('}', ']', ')')


def canonicalize_code(code, slot_map):
    string2slot = {x['value']: slot_name for slot_name, x in list(slot_map.items())}

    py_ast = ast.parse(code)
    replace_identifiers_in_ast(py_ast, string2slot)
    canonical_code = astor.to_source(py_ast).strip()

    # the following code handles the special case that
    # a list/dict/set mentioned in the intent, like
    # Intent: zip two lists `[1, 2]` and `[3, 4]` into a list of two tuples containing elements at the same index in each list
    # Code: zip([1, 2], [3, 4])

    entries_that_are_lists = [slot_name for slot_name, val in slot_map.items() if is_enumerable_str(val['value'])]
    if entries_that_are_lists:
        for slot_name in entries_that_are_lists:
            list_repr = slot_map[slot_name]['value']
            #if list_repr[0] == '[' and list_repr[-1] == ']':
            first_token = list_repr[0]  # e.g. `[`
            last_token = list_repr[-1]  # e.g., `]`
            fake_list = first_token + slot_name + last_token
            slot_map[fake_list] = slot_map[slot_name]
            # else:
            #     fake_list = slot_name

            canonical_code = canonical_code.replace(list_repr, fake_list)

    return canonical_code


def decanonicalize_code(code, slot_map):
    for slot_name, slot_val in slot_map.items():
        if is_enumerable_str(slot_name):
            code = code.replace(slot_name, slot_val['value'])

    slot2string = {x[0]: x[1]['value'] for x in list(slot_map.items())}
    py_ast = ast.parse(code)
    replace_identifiers_in_ast(py_ast, slot2string)
    raw_code = astor.to_source(py_ast).strip()
    # for slot_name, slot_info in slot_map.items():
    #     raw_code = raw_code.replace(slot_name, slot_info['value'])

    return raw_code
