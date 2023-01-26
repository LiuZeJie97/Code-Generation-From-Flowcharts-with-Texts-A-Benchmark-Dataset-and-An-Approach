import argparse
import io
import json
import os
import pickle
import random
import sys
import numpy as np
import dgl
import distutils.util
from asdl.hypothesis import *
from asdl.lang.py3.py3_transition_system import python_ast_to_asdl_ast, asdl_ast_to_python_ast, Python3TransitionSystem
from asdl.transition_system import *
from components.action_info import get_action_infos
from components.dataset import Example
from components.vocab import Vocab, VocabEntry
from datasets.conala.evaluator import ConalaEvaluator
from datasets.conala.flowchart import load_flowchart_asss, get_incomming_ass, get_outgoing_ass, load_flowchart_nodes, fc_addid
from datasets.conala.util import *


assert astor.__version__ == '0.7.1'

def canonical_dup(examples,mode):
    pass

def preprocess_fc2code_dataset(args,grammar_file, src_freq=3, code_freq=3,vocab_size=20000, num_mined=0, out_dir='data/conala'):
    np.random.seed(1234)

    asdl_text = open(grammar_file).read()
    grammar = ASDLGrammar.from_text(asdl_text)
    transition_system = Python3TransitionSystem(grammar)

    print('process gold training data...')
    train_dataset = load_dataset(args,"train", firstk=None)
    valid_dataset = load_dataset(args, "valid", firstk=None)
    test_dataset = load_dataset(args, "test", firstk=None)

    os.mkdir(out_dir)
    train_examples = preprocess_dataset(args,train_dataset,args.incomming, args.outgoing, args.direction, transition_system)

    print(f'{len(train_examples)} training instances', file=sys.stderr)

    np.random.shuffle(train_examples)

    dev_examples = preprocess_dataset(args,valid_dataset,args.incomming, args.outgoing, args.direction, transition_system)
    print(f'{len(dev_examples)} dev instances', file=sys.stderr)

    print('process testing data...')

    test_examples = preprocess_dataset(args,test_dataset,args.incomming, args.outgoing, args.direction, transition_system)
    print(f'{len(test_examples)} testing instances', file=sys.stderr)
    corpus = [e.src_sent for e in train_examples]
    k = 50
    for i in range(k+1):
        corpus.append(["incomming","outgoning"])

    for unk_id in range(50):
        for i in range(src_freq):
            corpus.append([f"unk_{unk_id}"])

    src_vocab = VocabEntry.from_corpus(corpus, size=vocab_size,
                                       freq_cutoff=src_freq)
    source_more_than_k = VocabEntry.from_corpus(corpus, size=vocab_size,
                                                freq_cutoff=k)
    primitive_tokens = [map(lambda a: a.action.token,
                            filter(lambda a: isinstance(a.action, GenTokenAction), e.tgt_actions))
                        for e in train_examples]
    primitive_vocab = VocabEntry.from_corpus(primitive_tokens, size=vocab_size, freq_cutoff=code_freq)

    # generate vocabulary for the code tokens!
    code_tokens = [transition_system.tokenize_code(e.tgt_code, mode='decoder') for e in train_examples]

    code_vocab = VocabEntry.from_corpus(code_tokens, size=vocab_size, freq_cutoff=code_freq)

    vocab = Vocab(source=src_vocab, primitive=primitive_vocab, code=code_vocab, source_more_than_k = source_more_than_k)

    print('generated vocabulary %s' % repr(vocab), file=sys.stderr)
    with open(out_dir+r"\vocab_code.txt", "w") as f:
        f.write(str(vocab.code.word2id))
    with open(out_dir+r"\vocab_primitive.txt", "w") as f:
        f.write(str(vocab.primitive.word2id))
    with open(out_dir+r"\vocab_source.txt", "w") as f:
        f.write(str(vocab.source.word2id))
    action_lens = [len(e.tgt_actions) for e in train_examples]
    print('Max action len: %d' % max(action_lens), file=sys.stderr)
    print('Avg action len: %d' % np.average(action_lens), file=sys.stderr)
    print('Actions larger than 100: %d' % len(list(filter(lambda x: x > 100, action_lens))), file=sys.stderr)

    pickle.dump(train_examples, open(os.path.join(out_dir, 'train.all_{}.bin'.format(num_mined)), 'wb'))
    pickle.dump(dev_examples, open(os.path.join(out_dir, 'dev.bin'), 'wb'))
    pickle.dump(test_examples, open(os.path.join(out_dir, 'test.bin'), 'wb'))

    vocab_name = 'vocab.src_freq%d.code_freq%d.bin' % (src_freq, code_freq)
    pickle.dump(vocab, open(os.path.join(out_dir, vocab_name), 'wb'))

def load_pseudo_code(pseudu_code_path):
    with open(pseudu_code_path)as f:
        lines = f.readlines()
        _lines = []
        for line in lines:
            _lines.append(line[:-1])
        lines = _lines
        l = 0
        assert lines[0] == "NSpace"
        NSpace_list = []
        l += 1
        while lines[l] != "":
            NSpace_list.append(int(lines[l]))
            l += 1
        l += 1
        assert lines[l] == "GUID"
        l += 1
        GUID_list = []
        while lines[l] != "":
            GUID_list.append(lines[l])
            l += 1
        l += 1
        assert lines[l] == "Str"
        l += 1
        Str_list = []
        while lines[l] != "":
            Str_list.append(lines[l])
            l += 1
        l += 1
        assert lines[l] == "Type"
        l += 1
        Type_list = []
        while l < len(lines) and lines[l] != "":
            Type_list.append(lines[l])
            l += 1
    assert len(NSpace_list) == len(GUID_list)
    assert len(NSpace_list) == len(Str_list)
    assert len(NSpace_list) == len(Type_list)
    return NSpace_list, Type_list, Str_list, GUID_list

def get_guid_pseudo_code(NSpace_list, Type_list, Str_list, GUID_list):
    dic = {}
    for i in range(len(NSpace_list)):
        str = Str_list[i].replace('@', '\n')
        if str[-1] == "\n":
            str = str[:-1]
        dic[GUID_list[i]] = {"NSpace":NSpace_list[i],"str":str,"type":Type_list[i]}
    return dic

TYPE_NEED_TO_GENCODE = ["Effect" ,"While","ElseIf","If"]

def load_code(code_path):
    guid_code = {}
    with open(code_path)as f:
        _code_list = f.readlines()
        for code in _code_list:
            n = len(code) - len(code.lstrip())
            assert n % 4 == 0

        for i in range(len(_code_list)):
            code = _code_list[i]
            j = len(code) - 1
            while code[j] != "#":
                j -= 1
            current_guid = code[j + 1:-1]
            if current_guid != "None":
                code_split = code[:j - 4].strip()
                if current_guid in guid_code:
                    guid_code[current_guid] += "\n" + code_split
                else:
                    guid_code[current_guid] = code_split
    return guid_code

def load_flowchart(flowchart_path):
    with open(flowchart_path) as f_flowchart:
        flowchart_lines = f_flowchart.readlines()

    _fc_node = []
    l = 0
    while flowchart_lines[l] != "\n":
        _fc_node.append(flowchart_lines[l])
        l += 1
    fc_nodes = load_flowchart_nodes("".join(_fc_node))

    l += 1
    _fc_ass = []
    while l < len(flowchart_lines):
        _fc_ass.append(flowchart_lines[l])
        l += 1
    fc_ass = load_flowchart_asss("".join(_fc_ass))
    return fc_nodes, fc_ass

def load_pseudo_code_by_context(args,split):
    if split=="train":
        pseudu_code_dataset_path = "pseudo_code\\train"
        dataset_path = "D:\\Learning-to-Generate-Code-from-Flowcharts\\train"
    elif split=="valid":
        pseudu_code_dataset_path = "pseudo_code\\valid"
        dataset_path = "D:\\Learning-to-Generate-Code-from-Flowcharts\\valid"
    elif split=="test":
        pseudu_code_dataset_path = "pseudo_code\\test"
        dataset_path = "D:\\Learning-to-Generate-Code-from-Flowcharts\\test"
    else:
        raise

    question_list = os.listdir(dataset_path)
    sample_list = []
    total_flowchart = 0
    for question in question_list:
        file_list = os.listdir(dataset_path+"\\" + question)
        for file in file_list:
            if file[-12:] == "question.txt":
                continue
            total_flowchart +=1

            code_with_guid_path = dataset_path + "\\" + question + "\\" + file + "\\mapping_relations.txt"
            actual_code_path = dataset_path + "\\" + question + "\\" + file + "\\code.txt"
            pseudo_code_path = pseudu_code_dataset_path + "\\" + question + "\\" + file + "\\pseudo_code.txt"
            flowchart_path = dataset_path + "\\" + question + "\\" + file + "\\flowchart.txt"

            guid_code = load_code(code_with_guid_path)
            fc_nodes, fc_ass = load_flowchart(flowchart_path)
            NSpace_list, Type_list, Str_list, GUID_list = load_pseudo_code(pseudo_code_path)
            guid_pseudo_code = get_guid_pseudo_code(NSpace_list, Type_list, Str_list, GUID_list)

            for current_guid, pseudo_code in guid_pseudo_code.items():
                if pseudo_code["type"] not in TYPE_NEED_TO_GENCODE:
                    continue

                example = {}
                example["pseudo_code_path"] = pseudo_code_path
                example["actual_code_path"] = actual_code_path
                example["fc_guid"] = current_guid
                example["intent"] = pseudo_code["str"]
                if current_guid in guid_code:
                    example["snippet"] = guid_code[current_guid].strip()
                else:
                    example["snippet"] = ""

                def get_context(name, dirction_fun, max_dis):
                    if args.context_type == "flowchart":
                        context_guid_dic = {current_guid: {"dirction": "", "distance": 0, "text": guid_pseudo_code[current_guid]["str"]}}
                        dis = 0
                        for dis in range(1, max_dis + 1):
                            for guid, dic in context_guid_dic.copy().items():
                                for dirction, fun, source_target in dirction_fun:
                                    ass_list = fun(fc_ass, guid)
                                    for ass in ass_list:
                                        neibour_guid = ass[source_target]
                                        if neibour_guid in context_guid_dic:
                                            continue
                                        if neibour_guid in guid_pseudo_code:
                                            text = guid_pseudo_code[neibour_guid]["str"]
                                        else:
                                            node = fc_nodes[neibour_guid]
                                            text = node["type"]+ " " + node["text"]
                                        context_guid_dic[neibour_guid] = {"dirction":dirction, "distance":dis,  "text":text}
                        context_guid_dic_with_id, context_ass = fc_addid(context_guid_dic, fc_ass)
                        example[f"{name}_{dis}_id"] = context_guid_dic_with_id[current_guid]["id"]
                        assert example[f"{name}_{dis}_id"] == 0
                        src = [0]
                        dst = [0]
                        if name == "incomming":
                            for ass in context_ass:
                                src.append(ass["source_id"])
                                dst.append(ass["target_id"])
                        elif name == "outgoing":
                            for ass in context_ass:
                                src.append(ass["target_id"])
                                dst.append(ass["source_id"])
                        elif name == "full":
                            for ass in context_ass:
                                src.append(ass["target_id"])
                                dst.append(ass["source_id"])
                            for ass in context_ass:
                                src.append(ass["source_id"])
                                dst.append(ass["target_id"])
                        else:
                            raise
                        dgl_graph = dgl.graph((src, dst))

                    elif args.context_type == "pseudo_code":
                        assert current_guid in GUID_list
                        for i_duid, guid in enumerate(GUID_list):
                            if guid == current_guid:
                                break
                        src = [0]
                        dst = [0]
                        context_guid_dic_with_id = {current_guid: {"id":0,"dirction": "", "distance": 0, "text": guid_pseudo_code[current_guid]["str"]}}
                        id = 1
                        for dirction, fun, source_target in dirction_fun:
                            for dis in range(1, max_dis + 1):
                                neibour_guid = 'None'
                                while neibour_guid == 'None' or neibour_guid in context_guid_dic_with_id:
                                    if dirction == "incomming":
                                        i_duid -= 1
                                    elif dirction == "outgoing":
                                        i_duid += 1
                                    else:
                                        raise
                                    if i_duid>=len(GUID_list) or i_duid<0:
                                        break
                                    neibour_guid = GUID_list[i_duid]
                                if neibour_guid == 'None' or neibour_guid in context_guid_dic_with_id:
                                    break

                                text = guid_pseudo_code[neibour_guid]["str"]


                                context_guid_dic_with_id[neibour_guid] = {"dirction": dirction, "distance": dis, "text": text, "id" : id}
                                src.append(id-1)
                                dst.append(id)
                                id +=1
                        dgl_graph = dgl.graph((dst, src)).add_self_loop()
                    else:
                        raise

                    assert dgl_graph.nodes().size(0) == len(context_guid_dic_with_id)
                    assert len(context_guid_dic_with_id.keys())!=0
                    example[f"{name}_{max_dis}_graph"] = dgl_graph
                    example[f"{name}_{max_dis}_guid_dic"] = context_guid_dic_with_id.copy()

                if args.incomming == 0 and args.outgoing == 0:
                    example["full_intent"]=example["intent"]
                else:
                    if args.direction == False:
                        dirction_fun = [["incomming", get_incomming_ass, "source"], ["outgoing", get_outgoing_ass, "target"]]
                        get_context("full", dirction_fun, int(args.incomming))
                    else:
                        get_context("incomming", [["incomming", get_incomming_ass, "source"]], int(args.incomming))
                        get_context("outgoing", [["outgoing", get_outgoing_ass, "target"]], int(args.outgoing))
                # =========== context =============
                    example["full_intent"] = example["intent"]
                    if args.direction == False:
                        for guid, dic in example[f'full_{args.incomming}_guid_dic'].items():
                            if dic["distance"] > 0:
                                example["full_intent"] += ' incomming ' + dic["text"]

                    else:
                        if args.incomming != 0:
                            for guid, dic in example[f'incomming_{args.incomming}_guid_dic'].items():
                                if dic["distance"]>0:
                                    example["full_intent"] += ' incomming ' + dic["text"]

                        if args.outgoing != 0 and args.add_outgoing == True:
                            for guid, dic in example[f'outgoing_{args.outgoing}_guid_dic'].items():
                                if dic["distance"] > 0:
                                    example["full_intent"] += ' outgoing ' + dic["text"]
                sample_list.append(example)

    print(split + " flowchart: " + str(total_flowchart))
    return sample_list

# def load_pseudo_code_full(args,split):
#     if split=="train":
#         pseudu_code_dataset_path = ".\\pseudo_code\\train"
#         dataset_path = "D:\\Learning-to-Generate-Code-from-Flowcharts\\train"
#     elif split=="valid":
#         pseudu_code_dataset_path = ".\\pseudo_code\\valid"
#         dataset_path = "D:\\Learning-to-Generate-Code-from-Flowcharts\\valid"
#     elif split=="test":
#         pseudu_code_dataset_path = ".\\pseudo_code\\test"
#         dataset_path = "D:\\Learning-to-Generate-Code-from-Flowcharts\\test"
#     else:
#         raise
#
#     question_list = os.listdir(dataset_path)
#     sample_list = []
#     total_flowchart = 0
#     for question in question_list:
#         file_list = os.listdir(dataset_path+"\\" + question)
#         requirement = None
#
#         for file in file_list:
#             if file[-12:] == "question.txt":
#                 with open(dataset_path + "\\" + question + "\\" + file) as f:
#                     requirement = "".join(f.readlines())
#         if requirement == None:
#             continue
#
#         for file in file_list:
#             if file[-12:] == "question.txt":
#                 continue
#             total_flowchart +=1
#
#             code_with_guid_path = dataset_path + "\\" + question + "\\" + file + "\\mapping_relations.txt"
#             actual_code_path = dataset_path + "\\" + question + "\\" + file + "\\code.txt"
#             pseudo_code_path = pseudu_code_dataset_path + "\\" + question + "\\" + file + "\\pseudo_code.txt"
#             flowchart_path = dataset_path + "\\" + question + "\\" + file + "\\flowchart.txt"
#
#             with open(actual_code_path)as f:
#                 code = "".join(f.readlines())
#
#             example = {}
#             example["pseudo_code_path"] = pseudo_code_path
#             example["actual_code_path"] = actual_code_path
#             example["fc_guid"] = "current_guid"
#             example["intent"] = requirement
#             example["snippet"] = code
#             example["full_intent"]=example["intent"]
#
#             sample_list.append(example)
#
#
#     print(split + " flowchart: " + str(total_flowchart))
#     return sample_list

def load_dataset(args,split, firstk=None):
    dataset = load_pseudo_code_by_context(args,split)
    # dataset = load_pseudo_code_full(args,split)
    dataset = dataset
    for i in range(len(dataset)):
        if len(dataset[i]["snippet"]) == 0:
            dataset[i]["snippet"] = "pass"
        elif dataset[i]["snippet"][:4]=="elif":
            dataset[i]["snippet"] = "if True: pass\n" + dataset[i]["snippet"]

        if dataset[i]["snippet"][-1]==":":
            dataset[i]["snippet"] += "pass"
    if firstk:
        dataset = dataset[:firstk]
    for example_dic in dataset:
        preprocess_example(args,example_dic)
    return dataset

def preprocess_dataset(args, dataset, incomming, outgoing, direction, transition_system):
    evaluator = ConalaEvaluator(transition_system)
    examples = []
    for i, example_dic in enumerate(dataset):
        python_ast = ast.parse(example_dic['canonical_snippet'])
        canonical_code = astor.to_source(python_ast).strip()
        tgt_ast = python_ast_to_asdl_ast(python_ast, transition_system.grammar)
        tgt_actions = transition_system.get_actions(tgt_ast)

        # sanity check
        hyp = Hypothesis()
        for t, action in enumerate(tgt_actions):
            assert action.__class__ in transition_system.get_valid_continuation_types(hyp)
            if isinstance(action, ApplyRuleAction):
                assert action.production in transition_system.get_valid_continuating_productions(hyp)
            hyp = hyp.clone_and_apply_action(action)

        assert hyp.frontier_node is None and hyp.frontier_field is None
        hyp.code = code_from_hyp = astor.to_source(asdl_ast_to_python_ast(hyp.tree, transition_system.grammar)).strip()
        assert code_from_hyp == canonical_code

        decanonicalized_code_from_hyp = decanonicalize_code(code_from_hyp, example_dic['slot_map'])
        assert compare_ast(ast.parse(example_dic['snippet']), ast.parse(decanonicalized_code_from_hyp))
        assert transition_system.compare_ast(transition_system.surface_code_to_ast(decanonicalized_code_from_hyp),
                                             transition_system.surface_code_to_ast(example_dic['snippet']))
        tgt_action_infos = get_action_infos(example_dic['canonical_full_intent'], tgt_actions)
        example_dic["tgt_actions"]=tgt_action_infos
        example_dic["tgt_code"]=canonical_code
        example_dic["tgt_ast"]=tgt_ast
        example_dic["idx"]=f'{i}'
        # except (AssertionError, SyntaxError, ValueError, OverflowError) as e:
        #     skipped_list.append(example_dic['question_id'])
        #     continue
        # log!
        # f.write(f'Example: '+f'{i}'+'\n')
        # f.write(f"Original Utterance: {example_dic['intent']}\n")
        # f.write(f"Original Snippet: {example_dic['snippet']}\n")
        # f.write(f"cannical_tokens: " + str(example_dic["canonical_tokens"]))
        # f.write(f"\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        example = Example(example_dic, incomming, outgoing, direction, f'{i}', canonical_code, tgt_ast, tgt_action_infos)
        assert evaluator.is_hyp_correct(example, hyp)
        examples.append(example)

    # print('Skipped due to exceptions: %d' % len(skipped_list), file=sys.stderr)
    return examples

def preprocess_example(args,example_json):
    intent_tokens = [tokenize_intent(example_json["intent"])]
    guid_list = [example_json["fc_guid"]]
    if args.direction == False:
        depth = int(args.incomming)
        if depth!=0:
            for guid, context in example_json[f'full_{depth}_guid_dic'].items():
                intent_tokens.append(tokenize_intent(context["text"]))
                guid_list.append(guid)
    else:
        if args.incomming!= 0:
            for guid, context in example_json[f'incomming_{args.incomming}_guid_dic'].items():
                intent_tokens.append(tokenize_intent(context["text"]))
                guid_list.append(guid)
        if args.outgoing != 0:
            for guid, context in example_json[f'outgoing_{args.outgoing}_guid_dic'].items():
                intent_tokens.append(tokenize_intent(context["text"]))
                guid_list.append(guid)

    intent_tokens.append(tokenize_intent(example_json["full_intent"]))
    guid_list.append(example_json["fc_guid"])

    assert example_json["fc_guid"] == guid_list[0]
    assert example_json["fc_guid"] == guid_list[-1]

    canonical_intent_tokens_list, slot_map = canonicalize_intent(intent_tokens)
    example_json['slot_map'] = slot_map
    example_json["canonical_intent"] = canonical_intent_tokens_list[0]
    example_json["canonical_full_intent"] = canonical_intent_tokens_list[-1]

    canonical_intent_tokens_list = canonical_intent_tokens_list[1:-1]
    guid_list=guid_list[1:-1]
    if args.incomming != 0 or args.outgoing != 0:
        canonical_tokens = {}
        for guid, canonical_intent_tokens in zip(guid_list,canonical_intent_tokens_list):
            canonical_tokens[guid] = canonical_intent_tokens
        example_json["canonical_context"] = canonical_tokens

    canonical_snippet = canonicalize_code(example_json['snippet'], slot_map)
    example_json['canonical_snippet'] = canonical_snippet

    decanonical_snippet = decanonicalize_code(canonical_snippet, slot_map)
    reconstructed_snippet = astor.to_source(ast.parse(example_json['snippet'])).strip()
    reconstructed_decanonical_snippet = astor.to_source(ast.parse(decanonical_snippet)).strip()
    assert compare_ast(ast.parse(reconstructed_snippet), ast.parse(reconstructed_decanonical_snippet))
    return example_json

def generate_dataset():
    arg_parser = argparse.ArgumentParser()
    #### TranX configuration ####
    arg_parser.add_argument('--pretrain', type=str, help='Path to pretrain file')
    arg_parser.add_argument('--out-dir', type=str, required=True, help='Path to output file')
    arg_parser.add_argument('--topk', type=int, default=0, help='First k number from mined file')
    arg_parser.add_argument('--freq', type=int, default=3, help='minimum frequency of tokens')
    arg_parser.add_argument('--vocabsize', type=int, default=20000, help='First k number from pretrain file')
    arg_parser.add_argument('--include_api', type=str, help='Path to apidocs file')
    ### FC2Code config ###
    arg_parser.add_argument('--incomming', type=int, required=True,help='')
    arg_parser.add_argument('--outgoing', type=int, required=True,help='')
    arg_parser.add_argument('--direction', type=lambda x:bool(distutils.util.strtobool(x)), required=True,help='')
    arg_parser.add_argument('--add-outgoing', type=str, required=True,help='')
    arg_parser.add_argument('--context-type', type=str, required=True,help='')
    args = arg_parser.parse_args()

    # the json files can be downloaded from http://conala-corpus.github.io
    preprocess_fc2code_dataset(args,grammar_file='./asdl/lang/py3/py3_asdl.simplified.txt',
                              src_freq=args.freq, code_freq=args.freq,
                              vocab_size=args.vocabsize,
                              num_mined=args.topk,
                              out_dir=args.out_dir)
r"""
    Parameters
    ----------
    incomming : int
        d_org, the window size on G_org. When direction is set to False, incomming represents the window size on the undirected graph.
    outgoing : int
        d_rev, the window size on G_rev.
    direction : bool
        Setting to False means that we do not construct the “reversed flowchart” and treat the flowchart as an undirected graph. 
    context_type: str
        We can get the neighbors of each node according to the "flowchart" or to the "pseudo_code".
"""
arg_list = [
    {"incomming": 8, "outgoing": 1, "direction": True, "add_outgoing": False, "context_type": "flowchart"},
    {"incomming": 8, "outgoing": 1, "direction": True, "add_outgoing": False, "context_type": "pseudo_code"},
    # {"incomming": 8, "outgoing": 0, "direction": True, "add_outgoing": False, "context_type": "flowchart"},
    # {"incomming": 0, "outgoing": 1, "direction": True, "add_outgoing": False, "context_type": "flowchart"},
    {"incomming": 0, "outgoing": 0, "direction": False, "add_outgoing": False, "context_type": "flowchart"},
    {"incomming": 4, "outgoing": 0, "direction":False, "add_outgoing": False, "context_type": "flowchart"},
    # {"incomming": 4, "outgoing": 1, "direction": True, "add_outgoing": False, "context_type": "flowchart"},
    # {"incomming": 2, "outgoing": 1, "direction": True, "add_outgoing": False, "context_type": "flowchart"},
]

if __name__ == '__main__':
    print(os.getcwd())
    for arg_dic in arg_list:
        # try:
            args = f"""
                --incomming {arg_dic["incomming"]}
                --outgoing {arg_dic["outgoing"]}
                --direction {arg_dic["direction"]}
                --out-dir preprocessed_dataset\\{arg_dic["incomming"]}_{arg_dic["outgoing"]}_{arg_dic["direction"]}_{arg_dic["context_type"]}
                --add-outgoing {arg_dic["add_outgoing"]}
                --context-type {arg_dic["context_type"]}
            """
            sys.argv[1:] = args.split()
            generate_dataset()
        # except Exception as E:
        #     print(E)