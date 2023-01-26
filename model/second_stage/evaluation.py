# coding=utf-8
from __future__ import print_function

import sys
import traceback
from tqdm import tqdm
from datasets.conala.bleu_score import compute_bleu
from datasets.FC2Code.fc2code_dataset import load_pseudo_code
from datasets.conala.util import tokenize_intent, decanonicalize_code, split_code

def decode(examples, model, args, verbose=False, **kwargs):
    ## TODO: create decoder for each dataset

    if verbose:
        print('evaluating %d examples' % len(examples))

    was_training = model.training
    model.eval()

    is_wikisql = args.parser == 'wikisql_parser'

    decode_results = []
    count = 0
    for example in tqdm(examples, desc='Decoding', file=sys.stdout, total=len(examples)):
        if is_wikisql:
            hyps = model.parse(example.src_sent, context=example.table, beam_size=args.beam_size)
        else:
            hyps = model.parse(example, context=None, beam_size=args.beam_size)
        decoded_hyps = []
        for hyp_id, hyp in enumerate(hyps):
            got_code = False
            try:
                hyp.code = model.transition_system.ast_to_surface_code(hyp.tree)
                got_code = True
                decoded_hyps.append(hyp)
            except:
                if verbose:
                    print("Exception in converting tree to code:", file=sys.stdout)
                    print('-' * 60, file=sys.stdout)
                    print('Example: %s\nIntent: %s\nTarget Code:\n%s\nHypothesis[%d]:\n%s' % (example.idx,
                                                                                             ' '.join(example.src_sent),
                                                                                             example.tgt_code,
                                                                                             hyp_id,
                                                                                             hyp.tree.to_string()), file=sys.stdout)
                    if got_code:
                        print()
                        print(hyp.code)
                    traceback.print_exc(file=sys.stdout)
                    print('-' * 60, file=sys.stdout)

        count += 1

        decode_results.append(decoded_hyps)

    if was_training: model.train()

    return decode_results

TYPE_NEED_TO_GENCODE = ["Effect" ,"While","ElseIf","If"]

def generate_code_from_pseudo_code(all_preds):
    actual_pseudo_dic = {}
    for pred in all_preds:
        actual_pseudo_dic[pred["actual_code_path"]] = pred["pseudo_code_path"]

    dic = {}
    for pred in all_preds:
        dic[pred["actual_code_path"]] = {}
    for pred in all_preds:
        dic[pred["actual_code_path"]][pred["guid"]] = pred
    actual_code_path_list = []

    actual_list = []
    for actual_code_path, guid_pred_dic in dic.items():
        actual_code_path_list.append(actual_code_path)
        with open(actual_code_path)as f:
            lines = f.readlines()
        actual = []
        for line in lines:
            lspace = len(line)-len(line.lstrip())
            assert lspace%4==0
            lspace/=4
            actual.extend(["INDENT"]*int(lspace))
            actual.extend(split_code(line))
            actual.append("NEWLINE")
        actual_list.append(actual)

    pred_list = []
    for actual_code_path, guid_pred_dic in dic.items():
        pseudo_code_path = actual_pseudo_dic[actual_code_path]
        NSpace_list, Type_list, Str_list, GUID_list = load_pseudo_code(pseudo_code_path)
        fc_pred = []

        for i in range(len(GUID_list)):
            # not all pseudo-code lines need to be converted into code, such as "else:", "continue".
            if Type_list[i] not in TYPE_NEED_TO_GENCODE:
                if "continue" not in Str_list[i]:
                    # word_list = tokenize_intent(Str_list[i])
                    word_list = split_code(Str_list[i])
                    _word_list = []
                    if "def" == word_list[0]:
                        assert word_list[2]=="("
                        ii = 3
                        while(ii < len(word_list)):
                            if word_list[ii]==',':
                                word_list.insert(ii+2, ":")
                                word_list.insert(ii+3, "int")
                                ii+=3
                            ii+=1
                        word_list.insert(-2, "->")
                        word_list.insert(-2, "int")
                    for w in word_list:
                        if w == '@':
                            _word_list.append("NEWLINE")
                        else:
                            _word_list.append(w)
                    fc_pred.extend(["INDENT"] * NSpace_list[i])
                    fc_pred.extend(_word_list)
            else:
                hypo_str = guid_pred_dic[GUID_list[i]]["decanonicalized_hypo_str"]
                if "pass" != hypo_str.strip():
                    if "if True:\n    pass" in hypo_str:
                        hypo_str = hypo_str.replace("if True:\n    pass\n", "")
                    if len(hypo_str)!=0:
                        hypo_str = split_code(hypo_str)

                        fc_pred.extend(["INDENT"] * NSpace_list[i])
                        for token in hypo_str:
                            if not token == "pass":
                                fc_pred.append(token)
                        fc_pred.append("NEWLINE")
        pred_list.append(fc_pred)
    return actual_list, pred_list, actual_code_path_list

from nltk.translate.bleu_score import sentence_bleu
def get_bleu(actual_list, pred_list):
    bleus = []
    for actual, pred in zip(actual_list, pred_list):
        bleus.append(sentence_bleu([actual], pred))
    return sum(bleus) / len(bleus) * 100, bleus

def get_bleu2(actual_list, pred_list):
    bleus = []
    for actual, pred in zip(actual_list, pred_list):
        bleus.append((compute_bleu([[actual]], [pred]))[0])
    return sum(bleus) / len(bleus) * 100

def get_bleu_8(actual_list, pred_list):
    bleus = []
    for actual, pred in zip(actual_list, pred_list):
        bleus.append((compute_bleu([[actual]], [pred] , max_order=8 ))[0])
    return sum(bleus) / len(bleus) * 100

def new_markdown_cell(source=None):
    return {
        'cell_type': 'markdown',
        'source': source if source else '',
        'metadata': {}
    }
def new_notebook(cells):
    return  {
        'cells': cells,
        'metadata': {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.6.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

def log_fc_preds_to_notebook(actual_list, pred_list, actual_code_path_list, outdir):
    cells = []
    for a, p, path in zip(actual_list, pred_list, actual_code_path_list):
        cells.append(new_markdown_cell("### " + path))

        cells.append(new_markdown_cell('actual: ' + " ".join(a).replace('NEWLINE', '\n')))
        cells.append(new_markdown_cell('predict: ' + " ".join(p).replace('NEWLINE', '\n')))

    json.dump(new_notebook(cells), open(outdir + '/fc_preds.ipynb', 'w'), indent=4)

def new_code_cell(source=None):
    return {
        'cell_type': 'code',
        'source': source if source else '',
        'metadata': {},
        'outputs': [],
        'execution_count': 0
    }

def get_filtered_preds(preds, function):
    cells = []
    # for p in preds:
    for p in filter(function, preds):
        # the logging format will be markdown with url, then src and tgt in a code block
        # followed by an empty code cell for annotation
        # cells.append(new_markdown_cell(f"### {p['id']}"))

        # cells.append(new_markdown_cell(p['url']))
        cells.append(new_markdown_cell('Source: ' + str(p['src_str'])))
        cells.append(new_markdown_cell('Source_Full: ' + str(p['src_str_full'])))
        if 'context' in p:
            cells.append(new_code_cell('Context: ' + p['context']))
        # these special tokens will be there in full code but not api sequence
        # so always replace them for readability
        # cells.append(new_code_cell('Target    : ' + p['tgt_str'].replace('NEWLINE', '\n').replace('INDENT', '    ')))
        cells.append(new_code_cell('Target    : ' + p['tgt_str']))
        # print(p['tgt_str'].replace('NEWLINE', '\n'))
        # cells.append(new_code_cell('Hypothesis: ' + p['hypo_str'].replace('NEWLINE', '\n').replace('INDENT', '    ')))
        cells.append(new_code_cell('Hypothesis: ' + p['hypo_str']))
        # cells.append(new_code_cell("whats correct: "))
        # cells.append(new_code_cell("whats incorrect: "))
        # cells.append(new_code_cell("whats needed: "))
        cells.append(new_code_cell('pseudo_code_path: ' + p['pseudo_code_path']))
        cells.append(new_code_cell('actual_code_path: ' + p['actual_code_path']))
        cells.append(new_code_cell('guid: ' + p['guid']))


    return cells

from pathlib import Path
def log_preds_to_notebook(preds, outdir, trunc=1000, func=None, code_key='code_tokens_clean'):
    ''' Writes out preds into notebooks, also a notebook with em corrects written otu.
    :param preds:
    :param outfile:
    :param randomize: the dev/test shuffled before hand so no need to
    :return:
    '''

    preds_trunc = preds[:trunc]
    Path(outdir).mkdir(parents=True, exist_ok=True)

    cells = get_filtered_preds(preds_trunc, lambda x: True)
    json.dump(new_notebook(cells), open(outdir + '/preds.ipynb', 'w'), indent=4)

    cells = get_filtered_preds(preds, lambda x: x['hypo_str'] == x['tgt_str'])
    json.dump(new_notebook(cells), open(outdir + '/preds_em_correct.ipynb', 'w'), indent=4)

    # cells = get_filtered_preds(preds, lambda x: any([t in x['tgt_str'] for t in x['hypo_str'].split()]))
    # json.dump(new_notebook(cells), open(outdir + '/preds_partial_correct.ipynb', 'w'), indent=4)

from os.path import dirname
import json

def evaluate(examples, parser, evaluator, args, verbose=False, return_decode_result=False, eval_top_pred_only=False):
    examples = examples
    decode_results = decode(examples, parser, args, verbose=verbose)

    eval_result = evaluator.evaluate_dataset(examples, decode_results, fast_mode=eval_top_pred_only, args=args)
    all_preds = []
    for decode_result,example in zip(decode_results,examples):
        dic = {}
        dic["src_str"] = example.src_sent
        dic["src_str_full"] = example.src_sent_full
        if args.incomming != 0 or args.outgoing != 0:
            dic["context"] = ""
            if args.direction:
                dic["context"] = "incomming\n" + str(example.src_incomming_context_guid_dic).replace("]}, ", "]}, \n")+ "\n"\
                  + "outgoing\n" + str(example.src_outgoing_context_guid_dic).replace("]}, ", "]}, \n")
            else:
                dic["context"] = str(example.src_context_guid_dic).replace("]}, ", "]}, \n")
        dic["tgt_str"] = example.tgt_code
        if len(decode_result)==0:
            dic["hypo_str"] = ""
        else:
            dic["hypo_str"]=decode_result[0].code
        dic["decanonicalized_hypo_str"] = decanonicalize_code(dic["hypo_str"],example.meta["slot_map"])
        dic["actual_code_path"]=example.actual_code_path
        dic["pseudo_code_path"] = example.pseudo_code_path
        dic["guid"] = example.fc_guid
        all_preds.append(dic)
    log_preds_to_notebook(all_preds, args.save_to)

    actual_list, pred_list, actual_code_path_list = generate_code_from_pseudo_code(all_preds)
    fc_bleu1, bleus = get_bleu(actual_list, pred_list)
    fc_bleu2 = get_bleu2(actual_list, pred_list)

    with open(args.save_to + r"\predict_fc.txt", "w")as f:
        f.write("actural\n\n"+str(actual_list)+"\n\n\n")
        f.write("pred\n\n"+str(pred_list))

    # fc_bleu8 = get_bleu_8(actual_list, pred_list)
    print(" flowchart_bleu1: " + str(fc_bleu1))
    print(" flowchart_bleu2: " + str(fc_bleu2))
    # print(" fc_bleu8: " + str(fc_bleu8))
    with open(args.save_to+r"\result.txt", "a")as f:
        f.write(" flowchart_bleu_1: " + str(fc_bleu1) + "\n")
        f.write(" flowchart_bleu_2: " + str(fc_bleu2) + "\n")
        # f.write(" fc_bleu8: " + str(fc_bleu8) + "\n")
        f.write(str(eval_result)+ "\n")
        # f.write(str(actual_code_path_list)+"\n")
        # f.write(str(bleus)+"\n")

        # preds_dir = dirname(args.path) + '/preds'
        log_fc_preds_to_notebook(actual_list, pred_list, actual_code_path_list, outdir=args.save_to)
    if return_decode_result:
        return eval_result, decode_results
    else:
        return eval_result
