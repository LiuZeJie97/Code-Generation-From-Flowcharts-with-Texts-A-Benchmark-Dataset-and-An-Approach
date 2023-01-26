import datetime
import os
import random
import sys
from exp import run_exp
from my_util import MaxEpoch, EarlyStop

dropout=0.3
lr=0.001
lr_decay=0.5
max_epoch=80
beam_size=5
lstm='lstm'  # lstm
lr_decay_after_epoch=5
dif_context_mark = True
sort_context_by_sequnce = False

args_global = f"""
    --cuda 
    --evaluator conala_evaluator 
    --asdl_file asdl/lang/py3/py3_asdl.simplified.txt 
    --transition_system python3 
    --lstm {lstm} 
    --no_parent_field_type_embed 
    --no_parent_production_embed 
    --dropout {dropout} 
    --patience 5
    --max_num_trial 3 
    --glorot_init 
    --lr {lr} 
    --lr_decay {lr_decay} 
    --lr_decay_after_epoch {lr_decay_after_epoch} 
    --max_epoch {max_epoch} 
    --beam_size {beam_size} 
    --log_every 50 
    --valid_every_epoch 5
"""

"""
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
args_local_list = [
    {"incomming": 8, "outgoing":1, "direction":True, "context_type": "flowchart", "add_outgoing": False, "batch_size": 32, "hidden_size": 128,  "embed_size": 128, "action_embed_size": 128, "field_embed_size": 64, "type_embed_size": 64},
    {"incomming": 8, "outgoing":1, "direction":True, "context_type": "pseudo_code", "add_outgoing": False, "batch_size": 32, "hidden_size": 128,  "embed_size": 128, "action_embed_size": 128, "field_embed_size": 64, "type_embed_size": 64},
    # {"incomming": 8, "outgoing":0, "direction":True, "context_type": "flowchart", "add_outgoing": False, "batch_size": 32, "hidden_size": 128,  "embed_size": 128, "action_embed_size": 128, "field_embed_size": 64, "type_embed_size": 64},
    # {"incomming": 0, "outgoing":1, "direction":True, "context_type": "flowchart", "add_outgoing": False, "batch_size": 32, "hidden_size": 128,  "embed_size": 128, "action_embed_size": 128, "field_embed_size": 64, "type_embed_size": 64},
    {"incomming": 0, "outgoing":0, "direction":False, "context_type": "flowchart", "add_outgoing": False, "batch_size": 32, "hidden_size": 128,  "embed_size": 128, "action_embed_size": 128, "field_embed_size": 64, "type_embed_size": 64},
    {"incomming": 4, "outgoing":0, "direction":False, "context_type": "flowchart", "add_outgoing": False, "batch_size": 32, "hidden_size": 128,  "embed_size": 128, "action_embed_size": 128, "field_embed_size": 64, "type_embed_size": 64},
    # {"incomming": 4, "outgoing":1, "direction":True, "context_type": "flowchart", "add_outgoing": False, "batch_size": 32, "hidden_size": 128,  "embed_size": 128, "action_embed_size": 128, "field_embed_size": 64, "type_embed_size": 64},
    # {"incomming": 2, "outgoing":1, "direction":True, "context_type": "flowchart", "add_outgoing": False, "batch_size": 32, "hidden_size": 128,  "embed_size": 128, "action_embed_size": 128, "field_embed_size": 64, "type_embed_size": 64},

]

for dic in args_local_list:
    # try:
        time = datetime.datetime.now()
        time = time.strftime("%m-%d--%H-%M")
        model_dir_name = "result" + f'/{time}-{dic["incomming"]}_{dic["outgoing"]}'+ "-" + str(dic["direction"])+"-" + dic["context_type"]
        os.mkdir(model_dir_name)
        # model_dir_name = "result//" + r"06-21--17-58--info-8_1-False"
        preprocessed_fc2code_dataset = "preprocessed_dataset" + "/" + f'{dic["incomming"]}_{dic["outgoing"]}' + "_" + str(dic["direction"])+"_"+dic["context_type"]

        pre_args=args_global+\
             f"""
                --load_model {model_dir_name}\\model.bin
                --incomming {dic["incomming"]}
                --outgoing {dic["outgoing"]}
                --direction {dic["direction"]}
                --save-to {model_dir_name}
                --train_file {preprocessed_fc2code_dataset}\\train.all_0.bin
                --test_file {preprocessed_fc2code_dataset}\\dev.bin
                --dev_file {preprocessed_fc2code_dataset}\\dev.bin
                --vocab {preprocessed_fc2code_dataset}\\vocab.src_freq3.code_freq3.bin
                --seed {random.randint(0,9999)}
                --hidden_size {dic["hidden_size"]} 
                --embed_size {dic["embed_size"]} 
                --action_embed_size {dic["action_embed_size"]} 
                --field_embed_size {dic["field_embed_size"]} 
                --type_embed_size {dic["type_embed_size"]} 
                --batch_size {dic["batch_size"]} 
            """

        args=pre_args+"\n--mode train"
        sys.argv[1:] = args.split()
        try:
            run_exp()
        except (MaxEpoch, EarlyStop) as e:
            print(e)
        #
        args = pre_args + "\n--mode test"
        sys.argv[1:] = args.split()
        run_exp()
    # except Exception as e:
    #     print(e)


