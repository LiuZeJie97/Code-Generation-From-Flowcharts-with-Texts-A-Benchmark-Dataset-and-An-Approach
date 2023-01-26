def load_flowchart_nodes(txt):
    fc_node = {}
    lines = txt.split("\n")
    for line in lines:
        if len(line)==0:
            continue
        guid,type,text = load_flowchart_node(line)
        fc_node[guid]={"type":type,"text":text}
    return fc_node

def load_flowchart_node(line):
    i = 0
    while line[i] != '=':
        i += 1
    guid = line[:i]
    pre = i
    while line[i] != ':':
        i += 1
    type = line[pre + 2:i]
    i += 2
    text = line[i:]
    return guid,type,text

# fc_node[guid]={"type":type,"text":text,"code_no":code_no}
# fc_ass.append({"source":source,"target":target,"guard":guard})
def fc_addid(fc_nodes, fc_ass):
    guid2id_dic = {}
    id = 0
    for guid, dic in fc_nodes.items():
        if guid not in guid2id_dic:
            guid2id_dic[guid] = id
            id += 1
        dic["id"] = guid2id_dic[guid]
        fc_nodes[guid] = dic

    _ass = []
    for i in range(len(fc_ass)):
        if fc_ass[i]["source"] in guid2id_dic and fc_ass[i]["target"] in guid2id_dic:
            dic = fc_ass[i]
            dic["source_id"] = guid2id_dic[fc_ass[i]["source"]]
            dic["target_id"] = guid2id_dic[fc_ass[i]["target"]]
            _ass.append(dic)
    return fc_nodes, _ass

def node2text(guid,dic):
    type,text = dic["type"], dic["text"]
    if len(type)>10 and type[:10] == "condition_" :
        type = "condition"
    return guid + "=>" + type + ": " + text
    #return guid+"=>"+type+": "+guid+ ":" + text

def load_flowchart_asss(txt):
    fc_ass = []
    lines = txt.split("\n")
    for line in lines:
        if len(line)==0:
            continue
        source,target,guard = load_flowchart_ass(line)
        fc_ass.append({"source":source,"target":target,"guard":guard})
    return fc_ass

def ass2text(ass):
    source, target, guard = ass["source"], ass["target"], ass["guard"]
    if guard == None:
        return source + "->" + target
    else:
        return source + "(" + guard + ")" + "->" + target

def load_flowchart_ass(line):
    i = 0

    while line[i] != '-' and line[i]!='(':
        i += 1
    source = line[:i]

    if line[i:i+5]=="(yes)":
        guard = "yes"
        i+=7
    elif line[i:i+4]=="(no)":
        guard = "no"
        i+=6
    elif line[i:i+6]=="(left)":
        guard = None
        i += 8
    elif line[i:i + 5] == "(right)":
        guard = None
        i += 9
    else:
        assert line[i:i+2]=="->"
        guard = None
        i+=2

    target = line[i:]
    return source,target,guard

def get_incomming_ass(fc_ass,guid,guard=None):
    ass_list = []
    for dic in fc_ass:
        if dic["target"]==guid:
            if guard == None or dic["guard"]==guard:
                ass_list.append(dic)
    return ass_list

def get_outgoing_ass(fc_ass,guid,guard=None):
    ass_list = []
    for dic in fc_ass:
        if dic["source"]==guid:
            if guard == None or dic["guard"]==guard:
                ass_list.append(dic)
    return ass_list

def add_ass(fc_ass, source,target,guard=None):
    fc_ass.append({"source":source,"target":target,"guard":guard})
    return fc_ass

def add_node(fc_node, t, node_type, text):
    max_num = 1
    guid = t + str(max_num)
    while guid in fc_node:
        max_num+=1
        guid = t + str(max_num)
    fc_node[guid]={"type": node_type, "text": text, "code_no": None}
    return fc_node,guid

def delete_ass(fc_ass, d_ass):
    _fc_ass = []
    for ass in fc_ass:
        if ass!=d_ass:
            _fc_ass.append(ass)
    return _fc_ass

def delete_node(fc_node,fc_ass, guid):
    fc_node.pop(guid)
    _fc_ass = []
    count = 0
    for dic in fc_ass:
        if dic["source"]!=guid and dic["target"]!=guid:
            _fc_ass.append(dic)
        else:
            count+=1
    return fc_node,_fc_ass,count


def split_line(line, split):
    line_list = list(line)
    s = split
    i = 0
    while i < len(line_list):
        isEn = ord(line_list[i]) in range(65, 91) or ord(line_list[i]) in range(97, 123)
        if isEn and i + 1 < len(line_list):
            isEn = ord(line_list[i + 1]) in range(65, 91) or ord(line_list[i + 1]) in range(97, 123)
        if (line_list[i].isdigit() or isEn):
            s -= 0.5
        else:
            s -= 1
        i += 1
        if s <= 0 and not isEn:
            line_list.insert(i, "\n")
            s = split

    return ''.join(line_list)

def fc2text(fc_node, fc_ass):
    res = ""
    for guid, dic in fc_node.items():
        res += node2text(guid, dic) + '\n'
    res += '\n'
    for ass in fc_ass:
        res += ass2text(ass) + "\n"
    return res

