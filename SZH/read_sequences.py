import re, os, sys

def read_protein_sequences(file):
    if os.path.exists(file) == False:
        print('Error: file %s does not exist.' % file)
        sys.exit(1)
    with open(file) as f:  # 打开数据文件
        records = f.read()
    if re.search('>', records) == None:  # 在字符串内查找模式匹配，只要找到第一个匹配然后返回
        print('Error: the input file %s seems not in FASTA format!' % file)
        sys.exit(1)
    records = records.split('>')[1:]  # 分隔出每个样本
    fasta_sequences = []
    for fasta in records:
        array = fasta.split('\n')  # ['Negative_1', 'TKTTRNSPDSISIP', '']
        header, sequence = array[0], array[1]
        label = '0' if header.split("_")[0] == "Positive" else '1'
        fasta_sequences.append([header, sequence, label])  # [样本名字，序列]
    return fasta_sequences
