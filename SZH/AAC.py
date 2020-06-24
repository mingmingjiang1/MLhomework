import re
from collections import Counter

def AAC(fastas, **kw):
    AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    header = ['#', 'label']
    for i in AA:
        header.append(i)
    encodings.append(header)

    for i in fastas:
        name, sequence, label = i[0], i[1], i[2]
        count = Counter(sequence)
        for key in count:
            count[key] = count[key]/len(sequence)
        code = [name, label]
        for aa in AA:
            code.append(count[aa])
        encodings.append(code)
    return encodings
