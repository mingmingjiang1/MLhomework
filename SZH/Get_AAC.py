import argparse
from read_sequences import *
from save_file import *
from AAC import AAC

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--method", required=True, choices=['AAC'])
    parser.add_argument("--path", dest='filePath')
    parser.add_argument("--order", dest='order',
                        choices=['alphabetically', 'polarity', 'sideChainVolume', 'userDefined'])
    parser.add_argument("--userDefinedOrder", dest='userDefinedOrder')
    parser.add_argument("--format", choices=['csv', 'tsv', 'svm', 'weka', 'tsv_1'], default='svm',
                        help="the encoding type")
    parser.add_argument("--out")
    args = parser.parse_args()

    fastas = read_protein_sequences(args.file)
    userDefinedOrder = args.userDefinedOrder if args.userDefinedOrder != None else 'ACDEFGHIKLMNPQRSTVWY'
    userDefinedOrder = re.sub('[^ACDEFGHIKLMNPQRSTVWY]', '', userDefinedOrder)
    if len(userDefinedOrder) != 20:
        userDefinedOrder = 'ACDEFGHIKLMNPQRSTVWY'
    myAAorder = {
        'alphabetically': 'ACDEFGHIKLMNPQRSTVWY',
        'polarity': 'DENKRQHSGTAPYVMCWIFL',
        'sideChainVolume': 'GASDPCTNEVHQILMKRFYW',
        'userDefined': userDefinedOrder
    }
    myOrder = myAAorder[args.order] if args.order != None else 'ACDEFGHIKLMNPQRSTVWY'
    kw = {'path': args.filePath, 'order': myOrder, 'type': 'Protein'}
    cmd = args.method + '(fastas, **kw)'
    print('Descriptor type: ' + args.method)
    encodings = eval(cmd)
    out_file = args.out if args.out != None else 'encoding.txt'
    save_file(encodings, args.format, out_file)
