from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
from os import listdir
from os.path import isfile, join

def genLatestFile(path,name):
    # all file follows name.%n.tsv
    # name has to distinct
    allFiles = [f for f in listdir(path) if isfile(join(path, f)) 
                                            and (len(f.split('.')) == 3 and f.split('.')[0] == name)]
    
    curIdx = len(allFiles)
    newFileName = name + '.' + str(curIdx) + '.tsv'
    print("generate output path:", join(path, newFileName))
    return join(path, newFileName), newFileName

def export2DArray(data, path):
    #print(data)

    df = pd.DataFrame(data)

    with open(path, 'w+') as f:
        df.to_csv(f, sep='\t', index=False, header=False)

    print("Exported:", path)
    #print(df)

    return
