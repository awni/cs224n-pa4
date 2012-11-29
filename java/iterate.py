#!/usr/bin/env python

import difflib
import re
import os
import sys
import traceback

if __name__ == "__main__":
    # regularization constant
    C = ["0", "0.01", "0.001", "0.0001", "0.00001"]
    
    # window size
    W = ["1", "3", "5", "7", "9", "11", "13", "15", "17"]
    
    # hidden layer size
    H = ["100", "200", "400", "600", "800", "1000"]

    # learning
    A = ["0.01", "0.001", "0.0001"]
    
    # regularization constant
    C = ["0"]
    
    # window size
    W = ["1"]
    
    # hidden layer size
    H = ["3"]

    # learning
    A = ["0.01"]
    
    command = "java -cp extlib/ejml.jar:classes -Xmx4G cs224n.deep.NER ../data/train ../data/dev -print %s %s %s %s"
    for i in C:
        for j in W:
            for k in H:
                for l in A:
                    print command % (i, j, k, l)
                    f = os.popen(command % (i, j, k, l))
                    results = "C = %s W = %s H = %s A = %s \n" % (i, j, k, l)
                    for line in f:
                        results += line
                    print results
