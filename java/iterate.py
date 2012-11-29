#!/usr/bin/env python

import os

if __name__ == "__main__":
    # regularization constant
    C = ["0", "0.01"]
    
    # window size
    W = ["7", "9", "11"]
    
    # hidden layer size
    H = ["200", "800"]

    # learning
    A = ["0.01", "0.001"]
    
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
                    with open("results.txt", "a") as file:
                        file.write(results)
                        file.write("\n")
