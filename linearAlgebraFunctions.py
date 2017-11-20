import numpy as np

def dotProduct(vec1,vec2):
    return np.sqrt(vec1[0]*vec2[0]-vec1[1]*vec2[1]-vec1[2]*vec2[2]-vec1[3]*vec2[3])

def gram(e,px,py,pz):
    '''Take arrays of e,px,py,pz for N different objects and return an
    NxN gram matrix (dot product of all the possible pairs)'''

    assert len(e)==len(px)
    assert len(px)==len(py)
    assert len(py)==len(pz)

    g = []
    for i in range(len(e)):
        g.append([])
        for j in range(len(e)):

            g[i].append(dotProduct([e[i],px[i],py[i],pz[i]], [e[j],px[j],py[j],pz[j]]))
    
    #TESTING
    # print 'e',e
    # print 'px',px
    # print 'py',py
    # print 'pz',pz
    # print g
    # exit()

    return g
