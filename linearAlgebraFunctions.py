import numpy as np

def dotProduct(vec1,vec2):
    return np.sqrt(vec1[0]*vec2[0]-vec1[1]*vec2[1]-vec1[2]*vec2[2]-vec1[3]*vec2[3])

    
def gram(e,px,py,pz,oneD=False):
    '''Take arrays of e,px,py,pz for N different objects and return an
    NxN gram matrix (dot product of all the possible pairs)'''

    assert len(e)==len(px)
    assert len(px)==len(py)
    assert len(py)==len(pz)

    g =[]
    for i in range(len(e)):
        g.append([])
        for j in range(len(e)):

            g[i].append(dotProduct([e[i],px[i],py[i],pz[i]], [e[j],px[j],py[j],pz[j]]))
    
    if oneD:
        g= list(np.array([g]).flatten())
    #TESTING
    # print 'e',e
    # print 'px',px
    # print 'py',py
    # print 'pz',pz
    # print g
    # exit()

    return g

def addGramToFlatDF(df,single=[],multi=[]):

    assert len(multi)+len(single) > 0

    gramVars = []
    gramNames = {}

    for v in single:
        gramVars.append([v+'_e',v+'_px',v+'_py',v+'_pz'])
        gramNames[v+'_e']=v
    for v,n in multi:
        for i in range(n):
            gramVars.append([v+'_e'+str(i),v+'_px'+str(i),v+'_py'+str(i),v+'_pz'+str(i)])
            gramNames[v+'_e'+str(i)]=v+str(i)


    for i,var1 in enumerate(gramVars):
        for j,var2 in enumerate(gramVars[i:]):

            df['gram_'+gramNames[var1[0]]+'_'+gramNames[var2[0]]]= df.apply(lambda row:\
                    dotProduct([row[var1[0]],row[var1[1]],row[var1[2]],row[var1[3]]],\
                        [row[var2[0]],row[var2[1]],row[var2[2]],row[var2[3]]]),axis=1)
        pass
    pass

    #return df
