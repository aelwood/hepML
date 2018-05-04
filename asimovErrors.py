from numpy import log,power,sqrt
eps=0.000001

# s,b cnts observed in experimentm 
# sig relative error on b

# Asimov significance
def Z(s,b,sig=None):
    #if sig == None: sig=eps
    return sqrt( -2.0/(sig*sig)*log( b/( b+(b*b)*(sig*sig))*(sig*sig)*s+1.0)+ \
           2.0*( b+s)*log(( b+s)*( b+(b*b)*(sig*sig))/( (b*b)+( b+s)*(b*b)*(sig*sig))))

# error propagation on Asimov significance
def eZ(s,es,b,eb,sig=None):
    #if sig == None: sig=eps
    #if sig < eps: sig=eps # to avoid stability issue in calculation
    return power(-(eb*eb)/( 1.0/(sig*sig)*log( b/( b+(b*b)*(sig*sig))*(sig*sig)*s+1.0)-( b+s)*log(( b+s)*( b+(b*b)*(sig*sig))/( (b*b)+( b+s)*(b*b)*(sig*sig))))*power( 1.0/( b/( b+(b*b)*(sig*sig))*(sig*sig)*s+1.0)/(sig*sig)*( 1.0/( b+(b*b)*(sig*sig))*(sig*sig)*s-b/power( b+(b*b)*(sig*sig),2.0)*(sig*sig)*( 2.0*b*(sig*sig)+1.0)*s)-( ( b+s)*( 2.0*b*(sig*sig)+1.0)/( (b*b)+( b+s)*(b*b)*(sig*sig))+( b+(b*b)*(sig*sig))/( (b*b)+( b+s)*(b*b)*(sig*sig))-( b+s)*( 2.0*( b+s)*b*(sig*sig)+2.0*b+(b*b)*(sig*sig))*( b+(b*b)*(sig*sig))/power( (b*b)+( b+s)*(b*b)*(sig*sig),2.0))/( b+(b*b)*(sig*sig))*( (b*b)+( b+s)*(b*b)*(sig*sig))-log(( b+s)*( b+(b*b)*(sig*sig))/( (b*b)+( b+s)*(b*b)*(sig*sig))),2.0)/2.0-1.0/( 1.0/(sig*sig)*log( b/( b+(b*b)*(sig*sig))*(sig*sig)*s+1.0)-( b+s)*log(( b+s)*( b+(b*b)*(sig*sig))/( (b*b)+( b+s)*(b*b)*(sig*sig))))*power( log(( b+s)*( b+(b*b)*(sig*sig))/( (b*b)+( b+s)*(b*b)*(sig*sig)))+1.0/( b+(b*b)*(sig*sig))*( ( b+(b*b)*(sig*sig))/( (b*b)+( b+s)*(b*b)*(sig*sig))-( b+s)*(b*b)*( b+(b*b)*(sig*sig))*(sig*sig)/power( (b*b)+( b+s)*(b*b)*(sig*sig),2.0))*( (b*b)+( b+s)*(b*b)*(sig*sig))-1.0/( b/( b+(b*b)*(sig*sig))*(sig*sig)*s+1.0)*b/( b+(b*b)*(sig*sig)),2.0)*(es*es)/2.0,(1.0/2.0))    

# n_s,n_b number of events used for testing etc.
def wghtd_Z(scale_s,n_s,scale_b,n_b,sig=None):
    return Z(scale_s*n_s,scale_b*n_b,sig)

def wghtd_eZ(scale_s,n_s,scale_b,n_b,sig=None):
    return eZ(scale_s*n_s,scale_s*sqrt(n_s),scale_b*n_b,scale_b*sqrt(n_b),sig)
    
# example usage
# Z(14,5,0.3)                   : 3.6149635359712184
# eZ(14,sqrt(14),5,sqrt(5),0.3) : 1.053697793635855
# wghtd_Z(2,7,10,0.5,0.3)       : 3.6149635359712184
# wghtd_eZ(2,7,10,0.5,0.3)      : 2.605193580282292

