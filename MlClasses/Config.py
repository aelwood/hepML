import os

class Config(object):

    def __init__(self,output=None):
        self.output=output
        self.config=''

    def addToConfig(self,name,variable):
        self.config = self.config+'\n'+name+'\t'+str(variable)

    def addLine(self,line):
        self.config = self.config+'\n'+line

    def saveConfig(self):
        if not self.output:
            print self.config
        else:
            if not os.path.exists(self.output): os.makedirs(self.output)
            outF = open(os.path.join(self.output,'config.txt'),'w')
            outF.write(self.config)
            outF.close()

