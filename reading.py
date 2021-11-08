import numpy as np
class Read:
    def __init__(self , name):
        self.name = name
        self.num_of_obj = 0
        self.num_of_moduls = []
        self.num_of_frames = []
        self.frames = []
        self.collocations = []
        self._read_(self.name)


    def _read_(self,name):

        src = open(f'data/{self.name}' ,'r')
        self.num_of_obj = int(src.readline())

        for obj in range (self.num_of_obj):
            self.num_of_moduls.append(int(src.readline()))


            for module in range(self.num_of_moduls[obj]):
                src.readline()
                self.num_of_frames.append(src.readline())
                self.frames.append(src.readline())

                for frame in range(int(self.num_of_frames[module])):
                    coll = list(map(float,src.readline().split()))
                    self.collocations.append(np.array(coll).reshape(4,3))
        print(self.collocations[5][1])



