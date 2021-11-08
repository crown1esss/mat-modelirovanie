import numpy as np
import math
from reading import Read
import integral
def main():

    #### tests
    frame = np.array([
        [2,2,0],
        [4,2,0],
        [4,4,0],
        [2,4,0]
    ])
    point_c = np.array(
        [3,3,0]
    )
    res = integral.sha_poschitay(frame , point_c)
    print(res)



if __name__ == '__main__':
    main()
