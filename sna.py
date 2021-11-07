# AUM SHREEGANESHAAYA NAMAH|| AUM SHREEHANUMATE NAMAH||
import constants
from PIL import Image
import numpy as np
import csv
import os
    
all = [ 'after_effecs', 'comments_rise', 'communities', 'pagerank', 'percent' ]

def saveImgs():

  if not os.path.exists(constants.ANALYSIS_SECONDARY_IMGS):
    os.makedirs(constants.ANALYSIS_SECONDARY_IMGS)

  for name in all:
    print(f"Doing for {name}.png")

    arr = []
    with open(constants.ANALYSIS_SECONDARY_DATA + name + ".csv") as f:
      for row in csv.reader(f):
        tmp_arr, three_arr = [], []
        for col in row:
          three_arr.append(int((float(col) + 0.5) * 255))
          if len(three_arr) == 3:
            tmp_arr.append([ three_arr[0], three_arr[1], three_arr[2] ])
            three_arr = []
        arr.append(tmp_arr)

    arr = np.array(arr, dtype=int)
    img = Image.fromarray(arr, 'RGB')
    img = Image.open(constants.ANALYSIS_SECONDARY_DATA + name + "_2.csv")
    # img.show()
    img.save(constants.ANALYSIS_SECONDARY_IMGS + name + ".png")
