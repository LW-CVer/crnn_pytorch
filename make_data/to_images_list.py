import os
import random
files=os.listdir('./imgs')
random.shuffle(files)
with open('image_list.txt',"w") as f:
    for i in files:
        label=i.split("_")[-1].split(".")[0]
        f.write(i+" "+label+'\n')
