import shutil
import os

files=os.listdir("./imgs")
for i in files:
    shutil.move("./imgs/"+i,"./imgs/"+i.split("_")[0]+".jpg")
