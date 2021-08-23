import shutil
import os

files=os.listdir("./imgs_test")
for i in files:
    shutil.move("./imgs_test/"+i,"./imgs_test/"+i.split("_")[0]+".jpg")
