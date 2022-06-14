import os

with open('requirements.txt') as openfileobject:
    for line in openfileobject:
            line = line.split("==")
            os.system("pip install " + line[0])


