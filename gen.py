alphalbet = 'QWERTYUIOPASDFGHJKLZXCVBNM'
def Convert(string): 
    list1=[] 
    list1[:0]=string 
    return list1 

import random
state = Convert(alphalbet)
from PIL import Image, ImageDraw, ImageFont

def strikethrough(text):
    return u'\u0336'.join(text) + u'\u0336'

with open('words.txt','w') as f:
    for i in range(1,5000):
        y = str(random.randint(0,9))
        x = str(random.choice(state))
        z = str(random.randint(0,9))
        t = str(random.randint(0,9))
        c = str(random.randint(0,9))
        v = str(random.randint(0,9))
        w = str(random.randint(0,9))
        b = str(random.randint(0,9))
        n = str(random.randint(0,9))
        m = str(random.choice(state))
        f.write(y+z+x+"-"+t+c+b+n+w+'\n')
        f.write(y+z+"-"+x+t+c+b+w+'\n')
        f.write(y+z+x+"-"+t+c+b+w+'\n')

        if int(y) > 7 :
            f.write(x+m+'-'+w+b+'-'+n+c+'\n')
        elif int(y) > 8 :
            f.write(y+z+'-'+t+c+v+'-'+x+m+n+b+'\n')
# print('\u0336'.join(alphalbet) + '\u0336')

