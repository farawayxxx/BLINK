from string import punctuation
import re

def preprocess_English(content):
    text = re.sub(r'[{}]+'.format(punctuation),'',content)
    return text
def preprocess(content):
    text = preprocess_English(content).lower().replace(' ','')
    return text
str = 'WTF,are you crazy? You fucking bitch!'

print(str)
print(preprocess_English(str))
print(preprocess_English(str).lower())
print(preprocess_English(str).lower().replace(' ',''))

print(preprocess(str))