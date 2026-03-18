import json
import os
import re
import unicodedata  # to convert unicode to proper English text

#---------------------------------#
#----- instantiate variables -----#
#---------------------------------#

# define a list of words to remove (in lowercase)
remove_words = ['petmd'
                ,'chewy'
                ,'featured image' 
                ,'image credit'
                ,'istock' 
                ,'https://' 
                ,'.com'
                ,'.gov'
                ,'.edu'
                ,'ncbi.'
                ,'nlm.' 
                ,'citations '
                ,'., et al.'
                ,', vol.'
                ,', no.'
                ,', pp.'
                ,'by: '
               ]

#----------------------------#
#----- Define functions -----#
#----------------------------#

def split_into_sentences(text: str) -> list[str]:
    '''
    Purpose: to split text body into sentences
    @params text: a body of text (str)
    returns: list of sentences (list)

    note: If the text contains substrings "<prd>" or "<stop>", they would lead 
          to incorrect splitting because they are used as markers for splitting.
    '''
    alphabets= "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov|edu|me)"
    digits = "([0-9])"
    multiple_dots = r'\.{2,}'

    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]

    return sentences


def word_check(sentence, words):
    res = [all([k.lower() not in s.lower() for k in words]) for s in sentence]
    return [sentence[i] for i in range(0, len(res)) if res[i]]


def clean_data(directory, folder, remove_words, savedir):
    '''
    Purpose: to clean all text data files in a given folder
    @params directory: the directory path that contains different folders of data
    @params folder: the individual folder that contains data for a topic
    @params remove_words: a list of words to remove from text body
    @params savedir: the directory path of the save location
    returns: a dictionary where key = filename, value = the difference in text split length
    '''
    # initiate lists to store split text lengths for validation
    old_text_len = []
    new_text_len = []
    
    # get a list of files for a given folder
    files = [f for f in os.listdir(f'{directory}/{folder}') if '.json' in f]  # ignore files that are not JSON
    
    for file in files:
        try:
            file_obj = open(f'{directory}/{folder}/{file}', 'r')
            json_data = json.load(file_obj)
            
            # clean text data
            raw_text = json_data[-1]['articleBody']  # extract raw article body
            text = unicodedata.normalize("NFKD", raw_text).replace('&#039;',"'").replace('&quot;','')  # clean up unicode
            text_splits = split_into_sentences(text)
            clean_text = word_check(text_splits, remove_words)
            new_text = ' '.join(clean_text)
            
            # store text length stats
            old_text_len.append(len(text_splits))
            new_text_len.append(len(clean_text))
            
            # export and save cleaned text data
            savefilename = file.replace('.json','')

            os.makedirs(f'{savedir}/{folder}', exist_ok=True)
            writefile = open(f'{savedir}/{folder}/{savefilename}.txt', 'w')
            writefile.write(new_text)
            writefile.close()

            file_obj.close()

        except Exception as ex:
            return f'Error (at {directory}/{folder}/{file}): {ex}'
        
    # merge outputs for validation of data cleansing
    text_len_diff = [new_text_len[i] - old_text_len[i] for i in range(len(files))]
    output = dict(zip(files, text_len_diff))
        
    return output


#-----------------------#
#----- Main script -----#
#-----------------------#

# show current path of user
print(f'You are currently in the path: {os.getcwd()}')

# have the user input directory path that contains data folders
print(f'Input directory path that contains all data folders:')
dir = str(input())

print(f'Input directory path to save cleaned data:')
savedir = str(input())

# get a list of data folders
folders = os.listdir(dir)
folders = [f for f in folders if '.' not in f] # only keep folders that contain data and not files

for folder in folders: 
    res = clean_data(dir, folder, remove_words, savedir)

    if type(res) == str:  # print out error
        print(res)
    else:
        pass

