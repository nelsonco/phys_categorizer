# Necessary Imports
import os
import pandas as pd
import numpy as np
from io import StringIO

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

"""

Functions for Creating the Data Set:
MODIFY: Modify the directory to where you downloaded the files
MODIFY: Lines 72 and 77 with the directory you would like to saves
the files to.

PdfDataSet(String): input directory. output data frame
The input directory should be the directory that the documents were downloaded into.
This structure assumes that the dicectory contains a set of folder within which there are
pdfs documents.

This function searches through the folders and then processes each pdf file within the folder
by calling the PdftoTxt() function.
Each documen is stored into a data frame which includes the unique ID of the document
its category assigned by the ArXiv and the text mined by the pdfminer

PdftoTxt(String, String): input filename, parent folder. output  [id_num, category, text]
This function extracts the ID number, category and text of each document and saves it
into a text file. Using text extraction from the PdftoString() function and identifying
the category using the Category() function.

PdftoString(String): input path_to_file. output string (containing all text) or string ('error')
This function uses the pdfminer tools to extract the text from each pdf.

Category(String): input data (full text of a converted pdf), output string (category)
Identifies the category of the documents or assigns its category as unknown



"""
def PdfDataSet(directory):
    data = list()
    for folder in os.listdir(directory):
        print(folder)
        full_directory = directory+'\\'+folder
        for file in os.listdir(full_directory):
            if file.endswith("v1.pdf"):
                print(file)
                parent_folder = full_directory
                data_array = PdftoTxt(file, parent_folder)
                data.append(data_array)
    df = pd.DataFrame(data, columns = ['ID','Category','Text'])
    return df

categories = ['physics.acc-ph', 'physics.app-ph', 'physics.ao-ph', 'physics.atom-ph',
             'physics.atm-clus', 'physics.bio-ph', 'physics.chem-ph', 'physics.class-ph',
             'physics.comp-ph', 'physics.data-an', 'physics.flu-dyn', 'physics.gen-ph',
             'physics.geo-ph', 'physics.hist-ph', 'physics.ins-det', 'physics.med-ph',
             'physics.optics', 'physics.ed-ph', 'physics.soc-ph', 'physics.plasm-ph',
             'physics.pop-ph', 'physics.space-ph']

def PdftoTxt(filename, parent_folder):
    id_num, text  = '', ''
    id_num = filename[:-4] # Assign each document a unique identification number
    text = PdfToString(parent_folder + '\\' + filename) # get string
    category = Category(text) # Make sure the parent folder is labelled with the category name
    if category in categories:
        text_file = open(r'C:\Users\Court\Downloads\physics'+'\\'+category+'\\'+id_num+'.txt', "w",
                         encoding="utf-8")
        text_file.write(text)
        text_file.close()
    else:
        text_file = open(r'C:\Users\Court\Downloads\physics'+'\\unknown\\'+id_num+'.txt', "w",
                         encoding="utf-8")
        text_file.write(text)
        text_file.close()
    #np.savetxt(r'C:\Users\Court\Downloads\physics'+'\\'+category+'\\'+id_num+'.txt', text)
    return [id_num, category, text]

def PdfToString(path_to_file):
    try:
        txt = StringIO()
        file = open(path_to_file, 'rb') #read binary file
        parse = PDFParser(file)
        document = PDFDocument(parse)
        manage = PDFResourceManager()
        convert = TextConverter(manage, txt, laparams = LAParams())
        interpret = PDFPageInterpreter(manage, convert)
        for page in PDFPage.create_pages(document):
            interpret.process_page(page)
        file.close()
        return(txt.getvalue())
    except:
        return('error')



def Category(data):
    category = 'unknown'
    lines = data.split('\n') #get each line
    text_arr_clean = list()
    #Remove empty lines
    for element in lines:
        if element != '':
            text_arr_clean.append(element)
    i = 0
    while i < len(text_arr_clean):
        if text_arr_clean[i] == 'p' and text_arr_clean[i+1] == '[':
            category = ''
            x = i
            while text_arr_clean[x] != ']':
                category = category + text_arr_clean[x]
                x = x - 1
            break
        i =  i + 1
    return category

directory = r'C:\Users\Court\Downloads\physics\pdf'
df = PdfDataSet(directory)
