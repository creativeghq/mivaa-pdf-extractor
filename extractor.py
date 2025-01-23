import pymupdf4llm
import pathlib
from pathlib import Path
import json
import os
# import pandas as pd
import fitz
import csv

# Extract PDF content as Markdown
def extract_pdf_to_markdown(file_name, page_number):
    
    page_number_list=None
    if not (page_number is None):
        page_number_list= [page_number]   
    
    return pymupdf4llm.to_markdown(file_name, pages=page_number_list)

def extract_pdf_tables(file_name,page_number,output_dir):

    doc = fitz.open(file_name)
    csv_path = os.path.join(output_dir, 'csv')
    os.makedirs(csv_path)
    page_count=1
    
    if not (page_number is None):
        page = doc[page_number]
        extract_table_to_csv(page.find_tables(),page_number, csv_path)
    else:
        for page in doc:    
            tabs = page.find_tables()
            # print(tabs.tables)
            extract_table_to_csv(tabs,page_count, csv_path)            
            page_count += 1

def extract_table_to_csv(tabs, page_count,output_dir):
    table_count=1
    
    for tab in tabs.tables:
        
        csv_file = output_dir + '/'+str(page_count)+'_'+str(table_count)+'.csv'
        
        data_file = open(csv_file, 'w', encoding="utf-8")

        # create the csv writer object
        csv_writer = csv.writer(data_file)
        
        for table in tab.extract():
            csv_writer.writerow(table)
        
        data_file.close()
        table_count += 1    


def extract_json_and_images(file_path, output_dir, page_number):

    page_number_list=None
    if not (page_number is None):
        page_number_list= [page_number-1]

    image_path = os.path.join(output_dir, 'images')
    os.makedirs(image_path)

    md_text_images = pymupdf4llm.to_markdown(doc=file_path,
                                             pages=page_number_list,
                                            page_chunks=True,
                                            write_images=True,
                                            image_path=image_path,
                                            image_format="jpg",
                                            dpi=200)
    
    pathlib.Path(output_dir+str("/output.json")).write_text(json.dumps(str(md_text_images)))