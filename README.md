# MIVAA PDF Extractor using PyMuPDF4LLM

## Introduction

Working with PDFs can be challenging, especially when dealing with documents containing tables, images, and metadata. This is particularly important for those in the AI field who are fine-tuning large language models (LLMs) or developing knowledge retrieval systems like RAG (Retrieval-Augmented Generation). Extracting accurate data is essential in these scenarios.

This solution contains a generic REST based API for extracting text, images, tables, and metadata data from PDF documents.

## Prerequisites

1. **Download the Repository**:
   - [Clone](https://github.com/MIVAA-ai/mivaa-pdf-extractor.git) or download the repository as a [ZIP](https://github.com/MIVAA-ai/mivaa-pdf-extractor/archive/refs/heads/main.zip) file.

2. **Unzip the Repository**:
   - Extract the downloaded ZIP file to a folder on your system.

3. **Install Docker**:
   - Ensure Docker is installed and running on your machine. You can download Docker [here](https://www.docker.com/).

## Installation 
This application is build in python using fastAPI and PyMuPDF4LLM

##### Direct installation
1. Install python and pip
2. Run following command:
```
    pip install -r requirements.txt
```
3. Run the following command:
```
    uvicorn main:app --host 0.0.0.0 --port 8000
```

##### Docker installation
1. Install docker
2. Build docker image
```
     docker build -t mivaa-pdf-extractor:1.0.0 .
```
3. Run docker container
```
     docker run -p 8000:8000 mivaa-pdf-extractor:1.0.0
```
## Run application

Launch swagger APIs:
```
    http://localhost:8000/docs
```
![Alt text](images/swagger_home.jpg)

## How to use APIs


#### Extract Markdown

If you have a PDF and simply want the content in a clean format that is compatible with Markdown
###### Input:
![Alt text](images/swagger_extract_markdown_input.jpg)

###### Response:
![Alt text](images/swagger_extract_markdown_response.jpg)



#### Extract Tables

Extracting tables from PDFs while preserving their formatting can be a challenging task. However, this API gracefully handles this process, ensuring that tables are extracted cleanly and returned as a CSV file.

###### Input:
![Alt text](images/swagger_extract_table_input.jpg)

###### Response:
![Alt text](images/swagger_extract_table_response.jpg)



#### Extract Images

The extraction of images along with text is often overlooked but incredibly significant, particularly for documents that contain figures, diagrams, or charts. Fortunately, this API seamlessly handles this process, ensuring that both images and text are extracted accurately.

###### Input:
![Alt text](images/swagger_extract_images_input.jpg)

###### Response:
![Alt text](images/swagger_extract_images_response.jpg)

## Additional Resources

- **Blog**:
  Read the detailed blog post about this application: [https://deepdatawithmivaa.com/2025/01/06/upgrade-your-well-log-data-workflow-vol-1-from-las-2-0-to-json/]

- **Demonstration Video**:
  Check out the video showcasing how to deploy and using this tool: [https://youtu.be/cYO-O94lHI8]
