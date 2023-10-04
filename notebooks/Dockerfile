FROM jupyter/scipy-notebook:python-3.11.5

RUN pip install --upgrade pip pip-tools 
COPY requirements.* .
RUN pip-compile requirements.* --output-file requirements.txt
RUN pip install -r requirements.txt

RUN rm -rf work requirements* 
