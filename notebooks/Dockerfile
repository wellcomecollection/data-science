FROM jupyter/scipy-notebook

RUN pip install pip-tools
COPY requirements.in requirements.in
COPY requirements.common requirements.common
RUN pip-compile requirements.in requirements.common --output-file requirements.txt
RUN pip install -r requirements.txt

RUN rm -rf work requirements* 

