FROM python:3.8

RUN pip install pip-tools
COPY pipeline/requirements.in requirements.in
RUN pip-compile
RUN pip install -r requirements.txt

COPY pipeline/src ./src
COPY pipeline/*.py ./
