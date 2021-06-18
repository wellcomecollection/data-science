FROM python:3.7-slim-buster

RUN pip install pip-tools
COPY requirements.in requirements.in
RUN pip-compile
RUN pip install -r requirements.txt

COPY src/ src/
COPY extract_colours.py extract_colours.py
COPY fetch_image_urls.py fetch_image_urls.py
