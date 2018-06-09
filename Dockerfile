FROM continuumio/miniconda3

# Remove (large file sizes) MKL optimizations.
RUN conda install nomkl

RUN uname -a

# bash install
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# required by h5py
# RUN apt-get install software-properties-common
RUN apt-get update
RUN apt-get -y install libhdf5-dev

# Grab requirements.txt.
ADD ./webapp/requirements.txt /tmp/requirements.txt

RUN python3 --version

RUN pip --version

RUN pip install --upgrade pip

# Install dependencies
RUN pip install -r /tmp/requirements.txt

# Add our code
ADD ./webapp /opt/webapp/
WORKDIR /opt/webapp

RUN if [ -z "$PORT"]; \
    then export MY_PORT=5000; \
    else export MY_PORT=$PORT; \
    fi 

EXPOSE $MY_PORT

CMD python3 image_web.py $MY_PORT