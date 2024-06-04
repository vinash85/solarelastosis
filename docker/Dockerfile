From nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

RUN apt-get update
RUN apt install -y python3.8 python3-pip
RUN echo "alias python=python3.8" >> ~/.bashrc

ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get install -y openjdk-8-jdk
RUN apt install -y maven build-essential
RUN apt install -y openslide-tools
#RUN apt-get install -y python-openslide

RUN mkdir /home/tmp
COPY ./requirements/requirements.txt /home/tmp/
WORKDIR /home/tmp
RUN pip3 install Cython
RUN pip3 install -r requirements.txt
RUN apt install -y git
RUN git clone https://github.com/oval-group/smooth-topk.git
WORKDIR /home/tmp/smooth-topk
Run python3 setup.py install
WORKDIR /
RUN rm -rf /home/tmp

