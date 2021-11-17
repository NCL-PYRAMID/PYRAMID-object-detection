FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
COPY env.yml .
RUN conda env create -f env.yml

# Make RUN commands use the new environment:
RUN echo "conda activate myenv" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure flask is installed:"
RUN python -c "import flask"

# The code to run when container is started:
COPY run.py entrypoint.sh ./
ENTRYPOINT ["./entrypoint.sh"]

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip \
  && apt-get update && apt-get install -y git

COPY requirements.txt /app/

RUN pip install -r app/requirements.txt

RUN apt-get update \
    && apt-get install -y libsm6 libxext6 libxrender-dev \
    && pip install opencv-python


COPY . /app

WORKDIR /app

CMD python -u app.py
