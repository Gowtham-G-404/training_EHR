FROM python:3.10
RUN mkdir /train
WORKDIR /train/
ADD . /train/
RUN pip install -r requirements.txt
CMD ["python","/train/training.py"]