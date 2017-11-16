FROM python:3.5
MAINTAINER hulk.oh "snuboy89@gmail.com" 

# timezone 설정
ENV TZ ROK

# required library 설치
ADD requirements.txt .
RUN pip install -r requirements.txt


WORKDIR /api
ADD *.py ./
ADD *.pb ./

EXPOSE 1037
CMD ["python", "flask_service.py"]

