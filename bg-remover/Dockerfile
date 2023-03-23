FROM manifoldai/orbyter-ml-dev:3.5
ADD requirements.txt /build/requirements.txt
WORKDIR /build/
RUN pip install -r requirements.txt
ADD api.py bg_service.py /bgremover/
ADD /src/ /bgremover/src/
ADD /pretrained/ /bgremover/pretrained/
WORKDIR /bgremover/
EXPOSE 5000
ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:5000", "api:app"]