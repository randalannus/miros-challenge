FROM python:3.10-slim
ADD requirements.txt /build/requirements.txt
ADD init_clip.py /build/init_clip.py
WORKDIR /build/
RUN pip install -r requirements.txt
RUN python init_clip.py
WORKDIR /imageSearch/
ADD api.py search_service.py /imageSearch/
ADD /products/ /imageSearch/products/
ADD /bglessProducts/ /imageSearch/bglessProducts/
EXPOSE 5000
ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:5000", "api:app"]