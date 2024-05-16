# pull python base image
FROM python:3.10
# copy application files
WORKDIR /heart_app
ADD /trained_model/* ./trained_model/
ADD /app.py .
ADD /requirements.txt .
# specify working directory

# update pip
RUN pip install --upgrade pip
# install dependencies
RUN pip install -r requirements.txt
# expose port for application
EXPOSE 8001
# start fastapi application
CMD ["python", "app.py"]