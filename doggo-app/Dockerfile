# Use an official Python runtime as a parent image
FROM python:3.7

# Set the working directory to /app
WORKDIR /doggo-app

COPY requirements.txt .
# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .
# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
# CMD ["python", "app.py"]
CMD ["gunicorn", "-b", ":80", "app:app", "--timeout", "240"]
