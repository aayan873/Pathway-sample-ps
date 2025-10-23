FROM pathwaycom/pathway:latest

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY pathway.py /app/
# Copy the rest of the application code
# Command to run the Pathway script
CMD [ "python", "./pathway.py" ]
