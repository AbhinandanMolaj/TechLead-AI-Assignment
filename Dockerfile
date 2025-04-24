FROM nvcr.io/nvidia/tritonserver:23.10-py3

# Install PyTorch and torchvision
RUN pip install torch torchvision torchaudio

# Install client libraries for testing
RUN pip install tritonclient[http] pillow numpy

# Set working directory
WORKDIR /app

# Copy model repository
COPY model_repository/ /app/model_repository/

# Copy the test image
COPY istockphoto-1412238848-612x612.jpg /app/test_image.jpg

# Expose Triton ports
EXPOSE 8000 8001 8002

# Command to start Triton
CMD ["tritonserver", "--model-repository=/app/model_repository"]