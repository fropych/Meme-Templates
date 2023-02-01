FROM python:3.9
COPY api/requirements.txt  requirements.txt
RUN python -m pip install --upgrade pip 
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt
WORKDIR /app
COPY  api/ .
COPY ./model/models/vit_model.pkl ./models/vit_model.pkl
COPY ./data/image_templates ./data/image_templates 
COPY ./data/image_templates.json ./data/image_templates.json 