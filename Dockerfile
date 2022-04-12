FROM python:3.9

WORKDIR opt

RUN pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ['python', 'api.py']