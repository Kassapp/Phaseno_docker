FROM continuumio/miniconda3

WORKDIR /usr/src/app
COPY . .

RUN conda env create -f env.yml
RUN echo "source activate phaseno" > ~/.bashrc
RUN conda install nvidia/label/cuda-12.1.1::cuda
RUN conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
ENV PATH = /opt/conda/envs/phaseno/bin:$PATH

CMD ["conda", "run", "--no-capture-output", "-n", "phaseno","python", "Predict.py" ]
