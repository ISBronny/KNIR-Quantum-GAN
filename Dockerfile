FROM  nvcr.io/nvidia/pytorch:22.07-py3

COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY ./knir_quantum_gan/ /workdir/knir_quantum_gan

RUN ls --recursive /workdir

WORKDIR "/workdir/knir_quantum_gan"

ENTRYPOINT ["python"]
CMD ["/workdir/knir_quantum_gan/__main__.py"]