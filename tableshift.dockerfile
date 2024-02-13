FROM ghcr.io/jpgard/tableshift
WORKDIR /tableshift
RUN mkdir /tableshift/tmp
COPY data /tableshift/tmp/
ENTRYPOINT ["/bin/bash"]