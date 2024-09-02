FROM tensorflow/tensorflow:latest-gpu

ARG FUNCTION_DIR="/function"

RUN mkdir -p ${FUNCTION_DIR}
COPY . ${FUNCTION_DIR}
WORKDIR ${FUNCTION_DIR}

RUN pip install awslambdaric
RUN pip install -r requirements.txt

ENTRYPOINT [ "python", "-m", "awslambdaric" ]
CMD [ "lambda_function.lambda_handler" ]
