# Use Lambda "Provided" base image for build
FROM public.ecr.aws/lambda/provided:al2023 AS build

RUN dnf install -y git tar
# Install UV directly instead of copying from GitHub Container Registry
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy the soure code into the image to install it and create the environment
# TODO do we want to copy the sdist or wheel instead?
COPY pyproject.toml uv.lock README.md LICENSE.md /src/poprox-recommender/
COPY src/ /src/poprox-recommender/src/
WORKDIR /src/poprox-recommender
RUN mkdir build

# install the production environment (will include poprox-recommender)
ENV UV_PYTHON_INSTALL_DIR=/opt/python
ENV UV_PROJECT_ENVIRONMENT=/opt/poprox
ENV UV_PYTHON=3.12
ENV UV_LOCKED=TRUE
RUN uv venv
RUN uv sync --no-editable --no-default-groups --group cpu --extra deploy

FROM public.ecr.aws/lambda/provided:al2023
ARG LOG_LEVEL=INFO

ENV VIRTUAL_ENV=/opt/poprox
ENV POPROX_MODELS=/opt/poprox/models

COPY --from=build /opt/ /opt/

# Download the punkt NLTK data
RUN /opt/poprox/bin/python3 -m nltk.downloader -d /opt/poprox/nltk_data punkt

# Bake the model data into the image
COPY models/ /opt/poprox/models/

# Make sure we can import the recommender
RUN /opt/poprox/bin/python -m poprox_recommender.api.main

# Copy the bootstrap script
COPY lambda-bootstrap.sh /var/runtime/bootstrap
RUN chmod 0555 /var/runtime/bootstrap

# Set the transformers cache to a writeable directory
ENV TRANSFORMERS_CACHE=/tmp/.transformers
ENV AWS_LAMBDA_LOG_LEVEL=${LOG_LEVEL}

# Run the bootstrap with our handler by default
CMD ["poprox_recommender.api.main.handler"]
