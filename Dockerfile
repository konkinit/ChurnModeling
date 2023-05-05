FROM python:3.10-slim

ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

USER ${USERNAME}

ENV PATH="/home/${USERNAME}}/.local/bin"

COPY . ./churn_modeling

WORKDIR /churn_modeling

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 8005

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "./src/frontend/Onboarding.py", "--server.port", "8005"]
