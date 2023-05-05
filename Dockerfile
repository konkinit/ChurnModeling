FROM python:3.10-slim

RUN pip install --upgrade pip

ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

USER ${USERNAME}

ENV PATH="${PATH}:/home/${USERNAME}}/.local/bin"

WORKDIR /home/${USERNAME}

COPY --chown=${USERNAME}:${USERNAME} . ./churn_modeling

RUN pip install --user -r ./churn_modeling/requirements.txt

EXPOSE 8005

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "./churn_modeling/src/frontend/Onboarding.py", "--server.port", "8005"]
