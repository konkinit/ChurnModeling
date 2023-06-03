FROM python:3.10-slim

ARG USERNAME=appuser
ARG USER_UID=1000

RUN useradd --uid $USER_UID -m $USERNAME

USER ${USERNAME}

ENV PATH="${PATH}:/home/${USERNAME}}/.local/bin"

COPY --chown=${USERNAME}:${USERNAME} . ./churn_modeling

WORKDIR /home/${USERNAME}/churn_modeling

RUN pip install --user --upgrade pip

RUN pip install --user -r ./churn_modeling/requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "./churn_modeling/src/frontend/Onboarding.py", "--server.port", "8501"]
