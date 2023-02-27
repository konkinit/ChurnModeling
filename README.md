<h1 align="center">
  ML App for Churn Prediction
  <br/>
</h1>


<p align="center">This project aims to predict churn using data set from the Telco industry<br/> </p>

---
<p align="center">
<img alt="GitHub Workflow Status" src="https://img.shields.io/github/actions/workflow/status/konkinit/ChurnModeling/churnapp_test_build.yml?label=TEST%20%26%20DOCKER%20BUILD&style=for-the-badge">
</p>

<p align="center">
<img alt="LICENSE" src="https://img.shields.io/bower/l/p?color=blue&style=for-the-badge">  <img alt="Docker Size" src="https://img.shields.io/docker/image-size/kidrissa/churnapp?style=for-the-badge"> <img alt="Repo size" src="https://img.shields.io/github/repo-size/konkinit/ChurnModeling?label=REPO%20SIZE&style=for-the-badge">
</p>

---

## Onboarding 

```
docker pull kidrissa/churnapp:latest
```
```
docker run -p 8085:8085 -d kidrissa/churnapp:latest
```

## Project description
The project is setted up in order ta practice a tuto followed in an online course. Using 
a dataset from Telco industry the idea in the online course was to model churn by implementing an end-to-end
data science project from data preprocessing, data modeling, model comparaison to putting champion model into production. 
The project was done using SAS viya, a low-code environment. Consequently I decide to reimplement the project using the 
first two C's of Cloud Native approach. My aim is to build a streamlit app on which user can following the project 
end-to-end by entering some hyperparams to his convenience.

### Continous integration
One continous integration (CI) procedure with 2 jobs mainly is designed and launched at every push to the main branch:
-  Testing. Pytest collects the test from the tests folder and execute them
  -  if Testing goes through, a Docker Image is built and pushed onto the docker hub
