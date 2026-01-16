
# AI2 - Module 7 - Industrialisation deploiement

L’objectif de ce projet était d’industrialiser le déploiement d’une petite API. Afin de ne pas reprendre une api trop minimaliste, j’ai décidé d’implémenter un petit endpoint, et de m’appuyer sur un dataset (fichier csv) très connu de Kaggle : les données des passagers du Titanic.

Les étapes du projet se décline comme suit : 
- construire l’API avec une librairie que je connaissais (FastAPI)
- construire un conteneur Docker avec l’api à l’intérieur (Dockerfile)
- construire plusieurs conteneurs qui communiquent (docker-compose)
- déployer un pod avec le conteneur sous Kubernetes
- déployer plusieurs pods, un service, configmap, secret, et un persistent volume sur Kubernetes
- construire un chart Helm avec tous les fichiers yaml des ressources kubernetes souhaitées
- déployer le chart Helm stocké sur un github, sur Kubernetes, avec l’aide de d’ArgoCD

## Arborescence

```bash
├── .github
│   └── workflows
│       └── docker-image.yml
├── app
│   └── app.py
├── helm
│   └── titanic-app
│       ├── .helmignore
│       ├── Chart.yaml
│       ├── value.yaml
│       └── templates
│           ├── deployment.yaml
│           ├── service.yaml
│           ├── serviceaccount.yaml
│           ├── ingress.yaml
│           ├── hpa.yaml
│           ├── httproute.yaml
│           ├── NOTES.txt
│           ├── _helpers.tpl
│           └── tests
│               └── test-connection.yaml
├── k8s
│   ├── pv.yaml
│   ├── pvc.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── deployment.yaml
│   └── pod.yaml
├── rapport
│   └── rapport.pdf
├── README.md
├── docker-compose.yaml
├── requirements.txt
├── Dockerfile
├── .env
└── .gitignore
```