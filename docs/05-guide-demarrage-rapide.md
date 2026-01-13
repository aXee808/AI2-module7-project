# Guide de Démarrage Rapide

Ce guide vous permet de démarrer rapidement avec chaque approche MLOps.

---

## Prérequis Communs

### Tous les OS

1. **Python 3.9+** : https://www.python.org/downloads/
2. **Git** : https://git-scm.com/downloads
3. **Docker Desktop** : https://www.docker.com/products/docker-desktop

### Installation selon votre OS

#### macOS
```bash
# Homebrew (si pas installé)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Outils
brew install python git kubectl minikube

# Docker Desktop depuis https://www.docker.com/products/docker-desktop
```

#### Linux (Ubuntu/Debian)
```bash
# Python et Git
sudo apt update
sudo apt install python3 python3-pip python3-venv git curl

# Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

#### Windows (PowerShell Admin)
```powershell
# Installer Chocolatey (si pas installé)
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))

# Outils
choco install python git docker-desktop minikube kubernetes-cli -y
```

---

## Démarrage Rapide - En 5 minutes

### 1. Cloner et configurer

```bash
# Cloner le projet
git clone https://github.com/YOUR_USERNAME/mlops-stock-prediction.git
cd mlops-stock-prediction

# Créer l'environnement virtuel
python -m venv venv

# Activer (choisir selon votre OS)
source venv/bin/activate      # Mac/Linux
.\venv\Scripts\activate       # Windows PowerShell

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Entraîner le modèle

```bash
python src/train.py
```

### 3. Lancer l'API

```bash
python src/app.py
```

### 4. Tester

```bash
# Dans un autre terminal
curl http://localhost:5000/health
curl http://localhost:5000/predict/demo
```

---

## Approche 1 : GitHub Actions + ArgoCD + Kubernetes

### Étape 1 : Démarrer Kubernetes

```bash
# Démarrer Minikube
minikube start --cpus=4 --memory=4096

# Activer les addons
minikube addons enable ingress
minikube addons enable metrics-server

# Vérifier
kubectl get nodes
```

### Étape 2 : Installer ArgoCD

```bash
# Créer le namespace
kubectl create namespace argocd

# Installer ArgoCD
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Attendre que les pods soient prêts
kubectl wait --for=condition=Ready pods --all -n argocd --timeout=300s

# Obtenir le mot de passe admin
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d
echo ""

# Port-forward pour accéder à l'UI
kubectl port-forward svc/argocd-server -n argocd 8080:443 &
```

### Étape 3 : Déployer l'application

```bash
# Construire l'image Docker (avec le Docker de Minikube)
eval $(minikube docker-env)
docker build -t stock-prediction:local .

# Appliquer les manifests
kubectl apply -f kubernetes/

# Vérifier
kubectl get pods -n stock-prediction
kubectl port-forward svc/stock-prediction-service 5000:80 -n stock-prediction &

# Tester
curl http://localhost:5000/health
```

### Étape 4 : Configurer GitHub Actions

1. Dans votre repo GitHub, aller dans Settings > Secrets and variables > Actions
2. Ajouter les secrets :
   - `DOCKER_USERNAME`
   - `DOCKER_PASSWORD`
3. Push sur main pour déclencher le pipeline

---

## Approche 2 : Kubeflow

### Étape 1 : Installer Kubeflow

```bash
# Prérequis: Kubernetes cluster avec au moins 8GB RAM

# Option 1: Kubeflow sur Minikube (simplifié)
minikube start --cpus=4 --memory=8192 --disk-size=40g

# Installer kfctl
curl -LO https://github.com/kubeflow/kfctl/releases/download/v1.2.0/kfctl_v1.2.0-0-gbc038f9_linux.tar.gz
tar -xvf kfctl_v1.2.0-0-gbc038f9_linux.tar.gz
sudo mv kfctl /usr/local/bin/

# Déployer Kubeflow
export KF_NAME=kubeflow
export BASE_DIR=/opt
export KF_DIR=${BASE_DIR}/${KF_NAME}
export CONFIG_URI="https://raw.githubusercontent.com/kubeflow/manifests/v1.2-branch/kfdef/kfctl_k8s_istio.v1.2.0.yaml"

mkdir -p ${KF_DIR}
cd ${KF_DIR}
kfctl apply -V -f ${CONFIG_URI}

# Accéder à l'UI
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80 &
```

### Étape 2 : Compiler et exécuter le pipeline

```bash
# Installer le SDK Kubeflow Pipelines
pip install kfp

# Compiler le pipeline
cd kubeflow
python pipeline.py

# Uploader via l'UI ou CLI
kfp pipeline upload pipeline.yaml

# Créer un run depuis l'UI Kubeflow
```

---

## Approche 3 : MLflow (Le plus simple!)

### Étape 1 : Lancer le serveur MLflow

```bash
# Terminal 1: Serveur MLflow
mlflow server --host 0.0.0.0 --port 5001 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

### Étape 2 : Entraîner avec tracking

```bash
# Terminal 2: Entraînement
python mlflow/train_with_mlflow.py --model-type random_forest

# Comparer plusieurs modèles
python mlflow/train_with_mlflow.py --compare

# Enregistrer dans le Model Registry
python mlflow/train_with_mlflow.py --register
```

### Étape 3 : Voir les résultats

1. Ouvrir http://localhost:5001
2. Explorer les expériences, runs, métriques
3. Comparer les modèles
4. Promouvoir le meilleur modèle en "Production"

### Étape 4 : Servir le modèle

```bash
# Servir le modèle du registry
mlflow models serve -m "models:/stock-prediction/Production" -p 5002

# Tester
curl -X POST http://localhost:5002/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[105.2, 104.8, 103.5, 100.2, 105.5, 105.0, 104.0, 55.0, 0.8, 0.5, 0.3, 0.05, 0.6, 0.01, 0.03, 0.05, 0.015, 0.018, 1.2, 0.02, 0.7, 0.002]]}'
```

---

## Avec Docker Compose (Tout en un!)

La méthode la plus simple pour avoir tous les services :

```bash
# Lancer tous les services
docker-compose up -d

# Vérifier
docker-compose ps

# Accéder aux services
# - API:        http://localhost:5000
# - MLflow:     http://localhost:5001
# - Prometheus: http://localhost:9090
# - Grafana:    http://localhost:3000 (admin/admin)

# Tester l'API
curl http://localhost:5000/health
curl http://localhost:5000/predict/demo

# Voir les logs
docker-compose logs -f api

# Arrêter
docker-compose down
```

---

## Résumé des Commandes

| Action | Commande |
|--------|----------|
| **Setup** | `pip install -r requirements.txt` |
| **Train** | `python src/train.py` |
| **API** | `python src/app.py` |
| **Docker** | `docker-compose up -d` |
| **MLflow Server** | `mlflow server --port 5001` |
| **MLflow Train** | `python mlflow/train_with_mlflow.py` |
| **Kubernetes** | `kubectl apply -f kubernetes/` |
| **Kubeflow Pipeline** | `python kubeflow/pipeline.py` |

---

## Dépannage

### "Model not loaded"
```bash
# S'assurer que le modèle est entraîné
python src/train.py
ls models/  # Doit contenir model.joblib
```

### "Port already in use"
```bash
# Linux/Mac
lsof -i :5000 | grep LISTEN
kill -9 <PID>

# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### Docker ne démarre pas
```bash
# Vérifier le statut
docker info

# Redémarrer Docker Desktop
# ou sur Linux:
sudo systemctl restart docker
```

### Kubernetes pods en erreur
```bash
kubectl get pods -A
kubectl describe pod <pod-name> -n <namespace>
kubectl logs <pod-name> -n <namespace>
```

---

## Ressources

- [Documentation Flask](https://flask.palletsprojects.com/)
- [Documentation MLflow](https://mlflow.org/docs/latest/index.html)
- [Documentation Kubeflow](https://www.kubeflow.org/docs/)
- [Documentation Kubernetes](https://kubernetes.io/docs/home/)
- [ArgoCD Getting Started](https://argo-cd.readthedocs.io/en/stable/getting_started/)
