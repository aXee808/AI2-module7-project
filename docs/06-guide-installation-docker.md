# Guide d'Installation et Setup Docker

Ce guide vous permet de tester l'ensemble du projet MLOps en utilisant uniquement Docker, sans avoir besoin d'installer Python, Kubernetes ou d'autres outils localement.

---

## Table des Matières

1. [Prérequis](#prérequis)
2. [Installation de Docker](#installation-de-docker)
3. [Démarrage Rapide](#démarrage-rapide)
4. [Architecture des Services](#architecture-des-services)
5. [Tester Chaque Service](#tester-chaque-service)
6. [Commandes Utiles](#commandes-utiles)
7. [Configuration Avancée](#configuration-avancée)
8. [Dépannage](#dépannage)

---

## Prérequis

| Outil | Version Minimum | Vérification |
|-------|-----------------|--------------|
| Docker | 20.10+ | `docker --version` |
| Docker Compose | 2.0+ | `docker compose version` |
| RAM disponible | 4 GB | - |
| Espace disque | 5 GB | - |

---

## Installation de Docker

### macOS

```bash
# Option 1: Télécharger Docker Desktop
# https://www.docker.com/products/docker-desktop

# Option 2: Via Homebrew
brew install --cask docker

# Lancer Docker Desktop depuis Applications
# Attendre que l'icône Docker soit stable (pas d'animation)

# Vérifier l'installation
docker --version
docker compose version
```

### Linux (Ubuntu/Debian)

```bash
# Mettre à jour les packages
sudo apt update

# Installer les prérequis
sudo apt install -y ca-certificates curl gnupg lsb-release

# Ajouter la clé GPG Docker
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Ajouter le repository Docker
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Installer Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Ajouter l'utilisateur au groupe docker (évite sudo)
sudo usermod -aG docker $USER

# IMPORTANT: Se déconnecter/reconnecter ou exécuter
newgrp docker

# Vérifier
docker --version
docker compose version
```

### Linux (Fedora/RHEL)

```bash
# Installer le repository Docker
sudo dnf -y install dnf-plugins-core
sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo

# Installer Docker
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Démarrer Docker
sudo systemctl start docker
sudo systemctl enable docker

# Ajouter l'utilisateur au groupe docker
sudo usermod -aG docker $USER
newgrp docker

# Vérifier
docker --version
```

### Windows

```powershell
# Option 1: Télécharger Docker Desktop
# https://www.docker.com/products/docker-desktop

# Option 2: Via Chocolatey (PowerShell Admin)
choco install docker-desktop -y

# Option 3: Via winget
winget install Docker.DockerDesktop

# Après installation:
# 1. Redémarrer Windows si demandé
# 2. Lancer Docker Desktop
# 3. Accepter les conditions d'utilisation
# 4. Attendre que Docker soit prêt (icône stable)

# Vérifier (PowerShell ou CMD)
docker --version
docker compose version
```

### Vérification de l'Installation

```bash
# Test rapide
docker run hello-world

# Résultat attendu:
# Hello from Docker!
# This message shows that your installation appears to be working correctly.
```

---

## Démarrage Rapide

### Étape 1 : Cloner ou accéder au projet

```bash
cd /Users/andric/Documents/dev/claude/mlops-complet

# Vérifier la structure
ls -la
# Vous devez voir: docker-compose.yml, Dockerfile, src/, etc.
```

### Étape 2 : Lancer tous les services

```bash
# Construire et démarrer en arrière-plan
docker compose up -d --build

# Suivre les logs (optionnel)
docker compose logs -f
```

### Étape 3 : Vérifier que tout fonctionne

```bash
# Voir les conteneurs en cours
docker compose ps

# Résultat attendu:
# NAME                    STATUS          PORTS
# mlops-api               Up              0.0.0.0:5000->5000/tcp
# mlops-mlflow            Up              0.0.0.0:5001->5000/tcp
# mlops-prometheus        Up              0.0.0.0:9090->9090/tcp
# mlops-grafana           Up              0.0.0.0:3000->3000/tcp
```

### Étape 4 : Tester l'API

```bash
# Health check
curl http://localhost:5000/health

# Prédiction de démonstration
curl http://localhost:5000/predict/demo

# Ou ouvrir dans le navigateur:
# http://localhost:5000
```

### Étape 5 : Accéder aux interfaces web

| Service | URL | Credentials |
|---------|-----|-------------|
| API Flask | http://localhost:5000 | - |
| MLflow UI | http://localhost:5001 | - |
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | admin / admin |

---

## Architecture des Services

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DOCKER COMPOSE STACK                                 │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │   mlops-api     │    │  mlops-mlflow   │    │ mlops-prometheus│        │
│  │   (Flask)       │    │  (Tracking)     │    │  (Metrics)      │        │
│  │   Port: 5000    │    │   Port: 5001    │    │   Port: 9090    │        │
│  └────────┬────────┘    └─────────────────┘    └────────┬────────┘        │
│           │                                              │                  │
│           │         ┌─────────────────┐                 │                  │
│           └────────▶│  mlops-grafana  │◀────────────────┘                  │
│                     │  (Dashboards)   │                                    │
│                     │   Port: 3000    │                                    │
│                     └─────────────────┘                                    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Volumes Persistants                              │   │
│  │  mlflow_data    prometheus_data    grafana_data    models/          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Network: mlops-network                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Fichier docker-compose.yml Expliqué

```yaml
version: '3.8'

services:
  # ===========================================
  # API Flask - Service principal
  # ===========================================
  api:
    container_name: mlops-api
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "5000:5000"           # Exposé sur localhost:5000
    environment:
      - FLASK_ENV=production
      - LOG_LEVEL=INFO
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    volumes:
      - ./models:/app/models  # Modèle persistant
      - ./data:/app/data      # Données persistantes
    depends_on:
      - mlflow                # Attend que MLflow soit prêt
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - mlops-network

  # ===========================================
  # MLflow - Tracking des expériences
  # ===========================================
  mlflow:
    container_name: mlops-mlflow
    image: python:3.11-slim
    command: >
      bash -c "pip install mlflow &&
               mlflow server
               --host 0.0.0.0
               --port 5000
               --backend-store-uri sqlite:///mlflow.db
               --default-artifact-root /mlflow/artifacts"
    ports:
      - "5001:5000"           # Exposé sur localhost:5001
    volumes:
      - mlflow_data:/mlflow   # Données MLflow persistantes
    networks:
      - mlops-network

  # ===========================================
  # Prometheus - Collecte des métriques
  # ===========================================
  prometheus:
    container_name: mlops-prometheus
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    networks:
      - mlops-network

  # ===========================================
  # Grafana - Visualisation
  # ===========================================
  grafana:
    container_name: mlops-grafana
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
    depends_on:
      - prometheus
    networks:
      - mlops-network

# ===========================================
# Volumes persistants
# ===========================================
volumes:
  mlflow_data:
  prometheus_data:
  grafana_data:

# ===========================================
# Réseau partagé
# ===========================================
networks:
  mlops-network:
    driver: bridge
```

---

## Tester Chaque Service

### 1. API Flask (Port 5000)

```bash
# Health check
curl http://localhost:5000/health
# {"status": "healthy", "model_loaded": true}

# Page d'accueil
curl http://localhost:5000/
# Retourne la documentation de l'API

# Informations sur le modèle
curl http://localhost:5000/model/info

# Prédiction de démonstration
curl http://localhost:5000/predict/demo

# Prédiction avec données personnalisées
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [105.2, 104.8, 103.5, 100.2, 105.5, 105.0, 104.0,
                 55.0, 0.8, 0.5, 0.3, 0.05, 0.6, 0.01, 0.03,
                 0.05, 0.015, 0.018, 1.2, 0.02, 0.7, 0.002]
  }'

# Prédiction en lot (batch)
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {"features": [105.2, 104.8, 103.5, 100.2, 105.5, 105.0, 104.0, 55.0, 0.8, 0.5, 0.3, 0.05, 0.6, 0.01, 0.03, 0.05, 0.015, 0.018, 1.2, 0.02, 0.7, 0.002]},
      {"features": [102.1, 101.5, 100.8, 98.5, 102.3, 101.9, 101.2, 48.0, 0.6, 0.4, 0.2, 0.03, 0.5, 0.02, 0.04, 0.06, 0.012, 0.015, 1.1, 0.015, 0.65, 0.001]}
    ]
  }'

# Métriques Prometheus
curl http://localhost:5000/metrics
```

### 2. MLflow UI (Port 5001)

```bash
# Vérifier que MLflow répond
curl http://localhost:5001/health
# ou simplement
curl http://localhost:5001/

# Ouvrir dans le navigateur
open http://localhost:5001  # macOS
xdg-open http://localhost:5001  # Linux
start http://localhost:5001  # Windows
```

**Dans l'interface MLflow :**
- Voir les expériences dans le menu de gauche
- Comparer les runs
- Voir les métriques, paramètres et artifacts
- Gérer le Model Registry

### 3. Prometheus (Port 9090)

```bash
# Interface web
open http://localhost:9090

# API de requête
curl 'http://localhost:9090/api/v1/query?query=up'

# Vérifier les targets
curl http://localhost:9090/api/v1/targets
```

**Requêtes utiles dans Prometheus :**
```promql
# Requêtes de prédiction par minute
rate(prediction_requests_total[1m])

# Latence moyenne des prédictions
histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m]))

# Status des services
up
```

### 4. Grafana (Port 3000)

```bash
# Ouvrir l'interface
open http://localhost:3000

# Credentials par défaut:
# Username: admin
# Password: admin
```

**Configuration de Grafana :**

1. **Ajouter Prometheus comme source de données :**
   - Configuration > Data Sources > Add data source
   - Sélectionner "Prometheus"
   - URL: `http://prometheus:9090`
   - Save & Test

2. **Importer un dashboard :**
   - Create > Import
   - Coller un ID de dashboard (ex: 1860 pour Node Exporter)
   - Ou créer un dashboard personnalisé

---

## Commandes Utiles

### Gestion des Conteneurs

```bash
# Démarrer tous les services
docker compose up -d

# Démarrer avec reconstruction
docker compose up -d --build

# Arrêter tous les services
docker compose down

# Arrêter et supprimer les volumes
docker compose down -v

# Redémarrer un service spécifique
docker compose restart api

# Voir les logs de tous les services
docker compose logs -f

# Voir les logs d'un service spécifique
docker compose logs -f api
docker compose logs -f mlflow

# État des services
docker compose ps

# Statistiques des conteneurs (CPU, RAM)
docker stats
```

### Accéder aux Conteneurs

```bash
# Shell dans le conteneur API
docker compose exec api bash

# Shell dans le conteneur MLflow
docker compose exec mlflow bash

# Exécuter une commande
docker compose exec api python -c "print('Hello from container')"

# Voir les fichiers du modèle
docker compose exec api ls -la /app/models/
```

### Gestion des Images

```bash
# Lister les images du projet
docker images | grep mlops

# Reconstruire l'image API
docker compose build api

# Reconstruire sans cache
docker compose build --no-cache api

# Supprimer les images non utilisées
docker image prune -a
```

### Gestion des Volumes

```bash
# Lister les volumes
docker volume ls

# Inspecter un volume
docker volume inspect mlops-complet_mlflow_data

# Sauvegarder un volume
docker run --rm -v mlops-complet_mlflow_data:/data -v $(pwd):/backup \
  alpine tar cvf /backup/mlflow_backup.tar /data

# Restaurer un volume
docker run --rm -v mlops-complet_mlflow_data:/data -v $(pwd):/backup \
  alpine tar xvf /backup/mlflow_backup.tar -C /
```

### Nettoyage

```bash
# Arrêter et tout nettoyer
docker compose down -v --rmi all

# Nettoyage global Docker
docker system prune -a

# Nettoyage des volumes non utilisés
docker volume prune

# Espace utilisé par Docker
docker system df
```

---

## Configuration Avancée

### Variables d'Environnement

Créer un fichier `.env` à la racine du projet :

```bash
# .env
FLASK_ENV=development
LOG_LEVEL=DEBUG
MLFLOW_TRACKING_URI=http://mlflow:5000

# Grafana
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=votre_mot_de_passe_securise

# Prometheus
PROMETHEUS_RETENTION=15d
```

### Fichier docker-compose.override.yml

Pour le développement local avec hot-reload :

```yaml
# docker-compose.override.yml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.dev  # Dockerfile de développement
    volumes:
      - ./src:/app/src:ro          # Code source monté (lecture seule)
      - ./models:/app/models
    environment:
      - FLASK_ENV=development
      - FLASK_DEBUG=1
    command: flask run --host=0.0.0.0 --port=5000 --reload
```

### Dockerfile de Développement

```dockerfile
# Dockerfile.dev
FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY . .

# Variables d'environnement
ENV FLASK_APP=src/app.py
ENV FLASK_ENV=development

# Port
EXPOSE 5000

# Commande (sera override par docker-compose)
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000", "--reload"]
```

### Lancer en Mode Développement

```bash
# Avec le fichier override automatiquement détecté
docker compose up -d

# Ou explicitement
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

---

## Dépannage

### Problème : Docker ne démarre pas

**macOS/Windows :**
```bash
# Vérifier que Docker Desktop est lancé
# Regarder l'icône Docker dans la barre de menu/système

# Si bloqué, redémarrer Docker Desktop
# macOS: Quit Docker Desktop, puis relancer
# Windows: Clic droit sur l'icône > Restart
```

**Linux :**
```bash
# Vérifier le status du service
sudo systemctl status docker

# Démarrer si arrêté
sudo systemctl start docker

# Voir les logs
sudo journalctl -u docker.service
```

### Problème : Port déjà utilisé

```bash
# Erreur: "port is already allocated"

# Trouver le processus utilisant le port (exemple: 5000)
# Linux/macOS
lsof -i :5000
# ou
netstat -tulpn | grep 5000

# Windows
netstat -ano | findstr :5000

# Tuer le processus
kill -9 <PID>  # Linux/macOS
taskkill /PID <PID> /F  # Windows

# Ou changer le port dans docker-compose.yml
ports:
  - "5050:5000"  # Utiliser 5050 au lieu de 5000
```

### Problème : Conteneur ne démarre pas

```bash
# Voir les logs du conteneur
docker compose logs api

# Erreurs courantes:
# 1. "No module named..." → Dépendance manquante
# 2. "Model not found" → Le modèle n'est pas entraîné
# 3. "Permission denied" → Problème de droits sur les volumes
```

**Solution pour "Model not found" :**
```bash
# Entraîner le modèle dans le conteneur
docker compose exec api python src/train.py

# Ou monter un modèle existant
docker compose down
# Placer model.joblib dans ./models/
docker compose up -d
```

### Problème : Pas assez de mémoire

```bash
# Erreur: "Cannot allocate memory" ou conteneur qui crash

# Vérifier la mémoire disponible
docker stats

# Augmenter la mémoire dans Docker Desktop:
# Settings > Resources > Memory > Augmenter à 4GB+

# Ou limiter la mémoire par service dans docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 512M
```

### Problème : Build très lent

```bash
# Utiliser le cache Docker efficacement
docker compose build

# Si le cache pose problème, reconstruire sans cache
docker compose build --no-cache

# Optimiser le Dockerfile avec des layers bien ordonnés
# (dépendances avant code source)
```

### Problème : Volumes non persistants

```bash
# Vérifier que les volumes existent
docker volume ls

# Inspecter le volume
docker volume inspect mlops-complet_mlflow_data

# Si les données disparaissent, vérifier le mapping dans docker-compose.yml
volumes:
  - mlflow_data:/mlflow  # Volume nommé (persistant)
  # vs
  - ./data:/app/data     # Bind mount (dépend du host)
```

### Problème : Services ne communiquent pas

```bash
# Vérifier le réseau
docker network ls
docker network inspect mlops-complet_mlops-network

# Tester la connectivité depuis un conteneur
docker compose exec api ping mlflow
docker compose exec api curl http://mlflow:5000/health

# S'assurer que tous les services sont sur le même réseau
docker compose exec api cat /etc/hosts
```

### Reset Complet

```bash
# Arrêter tout
docker compose down

# Supprimer les volumes
docker compose down -v

# Supprimer les images
docker compose down --rmi all

# Nettoyer Docker
docker system prune -a --volumes

# Recommencer
docker compose up -d --build
```

---

## Scripts de Commodité

### Script de démarrage (start.sh)

```bash
#!/bin/bash
# start.sh - Démarrer le projet MLOps

set -e

echo "🚀 Démarrage du projet MLOps..."

# Vérifier Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker n'est pas installé"
    exit 1
fi

# Vérifier que Docker fonctionne
if ! docker info &> /dev/null; then
    echo "❌ Docker n'est pas en cours d'exécution"
    exit 1
fi

# Construire et démarrer
echo "📦 Construction des images..."
docker compose build

echo "🏃 Démarrage des services..."
docker compose up -d

# Attendre que l'API soit prête
echo "⏳ Attente de l'API..."
for i in {1..30}; do
    if curl -s http://localhost:5000/health > /dev/null; then
        echo "✅ API prête!"
        break
    fi
    sleep 1
done

# Afficher les URLs
echo ""
echo "🎉 Services disponibles:"
echo "   API Flask:  http://localhost:5000"
echo "   MLflow UI:  http://localhost:5001"
echo "   Prometheus: http://localhost:9090"
echo "   Grafana:    http://localhost:3000 (admin/admin)"
echo ""
echo "📋 Commandes utiles:"
echo "   docker compose logs -f     # Voir les logs"
echo "   docker compose down        # Arrêter"
echo "   docker compose ps          # Status"
```

### Script d'arrêt (stop.sh)

```bash
#!/bin/bash
# stop.sh - Arrêter le projet MLOps

echo "🛑 Arrêt des services..."
docker compose down

echo "✅ Services arrêtés"
```

### Rendre les scripts exécutables

```bash
chmod +x start.sh stop.sh

# Utilisation
./start.sh
./stop.sh
```

---

## Ressources

- [Documentation Docker](https://docs.docker.com/)
- [Documentation Docker Compose](https://docs.docker.com/compose/)
- [Best Practices Dockerfile](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [Docker Hub](https://hub.docker.com/)
