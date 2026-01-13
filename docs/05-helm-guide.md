# Partie 5 : Guide Complet Helm

## Table des Matieres

1. [Qu'est-ce que Helm ?](#1-quest-ce-que-helm-)
2. [Pourquoi utiliser Helm ?](#2-pourquoi-utiliser-helm-)
3. [Architecture et Concepts](#3-architecture-et-concepts)
4. [Installation de Helm](#4-installation-de-helm)
5. [Premiers pas avec Helm](#5-premiers-pas-avec-helm)
6. [Creer son propre Chart](#6-creer-son-propre-chart)
7. [Travaux Pratiques](#7-travaux-pratiques)
8. [Bonnes Pratiques](#8-bonnes-pratiques)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Qu'est-ce que Helm ?

### 1.1 Definition

**Helm** est le gestionnaire de paquets pour Kubernetes. Il permet de definir, installer et mettre a jour des applications Kubernetes complexes.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           HELM = Le "apt/brew" de Kubernetes                     │
│                                                                                  │
│   ┌────────────────┐                              ┌────────────────────────┐    │
│   │                │                              │                        │    │
│   │   Linux/macOS  │                              │      Kubernetes        │    │
│   │                │                              │                        │    │
│   │  apt install   │                              │    helm install        │    │
│   │  brew install  │          ≈                   │                        │    │
│   │  yum install   │                              │    helm upgrade        │    │
│   │                │                              │    helm rollback       │    │
│   └────────────────┘                              └────────────────────────┘    │
│                                                                                  │
│   Package = .deb, .rpm, .pkg                      Package = Chart (.tgz)        │
│   Repository = apt repo, homebrew tap             Repository = Helm repo        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Analogie Simple

> Imaginez que vous voulez installer une application complexe comme WordPress sur Kubernetes. Sans Helm, vous devez creer manuellement :
> - Un Deployment pour WordPress
> - Un Deployment pour MySQL
> - Des Services pour chacun
> - Des ConfigMaps et Secrets
> - Des PersistentVolumeClaims
> - Potentiellement un Ingress
>
> **Avec Helm :** `helm install my-wordpress bitnami/wordpress`
>
> Une seule commande, tout est configure automatiquement !

---

## 2. Pourquoi utiliser Helm ?

### 2.1 Les Problemes sans Helm

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PROBLEMES SANS HELM                                       │
│                                                                                  │
│   1. DUPLICATION DE CODE                                                         │
│   ──────────────────────                                                         │
│   dev/deployment.yaml     ┐                                                      │
│   staging/deployment.yaml ├─── 90% identiques, 10% de differences               │
│   prod/deployment.yaml    ┘                                                      │
│                                                                                  │
│   2. GESTION DES VERSIONS                                                        │
│   ───────────────────────                                                        │
│   Comment savoir quelle version est deployee ?                                   │
│   Comment faire un rollback facilement ?                                        │
│                                                                                  │
│   3. CONFIGURATION COMPLEXE                                                      │
│   ─────────────────────────                                                      │
│   Modifier 15 fichiers YAML pour changer une valeur ?                           │
│                                                                                  │
│   4. PARTAGE ET REUTILISATION                                                   │
│   ────────────────────────────                                                   │
│   Comment partager une config avec l'equipe ?                                   │
│   Comment reutiliser dans un autre projet ?                                     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Les Solutions avec Helm

| Probleme | Solution Helm |
|----------|---------------|
| Duplication de code | **Templates** avec variables |
| Gestion des versions | **Releases** versionnees avec historique |
| Configuration complexe | **values.yaml** centralise |
| Partage | **Charts** empaquetes et repositories |
| Rollback | `helm rollback` instantane |
| Dependencies | **Sub-charts** et requirements |

---

## 3. Architecture et Concepts

### 3.1 Les 3 Concepts Fondamentaux

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        LES 3 CONCEPTS HELM                                       │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                            1. CHART                                      │   │
│   │                                                                          │   │
│   │   = Un package Helm contenant tous les fichiers necessaires             │   │
│   │     pour deployer une application                                        │   │
│   │                                                                          │   │
│   │   my-chart/                                                              │   │
│   │   ├── Chart.yaml        # Metadata du chart                             │   │
│   │   ├── values.yaml       # Valeurs par defaut                            │   │
│   │   ├── templates/        # Templates Kubernetes                          │   │
│   │   │   ├── deployment.yaml                                               │   │
│   │   │   ├── service.yaml                                                  │   │
│   │   │   └── ...                                                           │   │
│   │   └── charts/           # Dependencies (sub-charts)                     │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                           2. RELEASE                                     │   │
│   │                                                                          │   │
│   │   = Une instance d'un chart deployee dans un cluster                    │   │
│   │                                                                          │   │
│   │   Exemple: Le chart "wordpress" peut etre installe plusieurs fois       │   │
│   │            avec des noms differents (releases):                          │   │
│   │                                                                          │   │
│   │   helm install blog-prod bitnami/wordpress -n production                │   │
│   │   helm install blog-staging bitnami/wordpress -n staging                │   │
│   │                                                                          │   │
│   │   → 2 releases du meme chart                                            │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                          3. REPOSITORY                                   │   │
│   │                                                                          │   │
│   │   = Un serveur HTTP qui heberge des charts                              │   │
│   │                                                                          │   │
│   │   Repositories populaires :                                              │   │
│   │   • bitnami   : https://charts.bitnami.com/bitnami                      │   │
│   │   • stable    : https://charts.helm.sh/stable (deprecie)                │   │
│   │   • jetstack  : https://charts.jetstack.io (cert-manager)               │   │
│   │   • prometheus: https://prometheus-community.github.io/helm-charts      │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Structure d'un Chart

```
my-chart/
├── Chart.yaml          # [OBLIGATOIRE] Metadata du chart (nom, version, description)
├── Chart.lock          # [GENERE] Lock file pour les dependencies
├── values.yaml         # [OBLIGATOIRE] Valeurs par defaut
├── values.schema.json  # [OPTIONNEL] Schema JSON pour valider values.yaml
├── .helmignore         # [OPTIONNEL] Fichiers a ignorer lors du packaging
├── templates/          # [OBLIGATOIRE] Dossier contenant les templates
│   ├── NOTES.txt       # Notes affichees apres installation
│   ├── _helpers.tpl    # Fonctions helper reutilisables
│   ├── deployment.yaml # Template de Deployment
│   ├── service.yaml    # Template de Service
│   ├── ingress.yaml    # Template d'Ingress
│   ├── configmap.yaml  # Template de ConfigMap
│   ├── secret.yaml     # Template de Secret
│   ├── hpa.yaml        # Template de HPA
│   └── tests/          # Tests du chart
│       └── test-connection.yaml
├── charts/             # [OPTIONNEL] Sub-charts (dependencies)
└── crds/               # [OPTIONNEL] Custom Resource Definitions
```

### 3.3 Le Moteur de Templates

Helm utilise le langage de templates **Go** pour generer les manifests Kubernetes.

```yaml
# Template (deployment.yaml)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}-{{ .Chart.Name }}
  labels:
    app: {{ .Values.app.name }}
    version: {{ .Chart.AppVersion }}
spec:
  replicas: {{ .Values.replicaCount }}
  template:
    spec:
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          {{- if .Values.resources }}
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          {{- end }}
```

```yaml
# values.yaml
app:
  name: my-app
replicaCount: 3
image:
  repository: nginx
  tag: "1.21"
resources:
  requests:
    memory: "64Mi"
    cpu: "250m"
```

```yaml
# Resultat genere (apres helm template)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-release-my-chart
  labels:
    app: my-app
    version: 1.0.0
spec:
  replicas: 3
  template:
    spec:
      containers:
        - name: my-chart
          image: "nginx:1.21"
          resources:
            requests:
              memory: "64Mi"
              cpu: "250m"
```

---

## 4. Installation de Helm

### 4.1 Pre-requis

- **kubectl** installe et configure
- Acces a un cluster Kubernetes (local ou distant)
- macOS, Linux, ou Windows

### 4.2 Installation sur macOS

```bash
# Methode 1 : Homebrew (recommandee)
brew install helm

# Methode 2 : Script officiel
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

### 4.3 Installation sur Linux

```bash
# Methode 1 : Script officiel (recommandee)
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Methode 2 : Snap
sudo snap install helm --classic

# Methode 3 : APT (Debian/Ubuntu)
curl https://baltocdn.com/helm/signing.asc | gpg --dearmor | sudo tee /usr/share/keyrings/helm.gpg > /dev/null
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/helm.gpg] https://baltocdn.com/helm/stable/debian/ all main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
sudo apt-get update
sudo apt-get install helm
```

### 4.4 Installation sur Windows

```powershell
# Methode 1 : Chocolatey
choco install kubernetes-helm

# Methode 2 : Scoop
scoop install helm

# Methode 3 : Winget
winget install Helm.Helm
```

### 4.5 Verification de l'Installation

```bash
# Verifier la version
helm version

# Sortie attendue :
# version.BuildInfo{Version:"v3.14.0", GitCommit:"...", GitTreeState:"clean", GoVersion:"go1.21.5"}

# Verifier l'acces au cluster
helm list
# (devrait afficher une liste vide si aucun chart n'est installe)
```

---

## 5. Premiers pas avec Helm

### 5.1 Ajouter un Repository

```bash
# Ajouter le repository Bitnami (le plus populaire)
helm repo add bitnami https://charts.bitnami.com/bitnami

# Mettre a jour la liste des charts
helm repo update

# Lister les repos configures
helm repo list

# Rechercher un chart
helm search repo nginx
helm search repo wordpress
```

### 5.2 Installer un Chart

```bash
# Installation basique
helm install my-nginx bitnami/nginx

# Installation dans un namespace specifique
helm install my-nginx bitnami/nginx -n web --create-namespace

# Installation avec des valeurs personnalisees
helm install my-nginx bitnami/nginx --set service.type=NodePort

# Installation avec un fichier de valeurs
helm install my-nginx bitnami/nginx -f my-values.yaml

# Installation en mode dry-run (voir ce qui serait cree)
helm install my-nginx bitnami/nginx --dry-run --debug
```

### 5.3 Gerer les Releases

```bash
# Lister les releases installees
helm list
helm list -A  # Tous les namespaces

# Voir le statut d'une release
helm status my-nginx

# Voir l'historique des versions
helm history my-nginx

# Mettre a jour une release
helm upgrade my-nginx bitnami/nginx --set replicaCount=3

# Rollback a une version precedente
helm rollback my-nginx 1

# Desinstaller une release
helm uninstall my-nginx
```

### 5.4 Inspecter un Chart

```bash
# Voir les valeurs par defaut d'un chart
helm show values bitnami/nginx

# Voir toutes les infos d'un chart
helm show all bitnami/nginx

# Voir le README du chart
helm show readme bitnami/nginx

# Telecharger un chart localement pour l'inspecter
helm pull bitnami/nginx --untar
```

---

## 6. Creer son propre Chart

### 6.1 Scaffolding

```bash
# Creer un nouveau chart
helm create my-app

# Structure generee :
my-app/
├── Chart.yaml
├── values.yaml
├── charts/
├── templates/
│   ├── NOTES.txt
│   ├── _helpers.tpl
│   ├── deployment.yaml
│   ├── hpa.yaml
│   ├── ingress.yaml
│   ├── service.yaml
│   ├── serviceaccount.yaml
│   └── tests/
│       └── test-connection.yaml
└── .helmignore
```

### 6.2 Syntaxe des Templates

#### Variables de base

```yaml
# Variables disponibles dans les templates :

# Release info
{{ .Release.Name }}        # Nom de la release
{{ .Release.Namespace }}   # Namespace
{{ .Release.Revision }}    # Numero de revision
{{ .Release.IsInstall }}   # true si c'est une installation
{{ .Release.IsUpgrade }}   # true si c'est une mise a jour

# Chart info
{{ .Chart.Name }}          # Nom du chart
{{ .Chart.Version }}       # Version du chart
{{ .Chart.AppVersion }}    # Version de l'application

# Values
{{ .Values.replicaCount }} # Valeur definie dans values.yaml

# Capabilities (infos sur le cluster)
{{ .Capabilities.KubeVersion }}  # Version de Kubernetes
```

#### Fonctions courantes

```yaml
# Valeur par defaut
{{ .Values.name | default "my-app" }}

# Conversion en YAML
{{ toYaml .Values.resources | nindent 8 }}

# Encodage base64
{{ .Values.password | b64enc }}

# Trim et quote
{{ .Values.name | trim | quote }}

# Conditions
{{- if .Values.ingress.enabled }}
  # ... configuration ingress
{{- end }}

# Boucles
{{- range .Values.hosts }}
  - {{ . | quote }}
{{- end }}

# Include d'un template helper
{{ include "my-app.fullname" . }}
```

#### Exemple complet de template

```yaml
# templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "my-app.fullname" . }}
  labels:
    {{- include "my-app.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "my-app.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      annotations:
        checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
      labels:
        {{- include "my-app.selectorLabels" . | nindent 8 }}
    spec:
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: {{ .Values.service.port }}
          {{- with .Values.resources }}
          resources:
            {{- toYaml . | nindent 12 }}
          {{- end }}
```

### 6.3 Valider un Chart

```bash
# Linter le chart
helm lint ./my-app

# Generer les manifests sans installer (debug)
helm template my-release ./my-app

# Generer avec des valeurs specifiques
helm template my-release ./my-app -f values-prod.yaml

# Dry-run complet avec le serveur
helm install my-release ./my-app --dry-run --debug
```

### 6.4 Packager et Distribuer

```bash
# Creer un package .tgz
helm package ./my-app
# Resultat: my-app-1.0.0.tgz

# Creer un index pour un repository
helm repo index . --url https://my-charts.example.com

# Heberger sur GitHub Pages, S3, GCS, etc.
```

---

## 7. Travaux Pratiques

### TP 1 : Installation de Base (15 min)

**Objectif** : Installer et manipuler un chart existant

```bash
# 1. Demarrer un cluster local (si pas deja fait)
# Option A: Docker Desktop avec Kubernetes active
# Option B: Minikube
minikube start

# 2. Ajouter le repository Bitnami
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# 3. Rechercher le chart nginx
helm search repo bitnami/nginx

# 4. Voir les valeurs disponibles
helm show values bitnami/nginx | head -50

# 5. Installer nginx avec des valeurs personnalisees
helm install my-nginx bitnami/nginx \
  --set service.type=NodePort \
  --set service.nodePorts.http=30080

# 6. Verifier l'installation
helm list
kubectl get all -l app.kubernetes.io/instance=my-nginx

# 7. Acceder a nginx
# Minikube:
minikube service my-nginx --url
# Docker Desktop:
# Acceder a http://localhost:30080

# 8. Mettre a jour avec plus de replicas
helm upgrade my-nginx bitnami/nginx \
  --set service.type=NodePort \
  --set service.nodePorts.http=30080 \
  --set replicaCount=3

# 9. Verifier la mise a jour
kubectl get pods -l app.kubernetes.io/instance=my-nginx

# 10. Voir l'historique
helm history my-nginx

# 11. Rollback a la version 1
helm rollback my-nginx 1

# 12. Nettoyage
helm uninstall my-nginx
```

---

### TP 2 : Deployer notre API Stock Prediction (30 min)

**Objectif** : Deployer l'application MLOps avec notre chart Helm personnalise

```bash
# 1. Se positionner dans le projet
cd /chemin/vers/mlops-complet

# 2. Verifier la structure du chart
ls -la helm/stock-prediction/
ls -la helm/stock-prediction/templates/

# 3. Linter le chart pour verifier qu'il est valide
helm lint helm/stock-prediction/

# Sortie attendue:
# ==> Linting helm/stock-prediction/
# [INFO] Chart.yaml: icon is recommended
# 1 chart(s) linted, 0 chart(s) failed

# 4. Generer les templates pour verifier (dry-run)
helm template stock-api helm/stock-prediction/ \
  --namespace mlops \
  --debug

# 5. Creer le namespace
kubectl create namespace mlops

# 6. Installer en mode developpement
helm install stock-api helm/stock-prediction/ \
  -n mlops \
  -f helm/stock-prediction/values-dev.yaml

# 7. Verifier l'installation
helm list -n mlops
kubectl get all -n mlops

# 8. Voir les notes post-installation
helm status stock-api -n mlops

# 9. Acceder a l'application
kubectl port-forward -n mlops svc/stock-api-stock-prediction 8080:80

# Dans un autre terminal:
curl http://localhost:8080/health

# 10. Tester une prediction (si l'API est fonctionnelle)
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.5, 2.3, 0.8, 1.2, 0.5]}'

# 11. Mettre a jour vers la config production
helm upgrade stock-api helm/stock-prediction/ \
  -n mlops \
  -f helm/stock-prediction/values-prod.yaml \
  --set image.repository=nginx \
  --set image.tag=latest \
  --set ingress.enabled=false

# 12. Voir les changements
helm diff upgrade stock-api helm/stock-prediction/ \
  -n mlops \
  -f helm/stock-prediction/values-prod.yaml 2>/dev/null || echo "Installer helm-diff: helm plugin install https://github.com/databus23/helm-diff"

# 13. Verifier le HPA
kubectl get hpa -n mlops

# 14. Nettoyage
helm uninstall stock-api -n mlops
kubectl delete namespace mlops
```

---

### TP 3 : Creer un Chart from Scratch (45 min)

**Objectif** : Creer un chart Helm complet pour une application simple

```bash
# 1. Creer un nouveau chart
helm create simple-api
cd simple-api

# 2. Nettoyer les templates par defaut
rm -rf templates/tests
rm templates/hpa.yaml templates/ingress.yaml templates/serviceaccount.yaml

# 3. Modifier Chart.yaml
cat > Chart.yaml << 'EOF'
apiVersion: v2
name: simple-api
description: Une API simple pour apprendre Helm
version: 0.1.0
appVersion: "1.0.0"
type: application
EOF

# 4. Creer values.yaml simplifie
cat > values.yaml << 'EOF'
replicaCount: 1

image:
  repository: hashicorp/http-echo
  tag: latest
  pullPolicy: IfNotPresent

message: "Hello from Helm!"

service:
  type: ClusterIP
  port: 80
  targetPort: 5678

resources:
  requests:
    memory: "32Mi"
    cpu: "10m"
  limits:
    memory: "64Mi"
    cpu: "50m"
EOF

# 5. Creer le template de deployment
cat > templates/deployment.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "simple-api.fullname" . }}
  labels:
    {{- include "simple-api.labels" . | nindent 4 }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      {{- include "simple-api.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "simple-api.selectorLabels" . | nindent 8 }}
    spec:
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          args:
            - "-text={{ .Values.message }}"
          ports:
            - name: http
              containerPort: {{ .Values.service.targetPort }}
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
EOF

# 6. Creer le template de service
cat > templates/service.yaml << 'EOF'
apiVersion: v1
kind: Service
metadata:
  name: {{ include "simple-api.fullname" . }}
  labels:
    {{- include "simple-api.labels" . | nindent 4 }}
spec:
  type: {{ .Values.service.type }}
  ports:
    - port: {{ .Values.service.port }}
      targetPort: {{ .Values.service.targetPort }}
      protocol: TCP
      name: http
  selector:
    {{- include "simple-api.selectorLabels" . | nindent 4 }}
EOF

# 7. Creer les NOTES.txt
cat > templates/NOTES.txt << 'EOF'
Bravo ! Votre API simple a ete deployee.

Message configure: {{ .Values.message }}

Pour y acceder:
  kubectl port-forward svc/{{ include "simple-api.fullname" . }} 8080:{{ .Values.service.port }}
  curl http://localhost:8080
EOF

# 8. Revenir au dossier parent et linter
cd ..
helm lint simple-api/

# 9. Tester le template
helm template test-release simple-api/

# 10. Installer
kubectl create namespace test-helm
helm install my-api simple-api/ -n test-helm

# 11. Tester
kubectl port-forward -n test-helm svc/my-api-simple-api 8080:80 &
sleep 2
curl http://localhost:8080
# Sortie: Hello from Helm!

# 12. Modifier le message
helm upgrade my-api simple-api/ -n test-helm --set message="Helm is awesome!"

# 13. Retester
curl http://localhost:8080
# Sortie: Helm is awesome!

# 14. Packager le chart
helm package simple-api/
# Resultat: simple-api-0.1.0.tgz

# 15. Nettoyage
pkill -f "port-forward.*8080"
helm uninstall my-api -n test-helm
kubectl delete namespace test-helm
rm -rf simple-api simple-api-0.1.0.tgz
```

---

### TP 4 : Helm avec ArgoCD (GitOps) (30 min)

**Objectif** : Deployer un chart Helm via ArgoCD

```bash
# 1. S'assurer qu'ArgoCD est installe
kubectl get pods -n argocd

# Si pas installe:
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# 2. Attendre que les pods soient prets
kubectl wait --for=condition=Ready pods --all -n argocd --timeout=300s

# 3. Obtenir le mot de passe admin
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d

# 4. Port-forward vers l'UI ArgoCD
kubectl port-forward svc/argocd-server -n argocd 8443:443 &

# 5. Creer une Application ArgoCD pour notre chart Helm
cat > argocd-helm-app.yaml << 'EOF'
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: stock-prediction-helm
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/YOUR_USERNAME/mlops-complet.git
    targetRevision: HEAD
    path: helm/stock-prediction
    helm:
      valueFiles:
        - values-dev.yaml
  destination:
    server: https://kubernetes.default.svc
    namespace: mlops
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
EOF

# 6. Appliquer l'Application ArgoCD
kubectl apply -f argocd-helm-app.yaml

# 7. Verifier dans l'UI ArgoCD
# Ouvrir https://localhost:8443
# Login: admin / <mot de passe recupere>

# 8. Observer la synchronisation
kubectl get application -n argocd stock-prediction-helm -w

# 9. Nettoyage
kubectl delete -f argocd-helm-app.yaml
pkill -f "port-forward.*8443"
```

---

## 8. Bonnes Pratiques

### 8.1 Nommage et Versioning

```yaml
# Chart.yaml
apiVersion: v2
name: my-app                 # Nom en minuscules, tirets autorises
version: 1.2.3               # SemVer pour le chart
appVersion: "2.0.0"          # Version de l'application
```

### 8.2 Values par Defaut

```yaml
# values.yaml - Toujours fournir des valeurs par defaut fonctionnelles

# BON: Valeurs par defaut sensees
replicaCount: 1
image:
  repository: nginx
  tag: "stable"
  pullPolicy: IfNotPresent

# MAUVAIS: Valeurs vides qui causent des erreurs
# image:
#   repository: ""
#   tag: ""
```

### 8.3 Templates Robustes

```yaml
# Toujours utiliser des conditions pour les valeurs optionnelles
{{- if .Values.ingress.enabled }}
apiVersion: networking.k8s.io/v1
kind: Ingress
# ...
{{- end }}

# Utiliser with pour les blocs optionnels
{{- with .Values.nodeSelector }}
nodeSelector:
  {{- toYaml . | nindent 8 }}
{{- end }}

# Fournir des valeurs par defaut
image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
```

### 8.4 Securite

```yaml
# Ne jamais stocker de secrets en clair dans values.yaml
# Utiliser des references a des secrets externes

# BON: Reference a un secret existant
envFrom:
  - secretRef:
      name: {{ .Values.existingSecret }}

# MAUVAIS: Secret en clair
# password: "my-secret-password"
```

### 8.5 Documentation

```yaml
# values.yaml - Documenter chaque valeur

# -- Nombre de replicas du deployment
# @default -- 1
replicaCount: 1

# -- Configuration de l'image Docker
image:
  # -- Repository de l'image
  repository: nginx
  # -- Tag de l'image (defaut: appVersion du chart)
  tag: ""
  # -- Policy de pull de l'image
  pullPolicy: IfNotPresent
```

---

## 9. Troubleshooting

### 9.1 Commandes de Debug

```bash
# Voir les logs de la release
helm status <release-name>

# Debug complet d'un template
helm template <release> <chart> --debug

# Voir les manifests d'une release installee
helm get manifest <release-name>

# Voir les valeurs utilisees
helm get values <release-name>

# Voir tout (hooks, manifests, notes, values)
helm get all <release-name>
```

### 9.2 Erreurs Courantes

| Erreur | Cause | Solution |
|--------|-------|----------|
| `Error: INSTALLATION FAILED: cannot re-use a name that is still in use` | Release existe deja | `helm uninstall <name>` ou changer le nom |
| `Error: YAML parse error` | Syntaxe YAML invalide | Verifier l'indentation |
| `Error: template: ... undefined` | Variable non definie | Verifier values.yaml ou utiliser `default` |
| `Error: unable to build kubernetes objects` | Manifest invalide | `helm template --debug` pour voir le manifest |
| `Error: timed out waiting for the condition` | Pods ne demarrent pas | `kubectl describe pod <pod>` |

### 9.3 Reset Complet

```bash
# Supprimer une release coincee
helm uninstall <release-name> --no-hooks

# Supprimer les secrets Helm orphelins
kubectl delete secret -l owner=helm,status=deployed

# Forcer la suppression d'un namespace bloque
kubectl delete namespace <ns> --force --grace-period=0
```

---

## Resume

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          HELM EN UN COUP D'OEIL                                  │
│                                                                                  │
│   INSTALLER                          GERER                                       │
│   ─────────                          ─────                                       │
│   helm install <release> <chart>     helm list                                  │
│   helm install -f values.yaml        helm status <release>                      │
│   helm install --set key=value       helm history <release>                     │
│                                                                                  │
│   METTRE A JOUR                      ROLLBACK                                   │
│   ────────────                       ────────                                   │
│   helm upgrade <release> <chart>     helm rollback <release> <revision>         │
│   helm upgrade --install             helm rollback <release> 0 (precedent)      │
│                                                                                  │
│   DEBUGGER                           SUPPRIMER                                  │
│   ────────                           ─────────                                  │
│   helm template <release> <chart>    helm uninstall <release>                   │
│   helm lint <chart>                  helm uninstall --keep-history              │
│   helm get manifest <release>                                                    │
│                                                                                  │
│   REPOSITORIES                       CREER                                      │
│   ────────────                       ─────                                      │
│   helm repo add <name> <url>         helm create <name>                         │
│   helm repo update                   helm package <chart>                       │
│   helm search repo <keyword>         helm push <chart> <repo>                   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

**Navigation :**
- [Precedent : Architecture MLOps](./01-theorie-mlops.md)
- [Retour au README](../README.md)
