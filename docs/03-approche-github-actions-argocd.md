# Approche 1 : GitHub Actions + ArgoCD + Kubernetes

Cette approche représente une stack MLOps complète de niveau entreprise, combinant CI/CD avec GitHub Actions et déploiement GitOps avec ArgoCD.

---

## Vue d'Ensemble

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PIPELINE CI/CD COMPLET                             │
│                                                                             │
│  ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐       │
│  │  Code  │───▶│  Test  │───▶│ Train  │───▶│ Build  │───▶│  Push  │       │
│  │  Push  │    │  Unit  │    │ Model  │    │ Docker │    │Registry│       │
│  └────────┘    └────────┘    └────────┘    └────────┘    └────────┘       │
│                                                               │             │
│                                                               ▼             │
│  ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐       │
│  │  Prod  │◀───│Staging │◀───│ ArgoCD │◀───│ Update │◀───│Manifest│       │
│  │ Deploy │    │ Deploy │    │  Sync  │    │  Tag   │    │  Repo  │       │
│  └────────┘    └────────┘    └────────┘    └────────┘    └────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Concepts Clés

### GitHub Actions

GitHub Actions est une plateforme CI/CD intégrée à GitHub qui permet d'automatiser les workflows de build, test et déploiement.

**Composants principaux :**

| Composant | Description |
|-----------|-------------|
| **Workflow** | Fichier YAML définissant le pipeline (`.github/workflows/`) |
| **Job** | Ensemble d'étapes exécutées sur un runner |
| **Step** | Action individuelle (script ou action réutilisable) |
| **Runner** | Machine virtuelle exécutant les jobs |
| **Secret** | Variable sécurisée (tokens, mots de passe) |
| **Artifact** | Fichier produit/partagé entre jobs |

### ArgoCD

ArgoCD est un outil de déploiement continu GitOps pour Kubernetes. Il synchronise automatiquement l'état du cluster avec les manifests stockés dans Git.

**Principes GitOps :**

1. **Git comme source de vérité** - Tout est versionné dans Git
2. **Déclaratif** - L'état désiré est décrit, pas les étapes
3. **Automatisé** - Synchronisation automatique
4. **Auditable** - Historique complet des changements

---

## Architecture Détaillée

### 1. Structure du Workflow GitHub Actions

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:        # Tests unitaires et qualité
  train:       # Entraînement du modèle
  build:       # Construction image Docker
  update-manifests:  # Mise à jour des manifests K8s
  deploy-staging:    # Déploiement staging
  deploy-prod:       # Déploiement production (manuel)
```

### 2. Détail des Jobs

#### Job 1 : Test

```yaml
test:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4

    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
        cache: 'pip'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov flake8

    - name: Lint
      run: flake8 src/ --max-line-length=100

    - name: Test
      run: pytest tests/ -v --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

**Ce que fait ce job :**
- Configure Python 3.11 avec cache des dépendances
- Installe les dépendances du projet
- Exécute le linting avec flake8
- Lance les tests avec couverture de code
- Upload le rapport de couverture

#### Job 2 : Train

```yaml
train:
  needs: test
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4

    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
        cache: 'pip'

    - name: Install dependencies
      run: pip install -r requirements.txt

    - name: Train model
      run: python src/train.py

    - name: Upload model artifact
      uses: actions/upload-artifact@v4
      with:
        name: model
        path: models/
        retention-days: 30
```

**Ce que fait ce job :**
- S'exécute après le job `test` (`needs: test`)
- Entraîne le modèle avec les données
- Sauvegarde le modèle comme artifact GitHub

#### Job 3 : Build

```yaml
build:
  needs: train
  runs-on: ubuntu-latest
  outputs:
    image_tag: ${{ steps.meta.outputs.tags }}
  steps:
    - uses: actions/checkout@v4

    - name: Download model
      uses: actions/download-artifact@v4
      with:
        name: model
        path: models/

    - name: Docker meta
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ secrets.DOCKER_USERNAME }}/stock-prediction
        tags: |
          type=sha,prefix=
          type=ref,event=branch

    - name: Login to Docker Hub
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}

    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
```

**Ce que fait ce job :**
- Récupère le modèle entraîné depuis les artifacts
- Génère des tags Docker intelligents (SHA, branche)
- Construit l'image Docker multi-stage
- Push vers Docker Hub avec cache optimisé

#### Job 4 : Update Manifests

```yaml
update-manifests:
  needs: build
  runs-on: ubuntu-latest
  if: github.ref == 'refs/heads/main'
  steps:
    - uses: actions/checkout@v4

    - name: Update Kubernetes manifests
      run: |
        NEW_TAG="${{ needs.build.outputs.image_tag }}"
        sed -i "s|image:.*|image: $NEW_TAG|g" kubernetes/deployment.yaml

    - name: Commit and push
      run: |
        git config user.name "GitHub Actions"
        git config user.email "actions@github.com"
        git add kubernetes/deployment.yaml
        git commit -m "Update image to ${{ needs.build.outputs.image_tag }}"
        git push
```

**Ce que fait ce job :**
- Met à jour le tag de l'image dans les manifests K8s
- Commit et push les changements
- Déclenche automatiquement ArgoCD

---

## Configuration ArgoCD

### Installation d'ArgoCD

```bash
# Créer le namespace
kubectl create namespace argocd

# Installer ArgoCD
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Attendre que les pods soient prêts
kubectl wait --for=condition=Ready pods --all -n argocd --timeout=300s

# Récupérer le mot de passe admin initial
kubectl -n argocd get secret argocd-initial-admin-secret \
  -o jsonpath="{.data.password}" | base64 -d && echo

# Accéder à l'interface web
kubectl port-forward svc/argocd-server -n argocd 8080:443
# Ouvrir https://localhost:8080
```

### Application ArgoCD

```yaml
# kubernetes/argocd-application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: stock-prediction
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: default

  # Source des manifests
  source:
    repoURL: https://github.com/YOUR_USERNAME/mlops-stock-prediction.git
    targetRevision: HEAD
    path: kubernetes

  # Destination
  destination:
    server: https://kubernetes.default.svc
    namespace: stock-prediction

  # Politique de synchronisation
  syncPolicy:
    automated:
      prune: true        # Supprimer les ressources obsolètes
      selfHeal: true     # Corriger les drifts automatiquement
      allowEmpty: false
    syncOptions:
      - CreateNamespace=true
      - PrunePropagationPolicy=foreground
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
```

### Explication des Options

| Option | Description |
|--------|-------------|
| `prune: true` | Supprime les ressources K8s qui ne sont plus dans Git |
| `selfHeal: true` | Revient automatiquement à l'état Git si quelqu'un modifie manuellement |
| `CreateNamespace=true` | Crée le namespace s'il n'existe pas |
| `retry.limit: 5` | Réessaie 5 fois en cas d'échec |

---

## Manifests Kubernetes

### Namespace

```yaml
# kubernetes/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: stock-prediction
  labels:
    app: stock-prediction
    environment: production
```

### Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: stock-prediction-api
  namespace: stock-prediction
  labels:
    app: stock-prediction
    component: api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: stock-prediction
      component: api
  template:
    metadata:
      labels:
        app: stock-prediction
        component: api
    spec:
      serviceAccountName: stock-prediction-sa
      containers:
      - name: api
        image: username/stock-prediction:latest  # Mis à jour par CI
        ports:
        - containerPort: 5000
        env:
        - name: FLASK_ENV
          value: "production"
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: stock-prediction-config
              key: LOG_LEVEL
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Service et HPA

```yaml
# kubernetes/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: stock-prediction-service
  namespace: stock-prediction
spec:
  type: ClusterIP
  selector:
    app: stock-prediction
    component: api
  ports:
  - port: 80
    targetPort: 5000
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: stock-prediction-hpa
  namespace: stock-prediction
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: stock-prediction-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## Guide Pratique Pas à Pas

### Étape 1 : Préparer le Repository GitHub

```bash
# 1. Créer un nouveau repo sur GitHub
# 2. Cloner le projet
git clone https://github.com/YOUR_USERNAME/mlops-stock-prediction.git
cd mlops-stock-prediction

# 3. Copier les fichiers du projet
cp -r /chemin/vers/mlops-complet/* .

# 4. Configurer les secrets GitHub
# Settings > Secrets > Actions > New repository secret
# - DOCKER_USERNAME: votre username Docker Hub
# - DOCKER_PASSWORD: votre token Docker Hub
```

### Étape 2 : Démarrer Minikube

```bash
# Démarrer avec suffisamment de ressources
minikube start --cpus=4 --memory=4096

# Activer les addons nécessaires
minikube addons enable ingress
minikube addons enable metrics-server

# Vérifier
kubectl get nodes
```

### Étape 3 : Installer ArgoCD

```bash
# Installer
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Attendre
kubectl wait --for=condition=Ready pods --all -n argocd --timeout=300s

# Mot de passe
ARGO_PWD=$(kubectl -n argocd get secret argocd-initial-admin-secret \
  -o jsonpath="{.data.password}" | base64 -d)
echo "Password: $ARGO_PWD"

# Interface web (dans un autre terminal)
kubectl port-forward svc/argocd-server -n argocd 8080:443 &
```

### Étape 4 : Créer l'Application ArgoCD

```bash
# Via kubectl
kubectl apply -f kubernetes/argocd-application.yaml

# OU via l'interface web
# 1. Aller sur https://localhost:8080
# 2. Login: admin / $ARGO_PWD
# 3. New App > Remplir les champs
```

### Étape 5 : Déclencher le Pipeline

```bash
# Faire un changement et push
echo "# Update $(date)" >> README.md
git add .
git commit -m "Trigger pipeline"
git push

# Observer le pipeline sur GitHub Actions
# Observer la synchronisation sur ArgoCD
```

### Étape 6 : Vérifier le Déploiement

```bash
# Pods
kubectl get pods -n stock-prediction

# Logs
kubectl logs -f deployment/stock-prediction-api -n stock-prediction

# Port-forward pour tester
kubectl port-forward svc/stock-prediction-service 5000:80 -n stock-prediction &

# Test
curl http://localhost:5000/health
curl http://localhost:5000/predict/demo
```

---

## Flux de Travail Quotidien

### Développeur : Nouvelle Feature

```bash
# 1. Créer une branche
git checkout -b feature/nouvelle-feature

# 2. Développer et tester localement
python -m pytest tests/

# 3. Commit et push
git add .
git commit -m "Add nouvelle feature"
git push -u origin feature/nouvelle-feature

# 4. Créer une Pull Request sur GitHub
# Le pipeline CI s'exécute automatiquement

# 5. Après review et merge dans main
# Le pipeline complet se déclenche automatiquement
```

### Rollback en Production

```bash
# Option 1: Via ArgoCD UI
# History > Select previous version > Rollback

# Option 2: Via Git
git revert HEAD
git push

# Option 3: Via kubectl (temporaire, ArgoCD va re-sync)
kubectl rollout undo deployment/stock-prediction-api -n stock-prediction
```

---

## Monitoring et Observabilité

### Vérifier l'État ArgoCD

```bash
# Status de l'application
kubectl get applications -n argocd

# Détails
kubectl describe application stock-prediction -n argocd

# Logs ArgoCD
kubectl logs -f deployment/argocd-server -n argocd
```

### Dashboards

| Service | URL | Credentials |
|---------|-----|-------------|
| ArgoCD | https://localhost:8080 | admin / (secret) |
| API | http://localhost:5000 | - |
| GitHub Actions | github.com/repo/actions | GitHub login |

---

## Bonnes Pratiques

### 1. Sécurité

```yaml
# Ne jamais commiter de secrets
# Utiliser GitHub Secrets ou Sealed Secrets
- name: SENSITIVE_VAR
  valueFrom:
    secretKeyRef:
      name: my-secret
      key: sensitive-value
```

### 2. Gestion des Branches

```
main          → Production
├── develop   → Staging
└── feature/* → Développement
```

### 3. Stratégie de Tags

```yaml
# Utiliser des tags sémantiques
tags:
  - type=semver,pattern={{version}}
  - type=sha,prefix=
  - type=ref,event=branch
```

### 4. Tests avant Merge

```yaml
# Bloquer le merge si les tests échouent
on:
  pull_request:
    branches: [main]
# + Branch protection rules sur GitHub
```

---

## Dépannage

### Pipeline GitHub Actions échoue

```bash
# Vérifier les logs dans GitHub Actions UI
# Commun : secrets manquants, tests qui échouent

# Re-run le job
# Actions > Workflow > Re-run jobs
```

### ArgoCD ne synchronise pas

```bash
# Vérifier le status
kubectl get application stock-prediction -n argocd -o yaml

# Forcer la synchronisation
kubectl patch application stock-prediction -n argocd \
  --type merge -p '{"operation": {"sync": {}}}'

# Ou via CLI argocd
argocd app sync stock-prediction
```

### Pods en erreur

```bash
# Décrire le pod
kubectl describe pod <pod-name> -n stock-prediction

# Voir les logs
kubectl logs <pod-name> -n stock-prediction --previous

# Erreurs communes:
# - ImagePullBackOff: image non trouvée
# - CrashLoopBackOff: application crash
# - Pending: ressources insuffisantes
```

---

## Ressources

- [Documentation GitHub Actions](https://docs.github.com/en/actions)
- [Documentation ArgoCD](https://argo-cd.readthedocs.io/)
- [GitOps Principles](https://www.gitops.tech/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
