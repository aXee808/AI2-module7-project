# Partie 1 : Théorie MLOps

## Table des Matières

1. [Qu'est-ce que le MLOps ?](#1-quest-ce-que-le-mlops)
   - 1.1 [Definition](#11-définition)
   - 1.2 [Pourquoi le MLOps ?](#12-pourquoi-le-mlops-)
   - 1.3 [DevOps vs MLOps](#13-devops-vs-mlops)
   - 1.4 [GitOps vs DataOps vs MLOps](#14-gitops-vs-dataops-vs-mlops--les-trois-piliers-de-lautomatisation-moderne) ⭐ **NOUVEAU**
2. [Le Cycle de Vie ML](#2-le-cycle-de-vie-ml)
3. [Les Piliers du MLOps](#3-les-piliers-du-mlops)
4. [Architecture MLOps](#4-architecture-mlops)
5. [Comparaison des Outils](#5-comparaison-des-outils)

---

## 1. Qu'est-ce que le MLOps ?

### 1.1 Définition

**MLOps** (Machine Learning Operations) est un ensemble de pratiques qui combine le Machine Learning, le DevOps et l'ingénierie des données pour déployer et maintenir des systèmes ML en production de manière fiable et efficace.

```
┌─────────────────────────────────────────────────────────────────────┐
│                           MLOps                                      │
│                                                                       │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│   │   Machine   │     │   DevOps    │     │    Data     │          │
│   │   Learning  │  +  │             │  +  │ Engineering │          │
│   └─────────────┘     └─────────────┘     └─────────────┘          │
│         │                   │                   │                    │
│         └───────────────────┴───────────────────┘                    │
│                             │                                        │
│                             ▼                                        │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                     MLOps                                    │   │
│   │                                                              │   │
│   │  • Automatisation du pipeline ML                            │   │
│   │  • Versioning des données et modèles                        │   │
│   │  • CI/CD pour ML                                            │   │
│   │  • Monitoring en production                                 │   │
│   │  • Reproductibilité des expériences                         │   │
│   └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Pourquoi le MLOps ?

**Problèmes sans MLOps :**

| Problème | Conséquence |
|----------|-------------|
| "Ça marche sur mon PC" | Impossible à reproduire en production |
| Pas de versioning des données | On ne sait pas quelle donnée a entraîné quel modèle |
| Déploiement manuel | Lent, sujet aux erreurs, non scalable |
| Pas de monitoring | Dégradation du modèle non détectée |
| Code spaghetti | Maintenance impossible |

**Solutions avec MLOps :**

| Solution | Bénéfice |
|----------|----------|
| Infrastructure as Code | Environnement reproductible |
| Data versioning | Traçabilité complète |
| CI/CD automatisé | Déploiements rapides et fiables |
| Monitoring continu | Détection précoce des problèmes |
| Code structuré | Collaboration et maintenance facilitées |

### 1.3 DevOps vs MLOps

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DevOps vs MLOps                                   │
│                                                                       │
│  DevOps                              MLOps                           │
│  ───────                              ──────                          │
│  Code → Test → Deploy                Code → Data → Train → Test →   │
│                                      Deploy → Monitor                │
│                                                                       │
│  ┌──────────────────┐                ┌──────────────────┐           │
│  │                  │                │                  │           │
│  │   Code Source    │                │   Code Source    │           │
│  │        +         │                │        +         │           │
│  │   Config Files   │                │      Données     │           │
│  │                  │                │        +         │           │
│  └────────┬─────────┘                │     Modèles      │           │
│           │                          │        +         │           │
│           ▼                          │    Paramètres    │           │
│  ┌──────────────────┐                └────────┬─────────┘           │
│  │    Artefact      │                         │                      │
│  │   (Binary/Image) │                         ▼                      │
│  └──────────────────┘                ┌──────────────────┐           │
│                                      │    Artefacts     │           │
│                                      │  - Model file    │           │
│                                      │  - Scaler        │           │
│                                      │  - Metadata      │           │
│                                      │  - Métriques     │           │
│                                      └──────────────────┘           │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.4 GitOps vs DataOps vs MLOps : Les Trois Piliers de l'Automatisation Moderne

Ces trois disciplines partagent une philosophie commune : **automatiser et fiabiliser** les processus. Mais chacune cible un domaine different. Voici une explication complete pour bien les differencier.

#### Vue d'Ensemble : La Grande Image

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    LES TROIS "OPS" DE L'ENTREPRISE MODERNE                       │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                          PRODUIT FINAL                                   │   │
│   │                    (Application Intelligente)                            │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                    ▲                                             │
│                                    │                                             │
│         ┌──────────────────────────┼──────────────────────────┐                 │
│         │                          │                          │                 │
│         ▼                          ▼                          ▼                 │
│   ┌───────────┐            ┌───────────┐            ┌───────────┐              │
│   │  GitOps   │            │  DataOps  │            │  MLOps    │              │
│   │           │            │           │            │           │              │
│   │  Gere le  │            │  Gere les │            │ Gere les  │              │
│   │   CODE    │            │  DONNEES  │            │  MODELES  │              │
│   │           │            │           │            │    ML     │              │
│   └───────────┘            └───────────┘            └───────────┘              │
│         │                          │                          │                 │
│         ▼                          ▼                          ▼                 │
│   ┌───────────┐            ┌───────────┐            ┌───────────┐              │
│   │ Kubernetes│            │Data Lake/ │            │  MLflow/  │              │
│   │  ArgoCD   │            │ Warehouse │            │ Kubeflow  │              │
│   └───────────┘            └───────────┘            └───────────┘              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

#### 1.4.1 GitOps : L'Infrastructure Pilotee par Git

**Definition Simple :**
> GitOps = "Git est la source de verite pour TOUT ce qui concerne l'infrastructure"

**Analogie pour les eleves :**
> Imaginez que Git est comme un **plan d'architecte**. Tout ce qui est construit (serveurs, applications, configurations) doit correspondre exactement a ce plan. Si quelqu'un modifie le batiment sans changer le plan, un robot le remet automatiquement comme sur le plan.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              GITOPS EN ACTION                                    │
│                                                                                  │
│   DEVELOPPEUR                    GIT REPOSITORY                    CLUSTER K8S  │
│                                                                                  │
│   ┌─────────┐                    ┌─────────────┐                   ┌─────────┐  │
│   │  Push   │ ─────────────────► │  Manifests  │                   │ Etat    │  │
│   │ Config  │                    │  YAML/JSON  │                   │ Desire  │  │
│   └─────────┘                    │             │                   │         │  │
│                                  │ deployment: │                   │ 3 pods  │  │
│                                  │  replicas:3 │                   │ nginx   │  │
│                                  └──────┬──────┘                   └────▲────┘  │
│                                         │                               │       │
│                                         ▼                               │       │
│                                  ┌─────────────┐                        │       │
│                                  │   ArgoCD    │ ──────────────────────►│       │
│                                  │  (GitOps    │   Synchronisation      │       │
│                                  │  Operator)  │   Automatique          │       │
│                                  └─────────────┘                               │
│                                         │                                       │
│                                         ▼                                       │
│                                  ┌─────────────┐                               │
│                                  │  Detecte    │                               │
│                                  │  les diffs  │                               │
│                                  │  et corrige │                               │
│                                  └─────────────┘                               │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Principes Cles de GitOps :**

| Principe | Description | Exemple |
|----------|-------------|---------|
| **Declaratif** | On decrit l'etat souhaite, pas les etapes | "Je veux 3 replicas" vs "Lance 3 pods" |
| **Versionne** | Tout est dans Git avec historique | On peut revenir a n'importe quel etat |
| **Automatique** | Un agent synchronise en continu | ArgoCD verifie toutes les 3 minutes |
| **Auto-healing** | Correction automatique des derives | Si un pod est supprime, il est recree |

**Exemple Concret GitOps :**

```yaml
# Fichier dans Git : kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mon-api
spec:
  replicas: 3          # <- Je veux 3 instances
  selector:
    matchLabels:
      app: mon-api
  template:
    spec:
      containers:
      - name: api
        image: mon-api:v2.1.0   # <- Version precise
        resources:
          limits:
            memory: "256Mi"     # <- Limite de ressources
```

```
SCENARIO : Un admin supprime accidentellement un pod

  Avant GitOps                          Avec GitOps
  ────────────                          ────────────

  Admin: kubectl delete pod...          Admin: kubectl delete pod...
           │                                      │
           ▼                                      ▼
  [2 pods au lieu de 3]                 [2 pods au lieu de 3]
           │                                      │
           ▼                                      ▼
  Personne ne remarque                  ArgoCD detecte:
  pendant des heures                    "Etat reel != Etat Git"
           │                                      │
           ▼                                      ▼
  Incident en production!               ArgoCD recree le pod
                                        automatiquement en 30s
                                                  │
                                                  ▼
                                        [3 pods - Etat normal]
```

---

#### 1.4.2 DataOps : L'Excellence des Pipelines de Donnees

**Definition Simple :**
> DataOps = "Appliquer les pratiques DevOps au monde de la donnee"

**Analogie pour les eleves :**
> Imaginez une **usine de traitement d'eau**. L'eau brute (donnees brutes) arrive de differentes sources. Elle passe par des filtres (transformation), des controles qualite (validation), avant d'arriver propre et potable (donnees pretes) aux consommateurs. DataOps s'assure que cette usine fonctionne 24/7 sans erreur.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             DATAOPS PIPELINE                                     │
│                                                                                  │
│  SOURCES                TRANSFORMATION              DESTINATION                  │
│  ───────                ──────────────              ───────────                  │
│                                                                                  │
│  ┌─────────┐           ┌─────────────────────────────────────┐    ┌─────────┐  │
│  │   API   │──────────►│                                     │    │  Data   │  │
│  └─────────┘           │         ETL / ELT Pipeline          │───►│Warehouse│  │
│                        │                                     │    └─────────┘  │
│  ┌─────────┐           │  ┌───────┐  ┌───────┐  ┌───────┐  │                   │
│  │   CSV   │──────────►│  │Extract│─►│Transform│─►│ Load  │  │    ┌─────────┐  │
│  └─────────┘           │  └───────┘  └───────┘  └───────┘  │───►│Dashboard│  │
│                        │       │          │          │      │    └─────────┘  │
│  ┌─────────┐           │       ▼          ▼          ▼      │                   │
│  │Database │──────────►│  ┌─────────────────────────────┐   │    ┌─────────┐  │
│  └─────────┘           │  │      QUALITY CHECKS         │   │───►│   ML    │  │
│                        │  │  • Schema validation        │   │    │ Models  │  │
│  ┌─────────┐           │  │  • Null check               │   │    └─────────┘  │
│  │Streaming│──────────►│  │  • Anomaly detection        │   │                   │
│  └─────────┘           │  └─────────────────────────────┘   │                   │
│                        └─────────────────────────────────────┘                   │
│                                         │                                        │
│                                         ▼                                        │
│                               ┌─────────────────┐                               │
│                               │   MONITORING    │                               │
│                               │  • Data quality │                               │
│                               │  • Freshness    │                               │
│                               │  • Lineage      │                               │
│                               └─────────────────┘                               │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Les 7 Piliers de DataOps :**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         LES 7 PILIERS DATAOPS                                    │
│                                                                                  │
│    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│    │ 1. AGILITE  │  │ 2. VERSION  │  │ 3. QUALITE  │  │4. AUTOMATISE│          │
│    │             │  │   CONTROL   │  │             │  │             │          │
│    │ Iterations  │  │ Code + Data │  │ Tests auto  │  │ Pipelines   │          │
│    │ rapides     │  │ versioning  │  │ continus    │  │ CI/CD       │          │
│    └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                                                  │
│    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                           │
│    │5. MONITORING│  │ 6. SECURITE │  │7. GOUVERNANCE│                          │
│    │             │  │             │  │              │                          │
│    │ Observabilite│ │ Encryption  │  │ Lineage      │                          │
│    │ Alerting    │  │ Access ctrl │  │ Catalogage   │                          │
│    └─────────────┘  └─────────────┘  └─────────────┘                           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Exemple Concret DataOps avec Apache Airflow :**

```python
# Pipeline DataOps avec tests de qualite integres
from airflow import DAG
from airflow.operators.python import PythonOperator
from great_expectations import expect

def extract_data():
    """Extraire les donnees des sources"""
    return load_from_api("https://api.sales.com/daily")

def validate_data(data):
    """Controler la qualite des donnees"""
    # Test 1: Pas de valeurs nulles dans les colonnes critiques
    expect(data["customer_id"]).to_not_be_null()

    # Test 2: Les montants sont positifs
    expect(data["amount"]).to_be_greater_than(0)

    # Test 3: Les dates sont dans la plage attendue
    expect(data["date"]).to_be_between("2024-01-01", "2024-12-31")

def transform_data(data):
    """Transformer et enrichir les donnees"""
    data["region"] = data["zip_code"].apply(get_region)
    data["fiscal_quarter"] = data["date"].apply(get_quarter)
    return data

# Definition du DAG
dag = DAG("sales_pipeline", schedule_interval="@daily")

extract >> validate >> transform >> load
```

---

#### 1.4.3 MLOps : L'Operationnalisation du Machine Learning

**Definition Simple :**
> MLOps = "DevOps + DataOps appliques specifiquement au cycle de vie des modeles ML"

**Analogie pour les eleves :**
> Imaginez un **chef cuisinier etoile** (Data Scientist) qui cree une recette exceptionnelle (modele ML). MLOps, c'est tout ce qu'il faut pour transformer cette recette unique en **franchise de restaurants** : standardisation, controle qualite, formation du personnel, approvisionnement constant en ingredients frais (donnees), et satisfaction client (monitoring).

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              MLOPS : VUE COMPLETE                                │
│                                                                                  │
│   ┌────────────────────────────────────────────────────────────────────────┐    │
│   │                        EXPERIMENTATION                                  │    │
│   │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐        │    │
│   │  │  Donnees │───►│ Feature  │───►│ Training │───►│ Evaluate │        │    │
│   │  │          │    │   Eng.   │    │          │    │          │        │    │
│   │  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘        │    │
│   │       │                │               │               │              │    │
│   │       ▼                ▼               ▼               ▼              │    │
│   │  ┌─────────────────────────────────────────────────────────────┐     │    │
│   │  │              TRACKING (MLflow, W&B, Neptune)                │     │    │
│   │  │  • Parametres    • Metriques    • Artefacts    • Lineage   │     │    │
│   │  └─────────────────────────────────────────────────────────────┘     │    │
│   └────────────────────────────────────────────────────────────────────────┘    │
│                                      │                                           │
│                                      ▼                                           │
│   ┌────────────────────────────────────────────────────────────────────────┐    │
│   │                        MODEL REGISTRY                                   │    │
│   │  ┌──────────────────────────────────────────────────────────────┐     │    │
│   │  │  model_v1.0  │  model_v1.1  │  model_v2.0  │  model_v2.1   │     │    │
│   │  │  (Archived)  │  (Staging)   │  (Production)│  (Candidate)  │     │    │
│   │  └──────────────────────────────────────────────────────────────┘     │    │
│   └────────────────────────────────────────────────────────────────────────┘    │
│                                      │                                           │
│                                      ▼                                           │
│   ┌────────────────────────────────────────────────────────────────────────┐    │
│   │                        DEPLOYMENT & SERVING                             │    │
│   │                                                                          │    │
│   │     ┌─────────────┐        ┌─────────────┐        ┌─────────────┐      │    │
│   │     │   Shadow    │        │   Canary    │        │  Blue/Green │      │    │
│   │     │   Testing   │───────►│  (10% trafic)│───────►│   Switch    │      │    │
│   │     └─────────────┘        └─────────────┘        └─────────────┘      │    │
│   │                                                                          │    │
│   └────────────────────────────────────────────────────────────────────────┘    │
│                                      │                                           │
│                                      ▼                                           │
│   ┌────────────────────────────────────────────────────────────────────────┐    │
│   │                          MONITORING                                     │    │
│   │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐              │    │
│   │  │  Performance  │  │  Data Drift   │  │ Model Drift   │              │    │
│   │  │  (Latency,    │  │  (Input data  │  │ (Predictions  │              │    │
│   │  │   Throughput) │  │   changed)    │  │  degradent)   │              │    │
│   │  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘              │    │
│   │          │                  │                  │                       │    │
│   │          └──────────────────┴──────────────────┘                       │    │
│   │                             │                                           │    │
│   │                             ▼                                           │    │
│   │                    ┌─────────────────┐                                 │    │
│   │                    │   ALERTING &    │                                 │    │
│   │                    │  RE-TRAINING    │ ──────► Retour au Training      │    │
│   │                    └─────────────────┘                                 │    │
│   └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

#### 1.4.4 Tableau Comparatif Complet

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    GITOPS vs DATAOPS vs MLOPS                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Critere          │  GitOps           │  DataOps          │  MLOps             │
│  ─────────────────┼───────────────────┼───────────────────┼───────────────────│
│                   │                   │                   │                    │
│  FOCUS            │  Infrastructure   │  Pipelines de     │  Modeles ML        │
│                   │  & Applications   │  donnees          │                    │
│                   │                   │                   │                    │
│  SOURCE DE        │  Git Repository   │  Data Catalog     │  Model Registry    │
│  VERITE           │                   │  + Git            │  + Git             │
│                   │                   │                   │                    │
│  ARTEFACTS        │  Manifests YAML   │  Datasets         │  Modeles (.pkl,    │
│  PRINCIPAUX       │  Docker Images    │  Schemas          │  .h5, .onnx)       │
│                   │                   │  Pipelines        │  Features          │
│                   │                   │                   │                    │
│  VERSIONING       │  Git tags         │  Dataset versions │  Model versions    │
│                   │  Image tags       │  Schema versions  │  + Data versions   │
│                   │                   │                   │                    │
│  OUTILS           │  ArgoCD, Flux     │  Airflow, dbt     │  MLflow, Kubeflow  │
│  TYPIQUES         │  Terraform        │  Great Expect.    │  Vertex AI, SageMaker│
│                   │                   │  Fivetran         │                    │
│                   │                   │                   │                    │
│  TESTS            │  Config tests     │  Data quality     │  Model validation  │
│                   │  Integration      │  Schema tests     │  Performance tests │
│                   │                   │  Freshness        │  Bias/Fairness     │
│                   │                   │                   │                    │
│  MONITORING       │  App health       │  Data quality     │  Model drift       │
│                   │  Resource usage   │  Pipeline runs    │  Prediction quality│
│                   │  Sync status      │  Data freshness   │  Feature drift     │
│                   │                   │                   │                    │
│  ROLLBACK         │  Git revert       │  Restore dataset  │  Rollback model    │
│                   │  Image rollback   │  version          │  version           │
│                   │                   │                   │                    │
│  FREQUENCE        │  Plusieurs/jour   │  Horaire/Journa.  │  Hebdo/Mensuel     │
│  DE CHANGEMENT    │                   │                   │  (re-training)     │
│                   │                   │                   │                    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

#### 1.4.5 Comment Ils Travaillent Ensemble

**Illustration : Un Systeme de Recommandation E-commerce**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│              EXEMPLE COMPLET : SYSTEME DE RECOMMANDATION                         │
│                                                                                  │
│   JOUR 1 - DEVELOPPEMENT                                                        │
│   ──────────────────────                                                        │
│                                                                                  │
│   [DataOps]                                                                      │
│   ┌──────────────────────────────────────────────────────────────┐              │
│   │  1. Collecter les clics utilisateurs    (Extract)            │              │
│   │  2. Nettoyer et valider les donnees     (Quality Check)      │              │
│   │  3. Creer features: nb_vues, temps_page (Transform)          │              │
│   │  4. Stocker dans Feature Store          (Load)               │              │
│   └──────────────────────────────────────────────────────────────┘              │
│                                      │                                           │
│                                      ▼                                           │
│   [MLOps]                                                                        │
│   ┌──────────────────────────────────────────────────────────────┐              │
│   │  5. Entrainer modele de recommandation                       │              │
│   │  6. Tracker experience dans MLflow                           │              │
│   │  7. Enregistrer meilleur modele dans Registry                │              │
│   │  8. Tester performance (A/B test)                            │              │
│   └──────────────────────────────────────────────────────────────┘              │
│                                      │                                           │
│                                      ▼                                           │
│   [GitOps]                                                                       │
│   ┌──────────────────────────────────────────────────────────────┐              │
│   │  9. Push nouveau manifest avec image modele v2               │              │
│   │  10. ArgoCD detecte le changement                            │              │
│   │  11. Deploiement automatique sur Kubernetes                  │              │
│   │  12. Rollback automatique si health check echoue             │              │
│   └──────────────────────────────────────────────────────────────┘              │
│                                                                                  │
│   JOUR 2+ - PRODUCTION CONTINUE                                                 │
│   ─────────────────────────────                                                 │
│                                                                                  │
│   ┌────────────────┐    ┌────────────────┐    ┌────────────────┐              │
│   │    DataOps     │    │     MLOps      │    │     GitOps     │              │
│   │                │    │                │    │                │              │
│   │ Pipelines de   │───►│ Monitoring du  │───►│ Deploie        │              │
│   │ donnees fraiches│   │ modele en prod │    │ nouveau modele │              │
│   │ toutes les 6h  │    │ Alerte si drift│    │ si necessaire  │              │
│   └────────────────┘    └────────────────┘    └────────────────┘              │
│          │                      │                      │                       │
│          └──────────────────────┴──────────────────────┘                       │
│                                 │                                               │
│                                 ▼                                               │
│                    ┌─────────────────────────┐                                 │
│                    │   BOUCLE CONTINUE       │                                 │
│                    │   Data → Model → Deploy │                                 │
│                    │   (automatisee 24/7)    │                                 │
│                    └─────────────────────────┘                                 │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

#### 1.4.6 Resume

**L'Analogie de la Pizzeria :**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ANALOGIE : LA PIZZERIA AUTOMATISEE                           │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                          │   │
│   │   GITOPS = La CUISINE et l'EQUIPEMENT                                   │   │
│   │   ─────────────────────────────────────                                  │   │
│   │   "Le plan de la cuisine est dans un classeur. Si quelqu'un deplace    │   │
│   │   un four, un robot le remet automatiquement a sa place."              │   │
│   │                                                                          │   │
│   │   → Gere : Fours, refrigerateurs, plans de travail                      │   │
│   │   → Outil : ArgoCD (le robot qui remet tout en place)                   │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                          │   │
│   │   DATAOPS = Les INGREDIENTS                                             │   │
│   │   ─────────────────────────────                                          │   │
│   │   "Les tomates arrivent du fournisseur, sont lavees, coupees,          │   │
│   │   et stockees dans des bacs etiquetes avec controle qualite."          │   │
│   │                                                                          │   │
│   │   → Gere : Fraicheur, qualite, tracabilite des ingredients              │   │
│   │   → Outil : Airflow (l'ordonnanceur des livraisons)                     │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                          │   │
│   │   MLOPS = Les RECETTES                                                  │   │
│   │   ─────────────────────                                                  │   │
│   │   "La recette de la Margherita v2.3 est testee, approuvee,             │   │
│   │   et deployee dans toutes les pizzerias. Si les clients                │   │
│   │   n'aiment plus, on revient a la v2.2 ou on cree la v2.4."            │   │
│   │                                                                          │   │
│   │   → Gere : Creation, test, deploiement, suivi des recettes              │   │
│   │   → Outil : MLflow (le livre de recettes versionne)                     │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                          │   │
│   │   ENSEMBLE = LA PIZZERIA QUI TOURNE TOUTE SEULE                         │   │
│   │   ─────────────────────────────────────────────                          │   │
│   │   Les ingredients arrivent (DataOps) → La recette est appliquee (MLOps)│   │
│   │   → La cuisine est toujours prete (GitOps) → Clients satisfaits!        │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Questions pour verifier la comprehension :**

| Question | Reponse Attendue |
|----------|-----------------|
| "J'ai un bug dans mon pipeline de donnees. C'est quel Ops?" | DataOps |
| "Mon modele fait moins bonnes predictions qu'avant. C'est quel Ops?" | MLOps |
| "Mon serveur a ete supprime et je veux le recreer automatiquement. C'est quel Ops?" | GitOps |
| "Je veux versionner mes datasets. C'est quel Ops?" | DataOps |
| "Je veux deployer une nouvelle version de mon API. C'est quel Ops?" | GitOps |
| "Je veux comparer les performances de 3 modeles. C'est quel Ops?" | MLOps |

---

## 2. Le Cycle de Vie ML

### 2.1 Les Étapes

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CYCLE DE VIE ML                                   │
│                                                                       │
│   ┌─────────────┐                              ┌─────────────┐      │
│   │ 1. PROBLÈME │                              │ 7. MONITOR  │      │
│   │   BUSINESS  │                              │             │      │
│   └──────┬──────┘                              └──────▲──────┘      │
│          │                                            │              │
│          ▼                                            │              │
│   ┌─────────────┐                              ┌──────┴──────┐      │
│   │ 2. COLLECTE │                              │ 6. DEPLOY   │      │
│   │   DONNÉES   │                              │             │      │
│   └──────┬──────┘                              └──────▲──────┘      │
│          │                                            │              │
│          ▼                                            │              │
│   ┌─────────────┐                              ┌──────┴──────┐      │
│   │ 3. FEATURE  │──────────────────────────────│ 5. EVALUATE │      │
│   │ ENGINEERING │                              │             │      │
│   └──────┬──────┘                              └──────▲──────┘      │
│          │                                            │              │
│          ▼                          ┌────────────────┘              │
│   ┌─────────────┐                   │                               │
│   │ 4. TRAINING │───────────────────┘                               │
│   │             │                                                    │
│   └─────────────┘                                                    │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Détail de Chaque Étape

#### Étape 1 : Définition du Problème

```
Questions à se poser :
├── Quel problème business résout-on ?
├── Quelles métriques business suivre ?
├── Quelle est la baseline (solution actuelle) ?
├── Quel est le ROI attendu ?
└── Quelles contraintes (latence, coût, éthique) ?
```

**Notre cas : Prédiction de Stocks**
- **Problème** : Prédire si le prix d'une action va monter ou baisser
- **Métrique** : Accuracy, Precision/Recall pour décisions d'investissement
- **Contrainte** : Prédiction en temps réel (< 100ms)

#### Étape 2 : Collecte des Données

```
Sources de données :
├── APIs financières (Yahoo Finance, Alpha Vantage)
├── Données historiques
├── Données alternatives (news, social media)
└── Dans notre cas : Données synthétiques pour l'apprentissage
```

#### Étape 3 : Feature Engineering

```
Transformation des données brutes en features :
├── Indicateurs techniques (SMA, EMA, RSI, MACD)
├── Features statistiques (volatilité, momentum)
├── Features temporelles (jour de la semaine, mois)
└── Normalisation et encoding
```

#### Étape 4 : Entraînement

```
Pipeline d'entraînement :
├── Split train/validation/test
├── Sélection d'algorithmes
├── Hyperparameter tuning
├── Cross-validation
└── Tracking des expériences
```

#### Étape 5 : Évaluation

```
Métriques d'évaluation :
├── Classification : Accuracy, Precision, Recall, F1, AUC-ROC
├── Régression : MAE, MSE, RMSE, R²
├── Business : Profit/Loss simulé, Sharpe Ratio
└── Tests statistiques
```

#### Étape 6 : Déploiement

```
Stratégies de déploiement :
├── Shadow mode (en parallèle)
├── Canary (petit pourcentage)
├── Blue-Green (switch instantané)
└── A/B Testing
```

#### Étape 7 : Monitoring

```
Éléments à surveiller :
├── Data drift (distribution des données change)
├── Model drift (performance se dégrade)
├── Métriques système (latence, throughput)
└── Alerting et re-entraînement automatique
```

---

## 3. Les Piliers du MLOps

### 3.1 Les 4 Piliers

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LES 4 PILIERS DU MLOPS                           │
│                                                                       │
│   ┌─────────────────┐   ┌─────────────────┐                        │
│   │   VERSIONING    │   │  AUTOMATISATION │                        │
│   │                 │   │                 │                        │
│   │ • Code (Git)    │   │ • CI/CD         │                        │
│   │ • Données (DVC) │   │ • Pipelines     │                        │
│   │ • Modèles       │   │ • Tests auto    │                        │
│   │ • Expériences   │   │ • Déploiement   │                        │
│   └─────────────────┘   └─────────────────┘                        │
│                                                                       │
│   ┌─────────────────┐   ┌─────────────────┐                        │
│   │   MONITORING    │   │  COLLABORATION  │                        │
│   │                 │   │                 │                        │
│   │ • Performance   │   │ • Feature Store │                        │
│   │ • Data drift    │   │ • Model Registry│                        │
│   │ • Alerting      │   │ • Documentation │                        │
│   │ • Logging       │   │ • Standards     │                        │
│   └─────────────────┘   └─────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Niveaux de Maturité MLOps

**Google définit 3 niveaux de maturité MLOps :**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NIVEAUX DE MATURITÉ                              │
│                                                                       │
│  NIVEAU 0 : Manuel                                                   │
│  ───────────────────                                                 │
│  • Tout est fait à la main                                          │
│  • Pas de pipeline automatisé                                        │
│  • Notebooks en production (!)                                       │
│  • Déploiement manuel et rare                                        │
│                                                                       │
│  NIVEAU 1 : ML Pipeline Automation                                   │
│  ─────────────────────────────────                                   │
│  • Pipeline de training automatisé                                   │
│  • Tracking des expériences                                          │
│  • Déploiement semi-automatisé                                       │
│  • Tests basiques                                                    │
│                                                                       │
│  NIVEAU 2 : CI/CD for ML                                            │
│  ───────────────────────                                             │
│  • CI/CD complet pour code ET modèles                               │
│  • Tests automatisés (data, model, integration)                      │
│  • Déploiement automatique                                           │
│  • Monitoring et alerting                                            │
│  • Re-entraînement automatique                                       │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

**Notre tutoriel vise le Niveau 2.**

---

## 4. Architecture MLOps

### 4.1 Architecture de Référence

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ARCHITECTURE MLOPS                                    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         DATA LAYER                                      │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │ │
│  │  │ Data Sources │  │ Data Lake    │  │ Feature Store│                 │ │
│  │  │ (APIs, DBs)  │──│ (Raw Data)   │──│ (Features)   │                 │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                       TRAINING LAYER                                    │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │ │
│  │  │ Experiment   │  │ Training     │  │ Model        │                 │ │
│  │  │ Tracking     │──│ Pipeline     │──│ Registry     │                 │ │
│  │  │ (MLflow)     │  │ (Kubeflow)   │  │              │                 │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        CI/CD LAYER                                      │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │ │
│  │  │ Source Ctrl  │  │ CI Pipeline  │  │ CD Pipeline  │                 │ │
│  │  │ (GitHub)     │──│ (Actions)    │──│ (ArgoCD)     │                 │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                       SERVING LAYER                                     │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │ │
│  │  │ Kubernetes   │  │ Model Server │  │ API Gateway  │                 │ │
│  │  │ Cluster      │──│ (Flask/TF)   │──│ (Ingress)    │                 │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      MONITORING LAYER                                   │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │ │
│  │  │ Metrics      │  │ Logging      │  │ Alerting     │                 │ │
│  │  │ (Prometheus) │──│ (ELK)        │──│ (Grafana)    │                 │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Composants Clés

| Composant | Rôle | Outils |
|-----------|------|--------|
| **Version Control** | Versionner code, données, modèles | Git, DVC, MLflow |
| **Experiment Tracking** | Suivre les expériences ML | MLflow, W&B, Neptune |
| **Feature Store** | Stocker et servir les features | Feast, Tecton |
| **Model Registry** | Stocker et versionner les modèles | MLflow, Vertex AI |
| **Pipeline Orchestration** | Automatiser les workflows | Kubeflow, Airflow |
| **Model Serving** | Servir les prédictions | Flask, TFServing, Seldon |
| **Monitoring** | Surveiller performance et drift | Prometheus, Grafana, Evidently |

---

## 5. Comparaison des Outils

### 5.1 Nos 3 Approches

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPARAISON DES 3 APPROCHES                              │
│                                                                              │
│  APPROCHE 1: GitHub Actions + ArgoCD + Kubernetes                           │
│  ─────────────────────────────────────────────────                          │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐     │
│  │   GitHub   │───►│  Actions   │───►│   ArgoCD   │───►│ Kubernetes │     │
│  │   (Code)   │    │  (CI/CD)   │    │  (GitOps)  │    │  (Deploy)  │     │
│  └────────────┘    └────────────┘    └────────────┘    └────────────┘     │
│                                                                              │
│  ✅ Avantages : Standard industrie, GitOps, Scalable                        │
│  ❌ Inconvénients : Complexité initiale, Courbe d'apprentissage            │
│  🎯 Use case : Production enterprise, équipes DevOps existantes             │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  APPROCHE 2: Kubeflow                                                        │
│  ────────────────────                                                        │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐     │
│  │ Kubeflow   │───►│ Pipelines  │───►│  KFServing │───►│ Kubernetes │     │
│  │ Notebooks  │    │   (ML)     │    │  (Serving) │    │  (Infra)   │     │
│  └────────────┘    └────────────┘    └────────────┘    └────────────┘     │
│                                                                              │
│  ✅ Avantages : Conçu pour ML, Pipelines natifs, End-to-end                 │
│  ❌ Inconvénients : Complexe à installer, Resource-intensive                │
│  🎯 Use case : Grandes équipes ML, Workloads GPU, Expérimentation          │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  APPROCHE 3: MLflow                                                          │
│  ──────────────────                                                          │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐     │
│  │  MLflow    │───►│  MLflow    │───►│  MLflow    │───►│   Docker   │     │
│  │ Tracking   │    │  Projects  │    │  Registry  │    │  (Deploy)  │     │
│  └────────────┘    └────────────┘    └────────────┘    └────────────┘     │
│                                                                              │
│  ✅ Avantages : Simple, Léger, Rapide à démarrer, Open source              │
│  ❌ Inconvénients : Moins de features que Kubeflow, Scaling manuel         │
│  🎯 Use case : Petites équipes, Prototypes, Démarrage rapide               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Tableau Comparatif Détaillé

| Critère | Approche 1 (GH+ArgoCD) | Approche 2 (Kubeflow) | Approche 3 (MLflow) |
|---------|------------------------|----------------------|---------------------|
| **Difficulté** | Moyenne | Élevée | Faible |
| **Temps de setup** | 2-4h | 4-8h | 30min-1h |
| **Ressources requises** | 4GB RAM, 2 CPU | 8GB RAM, 4 CPU | 2GB RAM, 1 CPU |
| **Courbe d'apprentissage** | Moyenne | Longue | Courte |
| **Experiment Tracking** | MLflow (externe) | Intégré | Natif |
| **Pipeline ML** | Manuel/GitHub | Natif (DSL) | Projects |
| **Model Registry** | MLflow/Custom | Intégré | Natif |
| **Serving** | Custom (Flask/K8s) | KFServing | mlflow serve |
| **GitOps** | ArgoCD (natif) | Possible | Non |
| **Multi-cloud** | Oui | Oui | Limité |
| **GPU Support** | Via K8s | Natif | Via Docker |

### 5.3 Quand Utiliser Quelle Approche ?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ARBRE DE DÉCISION                                         │
│                                                                              │
│   Démarrer                                                                   │
│      │                                                                       │
│      ▼                                                                       │
│   ┌─────────────────────────────────────┐                                   │
│   │ Vous débutez en MLOps ?             │                                   │
│   └──────────────┬──────────────────────┘                                   │
│          OUI     │     NON                                                   │
│           │      │      │                                                    │
│           ▼      │      ▼                                                    │
│   ┌──────────┐   │   ┌─────────────────────────────────────┐               │
│   │ MLflow   │   │   │ Vous avez une équipe DevOps/K8s ?   │               │
│   │(Approche │   │   └──────────────┬──────────────────────┘               │
│   │   3)     │   │          OUI     │     NON                               │
│   └──────────┘   │           │      │      │                                │
│                  │           ▼      │      ▼                                │
│                  │   ┌───────────┐  │  ┌────────────────────────────────┐  │
│                  │   │GitHub     │  │  │ Workloads ML complexes (GPU) ? │  │
│                  │   │Actions +  │  │  └──────────────┬─────────────────┘  │
│                  │   │ArgoCD     │  │          OUI    │    NON              │
│                  │   │(Approche  │  │           │     │     │               │
│                  │   │   1)      │  │           ▼     │     ▼               │
│                  │   └───────────┘  │   ┌──────────┐  │  ┌──────────┐      │
│                  │                  │   │ Kubeflow │  │  │ MLflow   │      │
│                  │                  │   │(Approche │  │  │(Approche │      │
│                  │                  │   │   2)     │  │  │   3)     │      │
│                  │                  │   └──────────┘  │  └──────────┘      │
│                  │                  │                 │                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Résumé du Chapitre

### Ce que vous avez appris :

1. **MLOps** combine ML, DevOps et Data Engineering
2. Le **cycle de vie ML** comprend 7 étapes clés
3. Les **4 piliers** sont : Versioning, Automatisation, Monitoring, Collaboration
4. Il existe **3 niveaux de maturité** MLOps (0, 1, 2)
5. Chaque approche a ses **avantages et use cases**

### Prochaines étapes :

Dans le prochain chapitre, nous allons mettre en place le projet de base avec :
- Structure du code Python avec Flask
- Génération de données synthétiques
- Feature Engineering
- Entraînement du modèle
- API de prédiction

---

**Navigation :**
- [Suivant : Code Source Python →](./02-code-source.md)
- [Retour au README](../README.md)
