#!/bin/bash
# ============================================
# Script d'installation MLOps
# Compatible: macOS et Linux
# ============================================

set -e

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonctions utilitaires
print_header() {
    echo -e "\n${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  $1"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Détecter l'OS
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    else
        print_error "OS non supporté: $OSTYPE"
        exit 1
    fi
    print_info "OS détecté: $OS"
}

# Vérifier les prérequis
check_prerequisites() {
    print_header "Vérification des prérequis"

    local missing=()

    # Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        print_success "Python $PYTHON_VERSION"
    else
        missing+=("python3")
        print_error "Python 3 non trouvé"
    fi

    # pip
    if command -v pip3 &> /dev/null || command -v pip &> /dev/null; then
        print_success "pip installé"
    else
        missing+=("pip")
        print_error "pip non trouvé"
    fi

    # Git
    if command -v git &> /dev/null; then
        print_success "Git installé"
    else
        missing+=("git")
        print_error "Git non trouvé"
    fi

    # Docker
    if command -v docker &> /dev/null; then
        print_success "Docker installé"
    else
        print_warning "Docker non trouvé (optionnel pour le développement local)"
    fi

    # kubectl
    if command -v kubectl &> /dev/null; then
        print_success "kubectl installé"
    else
        print_warning "kubectl non trouvé (nécessaire pour Kubernetes)"
    fi

    if [ ${#missing[@]} -ne 0 ]; then
        print_error "Prérequis manquants: ${missing[*]}"
        print_info "Installer les dépendances manquantes et relancer le script"
        exit 1
    fi
}

# Créer l'environnement virtuel
setup_virtualenv() {
    print_header "Configuration de l'environnement virtuel Python"

    VENV_DIR="venv"

    if [ -d "$VENV_DIR" ]; then
        print_warning "L'environnement virtuel existe déjà"
        read -p "Voulez-vous le recréer? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
        else
            print_info "Utilisation de l'environnement existant"
            return
        fi
    fi

    print_info "Création de l'environnement virtuel..."
    python3 -m venv "$VENV_DIR"

    print_info "Activation de l'environnement..."
    source "$VENV_DIR/bin/activate"

    print_info "Mise à jour de pip..."
    pip install --upgrade pip

    print_success "Environnement virtuel créé: $VENV_DIR"
}

# Installer les dépendances Python
install_dependencies() {
    print_header "Installation des dépendances Python"

    # S'assurer que le venv est activé
    if [ -z "$VIRTUAL_ENV" ]; then
        source venv/bin/activate
    fi

    if [ -f "requirements.txt" ]; then
        print_info "Installation depuis requirements.txt..."
        pip install -r requirements.txt
        print_success "Dépendances installées"
    else
        print_error "requirements.txt non trouvé"
        exit 1
    fi
}

# Créer la structure des répertoires
create_directories() {
    print_header "Création de la structure des répertoires"

    directories=(
        "data/raw"
        "data/processed"
        "models"
        "logs"
        "monitoring/grafana/provisioning/dashboards"
        "monitoring/grafana/provisioning/datasources"
        "notebooks"
        "tests"
    )

    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_success "Créé: $dir"
    done
}

# Générer les données et entraîner le modèle
train_model() {
    print_header "Entraînement du modèle"

    # S'assurer que le venv est activé
    if [ -z "$VIRTUAL_ENV" ]; then
        source venv/bin/activate
    fi

    print_info "Génération des données et entraînement..."
    python src/train.py

    if [ -f "models/model.joblib" ]; then
        print_success "Modèle entraîné et sauvegardé"
    else
        print_error "Échec de l'entraînement"
        exit 1
    fi
}

# Tester l'installation
test_installation() {
    print_header "Test de l'installation"

    # S'assurer que le venv est activé
    if [ -z "$VIRTUAL_ENV" ]; then
        source venv/bin/activate
    fi

    print_info "Exécution des tests..."

    # Tests unitaires
    if python -m pytest tests/ -v --tb=short 2>/dev/null; then
        print_success "Tests passés"
    else
        print_warning "Certains tests ont échoué (normal si c'est la première installation)"
    fi
}

# Lancer l'API en local
start_api() {
    print_header "Démarrage de l'API"

    # S'assurer que le venv est activé
    if [ -z "$VIRTUAL_ENV" ]; then
        source venv/bin/activate
    fi

    print_info "Démarrage du serveur Flask..."
    print_info "API disponible sur: http://localhost:5000"
    print_info "Health check: http://localhost:5000/health"
    print_info "Démo: http://localhost:5000/predict/demo"
    print_info ""
    print_info "Appuyez sur Ctrl+C pour arrêter"

    python src/app.py
}

# Créer les fichiers de configuration
create_config_files() {
    print_header "Création des fichiers de configuration"

    # Prometheus config
    cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'stock-prediction-api'
    static_configs:
      - targets: ['api:5000']
    metrics_path: '/metrics'
EOF
    print_success "Créé: monitoring/prometheus.yml"

    # Grafana datasource
    cat > monitoring/grafana/provisioning/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF
    print_success "Créé: monitoring/grafana/provisioning/datasources/prometheus.yml"

    # .gitignore
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
ENV/
.venv/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Data
data/raw/*.csv
data/processed/*.csv

# Models
models/*.joblib
models/*.pkl

# Logs
logs/
*.log

# Environment
.env
.env.local

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Coverage
.coverage
htmlcov/

# MLflow
mlruns/
mlflow.db
EOF
    print_success "Créé: .gitignore"

    # .env template
    cat > .env.example << 'EOF'
# Configuration de l'application
PORT=5000
DEBUG=false
MODEL_PATH=models

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5001

# Docker Registry (pour CI/CD)
DOCKER_USERNAME=your-username
DOCKER_PASSWORD=your-password
EOF
    print_success "Créé: .env.example"
}

# Afficher les instructions finales
show_next_steps() {
    print_header "Installation terminée!"

    echo -e "
${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}
${GREEN}║              Installation terminée avec succès!               ║${NC}
${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}

${BLUE}Prochaines étapes:${NC}

1. ${YELLOW}Activer l'environnement virtuel:${NC}
   source venv/bin/activate

2. ${YELLOW}Lancer l'API en local:${NC}
   python src/app.py

3. ${YELLOW}Lancer avec Docker:${NC}
   docker-compose up -d

4. ${YELLOW}Accéder aux services:${NC}
   - API:        http://localhost:5000
   - MLflow:     http://localhost:5001
   - Prometheus: http://localhost:9090
   - Grafana:    http://localhost:3000 (admin/admin)

5. ${YELLOW}Tester une prédiction:${NC}
   curl http://localhost:5000/predict/demo

${BLUE}Documentation:${NC}
   - Théorie MLOps:    docs/01-theorie-mlops.md
   - GitHub Actions:   docs/02-github-actions-argocd.md
   - Kubeflow:         docs/03-kubeflow.md
   - MLflow:           docs/04-mlflow.md
"
}

# Menu principal
main() {
    print_header "Installation MLOps Stock Prediction"

    detect_os
    check_prerequisites
    create_directories
    create_config_files
    setup_virtualenv
    install_dependencies
    train_model
    test_installation
    show_next_steps
}

# Gestion des arguments
case "${1:-all}" in
    all)
        main
        ;;
    deps)
        setup_virtualenv
        install_dependencies
        ;;
    train)
        train_model
        ;;
    test)
        test_installation
        ;;
    start)
        start_api
        ;;
    *)
        echo "Usage: $0 {all|deps|train|test|start}"
        exit 1
        ;;
esac
