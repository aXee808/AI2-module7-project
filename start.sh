#!/bin/bash
# ===========================================
# start.sh - Démarrer le projet MLOps
# ===========================================

set -e

echo "========================================"
echo "   MLOps Stock Prediction - Démarrage"
echo "========================================"
echo ""

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fonction pour afficher les messages
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ===========================================
# Vérification des prérequis
# ===========================================
info "Vérification des prérequis..."

# Vérifier Docker
if ! command -v docker &> /dev/null; then
    error "Docker n'est pas installé. Veuillez l'installer: https://docs.docker.com/get-docker/"
    exit 1
fi

# Vérifier que Docker fonctionne
if ! docker info &> /dev/null; then
    error "Docker n'est pas en cours d'exécution. Veuillez démarrer Docker Desktop."
    exit 1
fi

# Vérifier Docker Compose
if ! docker compose version &> /dev/null; then
    error "Docker Compose n'est pas installé ou n'est pas accessible."
    exit 1
fi

info "Docker $(docker --version | cut -d' ' -f3 | tr -d ',')"
info "Docker Compose $(docker compose version --short)"

# ===========================================
# Construction et démarrage
# ===========================================
echo ""
info "Construction des images Docker..."
docker compose build --quiet

echo ""
info "Démarrage des services..."
docker compose up -d

# ===========================================
# Attente des services
# ===========================================
echo ""
info "Attente du démarrage des services..."

# Attendre l'API
echo -n "   API Flask: "
for i in {1..30}; do
    if curl -s http://localhost:5000/health > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${YELLOW}En attente...${NC}"
    fi
    sleep 1
done

# Attendre MLflow
echo -n "   MLflow:    "
for i in {1..30}; do
    if curl -s http://localhost:5001 > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${YELLOW}En attente...${NC}"
    fi
    sleep 1
done

# Attendre Prometheus
echo -n "   Prometheus:"
for i in {1..15}; do
    if curl -s http://localhost:9090/-/ready > /dev/null 2>&1; then
        echo -e " ${GREEN}OK${NC}"
        break
    fi
    if [ $i -eq 15 ]; then
        echo -e " ${YELLOW}En attente...${NC}"
    fi
    sleep 1
done

# Attendre Grafana
echo -n "   Grafana:   "
for i in {1..15}; do
    if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
        break
    fi
    if [ $i -eq 15 ]; then
        echo -e " ${YELLOW}En attente...${NC}"
    fi
    sleep 1
done

# ===========================================
# Afficher les URLs
# ===========================================
echo ""
echo "========================================"
echo "   Services disponibles"
echo "========================================"
echo ""
echo "   API Flask:   http://localhost:5000"
echo "   MLflow UI:   http://localhost:5001"
echo "   Prometheus:  http://localhost:9090"
echo "   Grafana:     http://localhost:3000"
echo "                (admin / admin)"
echo ""
echo "========================================"
echo "   Commandes utiles"
echo "========================================"
echo ""
echo "   Voir les logs:     docker compose logs -f"
echo "   Voir les logs API: docker compose logs -f api"
echo "   Status:            docker compose ps"
echo "   Arrêter:           ./stop.sh"
echo "                      ou: docker compose down"
echo ""
echo "========================================"
echo "   Tests rapides"
echo "========================================"
echo ""
echo "   curl http://localhost:5000/health"
echo "   curl http://localhost:5000/predict/demo"
echo ""

# Test rapide
info "Test de l'API..."
response=$(curl -s http://localhost:5000/health 2>/dev/null || echo "error")
if [[ "$response" == *"healthy"* ]]; then
    echo -e "   ${GREEN}API fonctionnelle!${NC}"
else
    warn "L'API n'a pas encore répondu. Vérifiez les logs: docker compose logs api"
fi

echo ""
info "Projet MLOps démarré avec succès!"
