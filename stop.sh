#!/bin/bash
# ===========================================
# stop.sh - Arrêter le projet MLOps
# ===========================================

echo "========================================"
echo "   MLOps Stock Prediction - Arrêt"
echo "========================================"
echo ""

# Couleurs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Afficher les conteneurs en cours
echo "Conteneurs en cours d'exécution:"
docker compose ps --format "table {{.Name}}\t{{.Status}}"
echo ""

# Demander confirmation si en mode interactif
if [ -t 0 ]; then
    read -p "Voulez-vous arrêter tous les services? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Annulé."
        exit 0
    fi
fi

# Arrêter les services
echo -e "${YELLOW}Arrêt des services...${NC}"
docker compose down

echo ""
echo -e "${GREEN}Services arrêtés avec succès!${NC}"
echo ""
echo "Pour supprimer aussi les volumes (données):"
echo "   docker compose down -v"
echo ""
echo "Pour redémarrer:"
echo "   ./start.sh"
