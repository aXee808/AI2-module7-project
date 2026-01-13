"""
Flask API for Stock Prediction
==============================
API REST pour servir les prédictions du modèle ML.

Endpoints:
    GET  /              - Info de l'API
    GET  /health        - Health check
    GET  /model/info    - Informations sur le modèle
    POST /predict       - Prédiction à partir de features
    GET  /metrics       - Métriques Prometheus
"""

import os
import json
import time
import logging
from datetime import datetime
from functools import wraps
from typing import Dict, Any

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Création de l'application Flask
app = Flask(__name__)
CORS(app)  # Permettre les requêtes cross-origin

# Configuration
app.config['MODEL_PATH'] = os.environ.get('MODEL_PATH', 'models')
app.config['DEBUG'] = os.environ.get('DEBUG', 'false').lower() == 'true'

# Variables globales
model = None
model_info = {}
request_count = 0
prediction_count = 0
error_count = 0
start_time = datetime.now()


def load_model():
    """Charge le modèle au démarrage."""
    global model, model_info

    model_path = app.config['MODEL_PATH']

    try:
        from train import StockPredictor
        model = StockPredictor.load(model_path)
        model_info = {
            'status': 'loaded',
            'type': model.model_type,
            'features': len(model.feature_names),
            'path': model_path,
            'loaded_at': datetime.now().isoformat()
        }
        logger.info(f"✅ Modèle chargé depuis {model_path}")
    except Exception as e:
        model = None
        model_info = {
            'status': 'not_loaded',
            'error': str(e),
            'path': model_path
        }
        logger.warning(f"⚠️ Impossible de charger le modèle: {e}")


def track_request(f):
    """Décorateur pour tracker les requêtes."""
    @wraps(f)
    def decorated(*args, **kwargs):
        global request_count
        request_count += 1
        start = time.time()

        try:
            response = f(*args, **kwargs)
            return response
        finally:
            duration = time.time() - start
            logger.info(f"{request.method} {request.path} - {duration:.3f}s")

    return decorated


def require_model(f):
    """Décorateur pour vérifier que le modèle est chargé."""
    @wraps(f)
    def decorated(*args, **kwargs):
        global error_count
        if model is None:
            error_count += 1
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Le modèle n\'est pas disponible. Veuillez réessayer plus tard.'
            }), 503
        return f(*args, **kwargs)
    return decorated


# ========================================
# ROUTES
# ========================================

@app.route('/')
@track_request
def index():
    """Page d'accueil avec info de l'API."""
    return jsonify({
        'name': 'Stock Prediction API',
        'version': '1.0.0',
        'description': 'API de prédiction de direction de stocks',
        'endpoints': {
            'GET /': 'Cette page',
            'GET /health': 'Health check',
            'GET /model/info': 'Informations sur le modèle',
            'POST /predict': 'Prédiction avec features',
            'GET /predict/demo': 'Exemple de prédiction',
            'GET /metrics': 'Métriques Prometheus'
        },
        'documentation': '/docs' if app.config['DEBUG'] else 'Disabled in production'
    })


@app.route('/health')
@track_request
def health():
    """
    Health check endpoint.

    Utilisé par Kubernetes pour les probes liveness/readiness.
    """
    uptime = (datetime.now() - start_time).total_seconds()

    health_status = {
        'status': 'healthy' if model is not None else 'degraded',
        'timestamp': datetime.now().isoformat(),
        'uptime_seconds': uptime,
        'model_loaded': model is not None,
        'checks': {
            'model': 'ok' if model is not None else 'not_loaded'
        }
    }

    status_code = 200 if model is not None else 503
    return jsonify(health_status), status_code


@app.route('/model/info')
@track_request
def get_model_info():
    """Retourne les informations sur le modèle chargé."""
    info = {
        **model_info,
        'feature_names': model.feature_names if model else []
    }
    return jsonify(info)


@app.route('/predict', methods=['POST'])
@track_request
@require_model
def predict():
    """
    Endpoint de prédiction.

    Expects JSON body with features:
    {
        "SMA_5": 105.2,
        "SMA_10": 104.8,
        ...
    }

    Returns:
    {
        "prediction": 1,
        "probability": 0.72,
        "direction": "UP",
        "confidence": 0.72
    }
    """
    global prediction_count, error_count

    try:
        # Récupérer les données
        data = request.get_json()

        if not data:
            error_count += 1
            return jsonify({
                'error': 'No data provided',
                'message': 'Veuillez fournir les features en JSON'
            }), 400

        # Vérifier les features requises
        missing_features = [f for f in model.feature_names if f not in data]
        if missing_features:
            error_count += 1
            return jsonify({
                'error': 'Missing features',
                'missing': missing_features,
                'required': model.feature_names
            }), 400

        # Faire la prédiction
        result = model.predict_single(data)
        prediction_count += 1

        # Ajouter des métadonnées
        result['timestamp'] = datetime.now().isoformat()
        result['model_type'] = model.model_type

        return jsonify(result)

    except Exception as e:
        error_count += 1
        logger.error(f"Erreur de prédiction: {e}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route('/predict/demo')
@track_request
@require_model
def predict_demo():
    """
    Démo de prédiction avec des données d'exemple.

    Utile pour tester que l'API fonctionne.
    """
    global prediction_count

    # Données d'exemple réalistes
    demo_features = {
        'SMA_5': 105.2,
        'SMA_10': 104.8,
        'SMA_20': 103.5,
        'SMA_50': 100.2,
        'EMA_5': 105.5,
        'EMA_10': 105.0,
        'EMA_20': 104.0,
        'RSI': 55.0,
        'MACD': 0.8,
        'MACD_Signal': 0.5,
        'MACD_Hist': 0.3,
        'BB_Width': 0.05,
        'BB_Position': 0.6,
        'Return_1d': 0.01,
        'Return_5d': 0.03,
        'Return_10d': 0.05,
        'Volatility_10d': 0.015,
        'Volatility_20d': 0.018,
        'Volume_Ratio': 1.2,
        'High_Low_Range': 0.02,
        'Close_Position': 0.7,
        'Gap': 0.002
    }

    result = model.predict_single(demo_features)
    prediction_count += 1

    result['demo'] = True
    result['input_features'] = demo_features
    result['timestamp'] = datetime.now().isoformat()

    return jsonify(result)


@app.route('/predict/batch', methods=['POST'])
@track_request
@require_model
def predict_batch():
    """
    Prédiction en batch pour plusieurs échantillons.

    Expects JSON body:
    {
        "samples": [
            {"SMA_5": 105.2, ...},
            {"SMA_5": 106.1, ...}
        ]
    }
    """
    global prediction_count, error_count

    try:
        data = request.get_json()

        if not data or 'samples' not in data:
            return jsonify({
                'error': 'Invalid format',
                'message': 'Expected {"samples": [...]}'
            }), 400

        samples = data['samples']
        results = []

        for i, sample in enumerate(samples):
            try:
                result = model.predict_single(sample)
                result['index'] = i
                results.append(result)
                prediction_count += 1
            except Exception as e:
                results.append({
                    'index': i,
                    'error': str(e)
                })
                error_count += 1

        return jsonify({
            'results': results,
            'total': len(samples),
            'successful': sum(1 for r in results if 'error' not in r)
        })

    except Exception as e:
        error_count += 1
        return jsonify({'error': str(e)}), 500


@app.route('/metrics')
def metrics():
    """
    Endpoint Prometheus pour le monitoring.

    Retourne les métriques au format Prometheus.
    """
    uptime = (datetime.now() - start_time).total_seconds()

    metrics_text = f"""# HELP stock_prediction_requests_total Total number of requests
# TYPE stock_prediction_requests_total counter
stock_prediction_requests_total {request_count}

# HELP stock_prediction_predictions_total Total number of predictions
# TYPE stock_prediction_predictions_total counter
stock_prediction_predictions_total {prediction_count}

# HELP stock_prediction_errors_total Total number of errors
# TYPE stock_prediction_errors_total counter
stock_prediction_errors_total {error_count}

# HELP stock_prediction_uptime_seconds Uptime in seconds
# TYPE stock_prediction_uptime_seconds gauge
stock_prediction_uptime_seconds {uptime}

# HELP stock_prediction_model_loaded Model loaded status
# TYPE stock_prediction_model_loaded gauge
stock_prediction_model_loaded {1 if model is not None else 0}
"""

    return Response(metrics_text, mimetype='text/plain')


@app.route('/reload', methods=['POST'])
@track_request
def reload_model():
    """
    Recharge le modèle depuis le disque.

    Utile après un re-entraînement.
    """
    try:
        load_model()
        return jsonify({
            'status': 'success',
            'model_info': model_info
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# ========================================
# ERROR HANDLERS
# ========================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'error': 'Not found',
        'message': 'L\'endpoint demandé n\'existe pas'
    }), 404


@app.errorhandler(500)
def internal_error(e):
    global error_count
    error_count += 1
    return jsonify({
        'error': 'Internal server error',
        'message': 'Une erreur interne est survenue'
    }), 500


# ========================================
# STARTUP
# ========================================

# Charger le modèle au démarrage
with app.app_context():
    load_model()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'

    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║           Stock Prediction API - Flask Server                 ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  URL:     http://localhost:{port}                              ║
    ║  Health:  http://localhost:{port}/health                       ║
    ║  Predict: http://localhost:{port}/predict                      ║
    ║  Demo:    http://localhost:{port}/predict/demo                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
