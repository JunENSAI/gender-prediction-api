from fastapi import FastAPI, HTTPException, Query
import joblib

app = FastAPI(title="Gender Prediction API", version="1.0")

# Chargement global du modèle au démarrage de l'API
# On utilise un bloc try/except pour éviter que l'API plante si le fichier manque
try:
    model = joblib.load("model.joblib")
except Exception as e:
    print(f"Erreur de chargement du modèle : {e}")
    model = None

@app.get("/")
def root():
    """
    Route de base pour vérifier que l'API est en ligne.
    """
    return {
        "status": "online",
        "message": "Gender Prediction API is running",
        "model_loaded": model is not None
    }

@app.get("/predict")
def predict(name: str = Query(..., description="Le prénom à prédire")):
    """
    Prédit le genre à partir d'un prénom.
    
    Args:
        name: Str, le prénom fourni dans l'URL.
    Returns:
        JSON avec le nom et le genre prédit ('M' ou 'F').
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé sur le serveur.")

    if not name or len(name.strip()) == 0:
        raise HTTPException(status_code=400, detail="Le nom ne peut pas être vide.")

    # Transformation du nom en liste pour le pipeline scikit-learn
    # On nettoie un peu la chaîne (minuscules, espaces)
    clean_name = name.strip().lower()
    
    # Prédiction (0 = Masculin, 1 = Féminin selon notre encodage précédent)
    prediction = model.predict([clean_name])[0]
    
    gender_label = "F" if prediction == 1 else "M"

    return {
        "name": name,
        "gender": gender_label,
        "probability": None # On pourrait ajouter model.predict_proba si besoin
    }