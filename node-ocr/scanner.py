import cv2
import json
import logging
from pyzbar.pyzbar import decode

# 1. Configuration du Logger (Indispensable pour un Worker)
# Au lieu des 'print', on garde une trace professionnelle des événements
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - WORKER-OCR - %(levelname)s - %(message)s'
)

def scan_code(image_path: str) -> str:
    # 2. Structure standardisée de la réponse (Le format de l'autre groupe)
    result = {
        "success": False,
        "type": None,
        "content": None,
        "error_message": None
    }

    try:
        logging.info(f"Démarrage du job d'analyse sur l'image : {image_path}")
        img = cv2.imread(image_path)
        
        if img is None:
            result["error_message"] = "Impossible de charger l'image. Fichier introuvable ou illisible."
            logging.error(result["error_message"])
            return json.dumps(result, ensure_ascii=False, indent=2)

        # 3. Tentative 1 : Lecture brute
        decoded_objects = decode(img)
        
        # 4. Traitement d'image si la lecture brute échoue
        if not decoded_objects:
            logging.warning("Échec de la lecture brute, application des filtres de traitement d'image...")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Tentative 2
            decoded_objects = decode(thresh)

        # 5. Formatage du succès
        if decoded_objects:
            obj = decoded_objects[0] # On prend le premier code trouvé
            result["success"] = True
            result["type"] = obj.type # Ex: 'QRCODE' ou 'EAN13'
            result["content"] = obj.data.decode('utf-8')
            logging.info(f"Succès ! Code trouvé -> Type: {result['type']} | Contenu: {result['content']}")
        else:
            result["error_message"] = "Image trop dégradée, aucun code détecté."
            logging.warning(result["error_message"])

    except Exception as e:
        # 6. Interception des crashs inattendus
        result["error_message"] = f"Erreur interne du worker : {str(e)}"
        logging.error(result["error_message"])

    # 7. Renvoi du JSON final
    return json.dumps(result, ensure_ascii=False, indent=2)


# Zone de test local
if __name__ == "__main__":
    # N'oublie pas de mettre une image dans tests/assets/
    chemin_image = "tests/assets/test.png" 
    reponse_json = scan_code(chemin_image)
    
    print("\n--- RÉPONSE QUE L'AMBASSADEUR RECEVRA ---")
    print(reponse_json)