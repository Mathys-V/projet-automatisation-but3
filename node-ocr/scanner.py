import cv2
import json
import logging
import os
import glob
import numpy as np

# Importation des moteurs de lecture (Pattern de redondance)
from pyzbar.pyzbar import decode as pyzbar_decode
import zxingcpp

# -----------------------------------------------------------------------------
# CONFIGURATION DU LOGGING
# -----------------------------------------------------------------------------
# En architecture microservices (Docker), la sortie standard (stdout) est le 
# seul moyen fiable de capturer l'etat de l'application. Le format inclut 
# l'horodatage, le nom du service et le niveau de severite pour l'agregation.
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - WORKER-OCR - %(levelname)s - %(message)s'
)

# -----------------------------------------------------------------------------
# LOGIQUE DE TRAITEMENT D'IMAGE
# -----------------------------------------------------------------------------
def apply_image_filters(image):
    """
    Genere une sequence de variantes de l'image source pour forcer la lecture 
    des codes degrades. 
    
    Architecture : Pattern "Chain of Responsibility" (Chaine de responsabilite).
    Chaque filtre adresse un probleme optique specifique. Le pipeline est ordonne
    du filtre le moins couteux en ressources au plus lourd.

    Args:
        image (numpy.ndarray): L'image brute chargee via OpenCV.

    Returns:
        list: Une liste de tuples contenant le nom du filtre et la matrice de l'image.
    """
    filters = []
    
    # 1. Conversion en niveaux de gris (Base pour les algorithmes de computer vision)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filters.append(("Grayscale", gray))
    
    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Resout les problemes de sous-exposition ou de reflets locaux en ajustant 
    # le contraste par grilles (tiles) plutot que sur l'image globale.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrasted = clahe.apply(gray)
    filters.append(("High Contrast", contrasted))
    
    # 3. Flou Gaussien suivi d'un Seuillage d'Otsu
    # Binarise l'image (noir et blanc strict) en calculant le seuil optimal 
    # mathematiquement. Le flou prealable evite que le bruit du capteur ne 
    # fausse le calcul du seuil.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    filters.append(("Otsu Threshold", thresh))
    
    # 4. Filtre de Nettete (Sharpening via convolution)
    # Augmente artificiellement les micro-contrastes sur les bords. Particulierement 
    # efficace pour les codes-barres pris hors focus (flous).
    kernel_sharpening = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel_sharpening)
    filters.append(("Sharpened", sharpened))
    
    # 5. Morphologie mathematique (Fermeture)
    # Dilate puis erode les pixels noirs. Permet de reboucher physiquement 
    # les micro-rayures ou les dechirures sur l'etiquette d'un code-barres.
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_morph)
    filters.append(("Morphology Close", morph_closed))

    return filters

def scan_with_engines(image) -> dict:
    """
    Encapsule la logique de double lecture. 
    L'objectif est d'assurer une haute disponibilite de la donnee (Redondance).
    
    Args:
        image (numpy.ndarray): L'image a analyser (brute ou filtree).

    Returns:
        dict: Dictionnaire structure contenant le statut, le type, le contenu 
              et le moteur ayant abouti.
    """
    
    # Moteur 1 : Pyzbar (Implementation historique, tres rapide sur codes nets)
    pyzbar_result = pyzbar_decode(image)
    if pyzbar_result:
        return {
            "success": True, 
            "type": pyzbar_result[0].type, 
            "content": pyzbar_result[0].data.decode('utf-8'),
            "engine": "pyzbar"
        }
    
    # Moteur 2 : ZXing (Portage C++ du standard industriel Android)
    # Plus lent mais possede des heuristiques de reparation avancees pour 
    # les codes inclines ou partiellement detruits.
    zxing_result = zxingcpp.read_barcodes(image)
    if zxing_result:
        return {
            "success": True, 
            "type": str(zxing_result[0].format).replace("BarcodeFormat.", ""), 
            "content": zxing_result[0].text,
            "engine": "zxing"
        }
        
    # Echec des deux moteurs sur cette image specifique
    return {"success": False}

# -----------------------------------------------------------------------------
# FONCTION PRINCIPALE (Point d'entree du Worker)
# -----------------------------------------------------------------------------
def scan_code(image_path: str) -> str:
    """
    Traite une image et retourne le resultat de l'OCR sous format JSON.
    Implemente un pattern de "Graceful Degradation" (Degradation gracieuse) : 
    on tente d'abord l'approche la plus legere, et on n'alloue de la puissance 
    CPU (filtres) qu'en cas d'echec.

    Args:
        image_path (str): Le chemin absolu ou relatif vers le fichier image.

    Returns:
        str: Une chaine de caracteres formattee en JSON, prete pour transmission reseau.
    """
    
    # Structure de reponse standardisee (Contrat d'interface avec l'API Ambassadeur)
    result = {
        "success": False,
        "type": None,
        "content": None,
        "error_message": None
    }

    try:
        logging.info(f"Demarrage de l'analyse : {os.path.basename(image_path)}")
        img = cv2.imread(image_path)
        
        # Validation de l'input (Prevention des crashs dus a des I/O defaillants)
        if img is None:
            result["error_message"] = "Impossible de charger l'image. Fichier corrompu ou chemin invalide."
            logging.error(result["error_message"])
            return json.dumps(result, ensure_ascii=False, indent=2)

        # ETAPE 1 : Tentative nominale (Image brute)
        scan_data = scan_with_engines(img)
        
        if scan_data["success"]:
            logging.info(f"Succes lecture brute (Moteur: {scan_data['engine']})")
        else:
            # ETAPE 2 : Strategie de repli (Filtres + Double Moteur)
            logging.warning("Lecture brute echouee. Essai du pipeline de filtres de secours...")
            for filter_name, filtered_img in apply_image_filters(img):
                scan_data = scan_with_engines(filtered_img)
                if scan_data["success"]:
                    logging.info(f"Code recupere grace au filtre [{filter_name}] via {scan_data['engine']}")
                    break # Interruption de la boucle des le premier resultat positif

        # Consolidation des resultats pour la serialisation JSON
        if scan_data["success"]:
            result["success"] = True
            result["type"] = scan_data["type"]
            result["content"] = scan_data["content"]
            logging.info(f"Resultat extrait -> {result['type']} : {result['content']}")
        else:
            result["error_message"] = "Image trop degradee. Aucun moteur ni filtre n'a reussi."
            logging.warning(result["error_message"])

    except Exception as e:
        # Tolerence aux pannes (Fault Tolerance) : 
        # Empeche la fermeture inopinee du processus Worker en cas d'exception non geree (ex: OOM).
        result["error_message"] = f"Erreur critique lors de l'execution du worker : {str(e)}"
        logging.error(result["error_message"])

    # Serialisation finale (ensure_ascii=False permet de conserver les caracteres speciaux)
    return json.dumps(result, ensure_ascii=False, indent=2)

# -----------------------------------------------------------------------------
# HARNAIS DE TEST (Mode Batch local)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    dossier_test = "tests/assets"
    extensions = ('*.jpg', '*.jpeg', '*.png')
    fichiers_images = []
    
    # Agregation des fichiers correspondants aux extensions cibles
    for ext in extensions:
        fichiers_images.extend(glob.glob(os.path.join(dossier_test, ext)))
    
    if not fichiers_images:
        logging.error(f"Aucune image trouvee dans le repertoire cible '{dossier_test}'.")
    else:
        print(f"\n{'='*60}\nLANCEMENT DU WORKER BATCH ({len(fichiers_images)} fichiers)\n{'='*60}\n")
        
        succes_count = 0
        for chemin_image in fichiers_images:
            reponse_json = scan_code(chemin_image)
            reponse_dict = json.loads(reponse_json)
            
            if reponse_dict["success"]:
                succes_count += 1
                
            print("\n--- PAYLOAD JSON GENERE ---")
            print(reponse_json)
            print("-" * 60 + "\n")
            
        print(f"BILAN DES TESTS : {succes_count}/{len(fichiers_images)} lectures reussies.")