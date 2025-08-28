import os
import cv2
from skimage.morphology import skeletonize

class Preprocessing:
    @staticmethod
    def save_step_image(image, step_name, base_path="etapas_preprocessamento"):
        os.makedirs(base_path, exist_ok=True)
        cv2.imwrite(os.path.join(base_path, f"{step_name}.png"), image)

    @staticmethod
    def preprocess_image(image_path):
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"Erro: Não foi possível carregar a imagem {image_path}")
            return None

        Preprocessing.save_step_image(img_gray, "1_escala_de_cinza")

        # binarização (texto em branco = 255)
        _, binary = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        Preprocessing.save_step_image(binary, "2_binarizacao_otsu")
        return binary

    @staticmethod
    def skeletonize_image(binary_image):
        # skimage espera booleano
        binary_bool = binary_image > 0
        skeleton = skeletonize(binary_bool).astype("uint8") * 255
        Preprocessing.save_step_image(skeleton, "4_esqueletizacao")
        return skeleton
