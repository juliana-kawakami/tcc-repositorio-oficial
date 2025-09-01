import os
import cv2
import numpy as np
from skimage.morphology import skeletonize

class Preprocessing:
    @staticmethod
    def _save_step_image(image, step_name, base_path="etapas_preprocessamento"):
        os.makedirs(base_path, exist_ok=True)
        cv2.imwrite(os.path.join(base_path, f"{step_name}.png"), image)

    @staticmethod
    def preprocess_image(image_path):
        """1) cinza; 2) Otsu invertido (texto=255)."""
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"Erro: Não foi possível carregar a imagem {image_path}")
            return None, None
        Preprocessing._save_step_image(gray, "1_escala_de_cinza")

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        Preprocessing._save_step_image(binary, "2_binarizacao_otsu")
        return gray, binary

    @staticmethod
    def correct_skew(gray, binary, angle_range=(-10, 10), step=1):
        """3) deskew via perfil de projeção (paper). Usa score = std(projeção horizontal)."""
        def score(img_gray):
            proj = np.sum(255 - img_gray, axis=1)
            return np.std(proj)

        h, w = gray.shape
        cx, cy = w // 2, h // 2
        best_a, best_s = 0, -1e9

        for a in range(angle_range[0], angle_range[1] + 1, step):
            M = cv2.getRotationMatrix2D((cx, cy), a, 1.0)
            rg = cv2.warpAffine(gray,   M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            s = score(rg)
            if s > best_s:
                best_s, best_a = s, a

        M = cv2.getRotationMatrix2D((cx, cy), best_a, 1.0)
        gray_corr = cv2.warpAffine(gray,   M, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        bin_corr  = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        Preprocessing._save_step_image(bin_corr, "3_correcao_de_inclinacao")
        return gray_corr, bin_corr

    @staticmethod
    def skeletonize_image(binary_image, save_step=True):
        """4) thinning (1 px)."""
        skel = skeletonize(binary_image > 0).astype("uint8") * 255
        if save_step:
            Preprocessing._save_step_image(skel, "4_esqueletizacao")
        return skel
