import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize


class Segmentation:
    # ---------- LINHAS: projeção axial (cinza) + region growth por PDF ----------
    @staticmethod
    def segment_lines(gray_image, binary_image, min_h=8, k_smooth=31, peak_tau=0.05, grow_alpha=0.35):
        """
        - Projeção horizontal do sinal de tinta (255 - cinza)
        - Suavização 1D
        - Detecta picos > peak_tau*max
        - Cresce região para cima/baixo enquanto PDF >= grow_alpha * média local
        (Alinha com: picos da projeção + region growth dependente de PDF). 
        """
        ink = (255 - gray_image).astype(np.float64)
        proj = ink.sum(axis=1)

        k = max(3, k_smooth | 1)
        kernel = np.ones(k, np.float64) / k
        proj_s = np.convolve(proj, kernel, mode="same")

        total = proj_s.sum() + 1e-9
        pdf = proj_s / total
        gmean = pdf.mean()

        # picos candidatos
        peaks = []
        thr_peak = proj_s.max() * peak_tau
        for y in range(1, len(proj_s) - 1):
            if proj_s[y] >= thr_peak and proj_s[y] >= proj_s[y-1] and proj_s[y] >= proj_s[y+1]:
                peaks.append(y)

        used = np.zeros_like(proj_s, dtype=bool)
        boxes = []
        for y0 in peaks:
            if used[y0]:
                continue
            # crescimento com critério em PDF
            left = y0
            right = y0
            local_vals = [pdf[y0]]

            while left - 1 >= 0:
                cand = left - 1
                if pdf[cand] < grow_alpha * (np.mean(local_vals)):
                    break
                left = cand
                local_vals.append(pdf[left])
            while right + 1 < len(pdf):
                cand = right + 1
                if pdf[cand] < grow_alpha * (np.mean(local_vals)):
                    break
                right = cand
                local_vals.append(pdf[right])

            # marca usados e guarda caixa
            used[left:right+1] = True
            if (right - left + 1) >= min_h:
                boxes.append((left, right+1))

        # funde sobreposições e ordena
        boxes.sort()
        merged = []
        for tb in boxes:
            if not merged:
                merged.append(tb)
            else:
                lt, lb = merged[-1]
                if tb[0] <= lb + 2:
                    merged[-1] = (lt, max(lb, tb[1]))
                else:
                    merged.append(tb)

        # extrai imagens de linha do BINÁRIO corrigido (coerente com paper)
        line_images = [binary_image[t:b, :] for (t, b) in merged]
        return line_images, merged

    # ---------- PALAVRAS: projeção vertical + gaps com Otsu (inter-palavra) ----------
    @staticmethod
    def segment_words(line_bin, min_w=10):
        col = np.sum(line_bin, axis=0).astype(np.float32)
        maxv = col.max() + 1e-6
        empty = (col / maxv) < 0.02

        # runs de vazio -> comprimentos (gaps)
        gaps = []
        in_gap, s = False, 0
        for x, v in enumerate(empty):
            if v and not in_gap:
                in_gap, s = True, x
            elif not v and in_gap:
                in_gap = False
                gaps.append((s, x))
        if in_gap:
            gaps.append((s, len(empty)))

        if not gaps:
            return [line_bin], [(0, line_bin.shape[1])]

        glens = np.array([e - s for s, e in gaps], dtype=np.int32)
        # Otsu 1D nos comprimentos de gaps
        hist = np.bincount(glens)
        total = glens.size
        sum_total = np.dot(np.arange(hist.size), hist)
        wB = 0.0
        sumB = 0.0
        maxVar = 0.0
        thr_len = 0
        for t in range(hist.size):
            wB += hist[t]
            if wB == 0: 
                continue
            wF = total - wB
            if wF == 0:
                break
            sumB += t * hist[t]
            mB = sumB / wB
            mF = (sum_total - sumB) / wF
            varBetween = wB * wF * (mB - mF) ** 2
            if varBetween > maxVar:
                maxVar = varBetween
                thr_len = t

        cuts = [0]
        for (s, e) in gaps:
            if (e - s) > max(thr_len, 4):
                cuts.extend([s, e])
        cuts.append(line_bin.shape[1])
        cuts = sorted(set(cuts))

        boxes = []
        for i in range(0, len(cuts) - 1, 2):
            a, b = cuts[i], cuts[i+1]
            if (b - a) >= min_w:
                boxes.append((a, b))
        if not boxes:
            boxes = [(0, line_bin.shape[1])]

        word_images = [line_bin[:, a:b] for (a, b) in boxes]
        return word_images, boxes

    # ---------- CARACTERES: gaps intra-palavra ----------
    @staticmethod
    def segment_characters(word_bin, min_c=4):
        col = np.sum(word_bin, axis=0).astype(np.float32)
        k = max(5, int(0.01 * len(col)) | 1)
        kernel = np.ones(k, np.float32) / k
        col_s = np.convolve(col, kernel, mode="same")
        thr = col_s.max() * 0.25
        is_ink = col_s > thr

        bounds, in_c, s = [], False, 0
        for x, v in enumerate(is_ink):
            if v and not in_c:
                in_c, s = True, x
            elif not v and in_c:
                in_c = False
                e = x
                if (e - s) >= min_c:
                    bounds.append((s, e))
        if in_c:
            e = len(is_ink)
            if (e - s) >= min_c:
                bounds.append((s, e))

        if not bounds:
            bounds = [(0, word_bin.shape[1])]

        char_images = [word_bin[:, a:b] for (a, b) in bounds]
        return char_images, bounds

    # ---------- VISUAIS ----------
    @staticmethod
    def visualize_segmentation(bin_page, lines, words_list, chars_list, save_path=None):
        result = cv2.cvtColor(bin_page, cv2.COLOR_GRAY2BGR)
        # linhas
        for (t, b) in lines:
            cv2.rectangle(result, (0, t), (result.shape[1]-1, b), (0, 0, 255), 2)
        # palavras
        for li, wcoords in enumerate(words_list):
            t, b = lines[li]
            for (a, c) in wcoords:
                cv2.rectangle(result, (a, t), (c, b), (0, 255, 0), 1)
        # caracteres
        for li, line_chars in enumerate(chars_list):
            t, b = lines[li]
            for word_chars in line_chars:
                for (a, c) in word_chars:
                    cv2.rectangle(result, (a, t), (c, b), (255, 0, 0), 1)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title("Segmentação: Linhas (vermelho), Palavras (verde), Caracteres (azul)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()
        return result

    @staticmethod
    def visualize_lines(line_images, save_path=None):
        n = len(line_images)
        if n == 0:
            print("Nenhuma linha detectada!")
            return
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        plt.figure(figsize=(15, 5 * rows))
        for i, line_img in enumerate(line_images):
            plt.subplot(rows, cols, i + 1)
            disp = 255 - line_img
            plt.imshow(disp, cmap="gray", vmin=0, vmax=255)
            plt.title(f"Linha {i+1}")
            plt.axis("off")
        plt.tight_layout()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.show()
        plt.close()

    @staticmethod
    def save_gray(img, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, img)

    @staticmethod
    def segment_words_by_components(line_bin, join_ratio=0.018, min_area=25):
        """
        Segmenta PALAVRAS diretamente dos COMPONENTES CONECTADOS do ESQUELETO da linha.
        - Faz skeleton (1px)
        - Faz um 'join' horizontal leve (dilatação) para unir letras da MESMA palavra
        - Componentes conectados => caixas de palavra
        Retorna:
            word_images: recortes da linha binária por palavra
            word_boxes_x: lista de (x1, x2) relativos à linha (compatível com seu desenho)
            word_boxes_xyxy: lista de (x1, y1, x2, y2) completos na linha
            line_skel: esqueleto 1px (uint8 0/255)
        """
        # 1) esqueleto 1px
        skel = (skeletonize((line_bin > 0)) .astype("uint8")) * 255
        h, w = skel.shape

        # 2) join horizontal pequeno (une letras, não palavras)
        kx = max(3, int(w * float(join_ratio)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1))
        merged = cv2.dilate(skel, kernel, iterations=1)

        # 3) componentes conectados
        num, labels, stats, _ = cv2.connectedComponentsWithStats((merged > 0).astype("uint8"), connectivity=8)

        boxes_xyxy = []
        for lbl in range(1, num):
            x, y, ww, hh, area = stats[lbl]
            if area < min_area or ww < 5 or hh < 5:
                continue
            boxes_xyxy.append((x, y, x + ww, y + hh))

        # ordenar por x
        boxes_xyxy.sort(key=lambda b: b[0])

        # recortes na linha binária
        word_images = [line_bin[y:y2, x:x2] for (x, y, x2, y2) in boxes_xyxy]
        # também como (x1, x2) p/ compatibilidade com seus desenhadores
        word_boxes_x = [(x, x2) for (x, y, x2, y2) in boxes_xyxy]

        return word_images, word_boxes_x, boxes_xyxy, skel

