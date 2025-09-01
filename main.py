from src.preprocessing import Preprocessing
from src.segmentation import Segmentation
from src.graph_creation import GraphKPGraph
import os

def main(image_path: str = "images/CF00315_01.bmp") -> None:
    # 1) tons de cinza + 2) Otsu invertido (salva 1_ e 2_)
    gray, binary = Preprocessing.preprocess_image(image_path)
    if binary is None:
        return

    # 3) correção de inclinação por projeção (aplica no cinza e binário; salva 3_)
    gray_corr, bin_corr = Preprocessing.correct_skew(gray, binary)

    # 4) (opcional) esqueleto da página inteira só como referência visual (salva 4_)
    _ = Preprocessing.skeletonize_image(bin_corr, save_step=True)

    # ===== SEGMENTAÇÃO NAS IMAGENS CORRIGIDAS =====
    os.makedirs("segmentacao/lines", exist_ok=True)
    os.makedirs("segmentacao/words", exist_ok=True)
    os.makedirs("segmentacao/graphs", exist_ok=True)

    # Linhas: projeção axial + region growth por PDF (paper)
    line_imgs, line_boxes = Segmentation.segment_lines(gray_corr, bin_corr)
    Segmentation.visualize_lines(line_imgs, save_path="segmentacao/linhas_preview.png")

    words_per_line, word_imgs_per_line, chars_per_line = [], [], []
    for i, (line_img, (top, bot)) in enumerate(zip(line_imgs, line_boxes), start=1):
        Segmentation.save_gray(line_img, f"segmentacao/lines/line_{i}.png")

        # Palavras por linha: projeção vertical + gaps (distância inter-palavra)
        word_imgs, word_boxes = Segmentation.segment_words(line_img)
        words_per_line.append(word_boxes)
        word_imgs_per_line.append(word_imgs)
        for j, wimg in enumerate(word_imgs, start=1):
            Segmentation.save_gray(wimg, f"segmentacao/words/line_{i}_word_{j}.png")

        # Caracteres por palavra: projeção vertical + gaps intra-palavra
        line_chars = []
        for (w_start, _w_end), wimg in zip(word_boxes, word_imgs):
            c_imgs, c_boxes = Segmentation.segment_characters(wimg)
            # leva coords dos chars para o sistema da linha
            c_boxes_global = [(w_start + a, w_start + b) for (a, b) in c_boxes]
            line_chars.append(c_boxes_global)
        chars_per_line.append(line_chars)

    # Visual composto (linhas=vermelho, palavras=verde, chars=azul)
    Segmentation.visualize_segmentation(
        bin_corr, line_boxes, words_per_line, chars_per_line,
        save_path="segmentacao/segmentacao_composta.png"
    )

    # ===== GRAFOS POR SEGMENTO (LINHA e PALAVRA), SOBREPOSTOS AO ESQUELETO LOCAL =====
    for i, line_img in enumerate(line_imgs, start=1):
        line_skel = Preprocessing.skeletonize_image(line_img, save_step=False)
        g = GraphKPGraph(curv_thr_deg=8.0, win=7, step=1, nms_radius=5)
        kps = g.detect_keypoints(line_skel)
        g.create_graph(kps)
        g.draw_on_image(
            line_skel, save_path=f"segmentacao/graphs/line_{i}_graph.png", show=False
        )

    for i, word_imgs in enumerate(word_imgs_per_line, start=1):
        for j, wimg in enumerate(word_imgs, start=1):
            w_skel = Preprocessing.skeletonize_image(wimg, save_step=False)
            gw = GraphKPGraph(curv_thr_deg=8.0, win=7, step=1, nms_radius=3)
            kps_w = gw.detect_keypoints(w_skel)
            gw.create_graph(kps_w)
            gw.draw_on_image(
                w_skel, save_path=f"segmentacao/graphs/line_{i}_word_{j}_graph.png", show=False
            )

if __name__ == "__main__":
    main()
