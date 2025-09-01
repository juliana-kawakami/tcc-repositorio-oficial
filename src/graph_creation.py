import cv2
import numpy as np
import networkx as nx
from scipy.spatial.distance import euclidean

class GraphKPGraph:
    """
    KPs por desvio angular local (contorno do esqueleto) com janela deslizante.
    Arestas entre KPs consecutivos no contorno; peso = distância euclidiana.
    (Alinha com a seção de KPs e a conversão para grafo do artigo.)
    """

    def __init__(self, curv_thr_deg=8.0, win=7, step=1, nms_radius=5):
        self.graph = nx.Graph()
        self.curv_thr = np.deg2rad(curv_thr_deg)
        self.win = max(3, int(win))
        self.step = max(1, int(step))
        self.nms_r = max(1, int(nms_radius))

    @staticmethod
    def _binary_255(img):
        _, b = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
        return b

    @staticmethod
    def _contours_from_skeleton(skel):
        # esqueleto 1px -> contorno detalhado
        b = GraphKPGraph._binary_255(skel)
        contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return contours

    @staticmethod
    def _angle(p0, p1):
        return np.arctan2(p1[1]-p0[1], p1[0]-p0[0])

    def detect_keypoints(self, skeleton_image):
        kps_all = []
        contours = self._contours_from_skeleton(skeleton_image)

        for cnt in contours:
            pts = cnt[:, 0, :]  # Nx2
            if len(pts) < (self.win + 2):
                continue
            pts = pts[::self.step]
            n = len(pts)
            half = self.win // 2

            # curvatura/Δângulo local
            angles = []
            for i in range(1, n):
                angles.append(self._angle(pts[i-1], pts[i]))
            angles = np.array(angles)

            # varre com janela para medir desvio entre direções separadas por "half"
            dev = []
            for i in range(half, n - half):
                a1 = self._angle(pts[i-half], pts[i])
                a2 = self._angle(pts[i], pts[i+half])
                d = a2 - a1
                d = np.arctan2(np.sin(d), np.cos(d))  # [-pi, pi]
                dev.append(abs(d))
            dev = np.array(dev)

            # pontos candidatos: |Δθ| acima do limiar
            cand_idx = np.where(dev >= self.curv_thr)[0] + half
            cand_pts = [tuple(pts[i]) for i in cand_idx]

            # NMS: remove candidatos muito próximos (mantém maiores desvios)
            vals = dev[cand_idx - half] if len(cand_idx) else np.array([])
            order = np.argsort(-vals) if len(vals) else []
            taken = np.zeros(len(pts), dtype=bool)
            kept = []
            for oi in order:
                i = cand_idx[oi]
                if taken[max(0, i-self.nms_r):min(len(pts), i+self.nms_r+1)].any():
                    continue
                kept.append(tuple(pts[i]))
                taken[i] = True

            # sempre garante extremos do contorno
            if len(pts) > 0:
                kept = [tuple(pts[0])] + kept + [tuple(pts[-1])]
            kps_all.extend([(int(x), int(y)) for (x, y) in kept])

        # remove duplicados preservando ordem
        seen = set()
        uniq = []
        for p in kps_all:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return uniq

    def create_graph(self, keypoints):
        self.graph.clear()
        for idx, (x, y) in enumerate(keypoints):
            self.graph.add_node(idx, pos=(x, y))
        for i in range(len(keypoints) - 1):
            x1, y1 = keypoints[i]
            x2, y2 = keypoints[i + 1]
            w = euclidean((x1, y1), (x2, y2))
            self.graph.add_edge(i, i + 1, weight=w)

    def draw_on_image(self, base_image, save_path=None, show=False):
        if len(base_image.shape) == 2:
            canvas = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
        else:
            canvas = base_image.copy()

        # arestas (vermelho)
        for u, v, data in self.graph.edges(data=True):
            x1, y1 = self.graph.nodes[u]["pos"]
            x2, y2 = self.graph.nodes[v]["pos"]
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 1)

        # nós (verde)
        for _, data in self.graph.nodes(data=True):
            x, y = data["pos"]
            cv2.circle(canvas, (x, y), 2, (0, 255, 0), -1)

        if save_path:
            cv2.imwrite(save_path, canvas)

        if show:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 4))
            plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.tight_layout()
            plt.show()
        return canvas
