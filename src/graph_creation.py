import cv2
import numpy as np
import networkx as nx
from scipy.spatial.distance import euclidean

class GraphKPGraph:
    """
    KPs por desvio angular local no CONTORNO do esqueleto.
    Arestas: apenas entre KPs ADJACENTES dentro do MESMO contorno (stroke).
    Peso: distância euclidiana (paper).
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
        # esqueleto 1px -> contornos
        b = GraphKPGraph._binary_255(skel)
        contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return contours

    @staticmethod
    def _angle(p0, p1):
        return np.arctan2(p1[1]-p0[1], p1[0]-p0[0])

    def detect_keypoints(self, skeleton_image, return_sequences=False):
        """
        Retorna:
          - se return_sequences=True: List[List[(x,y)]] (uma sequência por contorno)
          - caso contrário: lista achatada (compatibilidade)
        """
        seqs = []
        contours = self._contours_from_skeleton(skeleton_image)

        for cnt in contours:
            pts = cnt[:, 0, :]  # Nx2
            if len(pts) < (self.win + 2):
                continue
            pts = pts[::self.step]
            n = len(pts)
            half = self.win // 2

            # desvio angular local
            dev = []
            for i in range(half, n - half):
                a1 = self._angle(pts[i-half], pts[i])
                a2 = self._angle(pts[i], pts[i+half])
                d = a2 - a1
                d = np.arctan2(np.sin(d), np.cos(d))  # [-pi, pi]
                dev.append(abs(d))
            dev = np.array(dev) if len(dev) else np.zeros(0)

            # candidatos: |Δθ| >= limiar
            cand_idx = (np.where(dev >= self.curv_thr)[0] + half) if len(dev) else np.array([], dtype=int)

            # NMS simples ao longo da sequência
            vals = dev[cand_idx - half] if len(cand_idx) else np.array([])
            order = np.argsort(-vals) if len(vals) else []
            taken = np.zeros(len(pts), dtype=bool)
            kept_idxs = []
            for oi in order:
                i = cand_idx[oi]
                if taken[max(0, i-self.nms_r):min(len(pts), i+self.nms_r+1)].any():
                    continue
                kept_idxs.append(i)
                taken[i] = True

            # garante extremos
            if len(pts) > 0:
                kept_idxs = [0] + sorted(kept_idxs) + [len(pts)-1]

            seq = [(int(pts[i][0]), int(pts[i][1])) for i in kept_idxs]
            # remove duplicatas consecutivas
            seq_dedup = [seq[0]] if seq else []
            for p in seq[1:]:
                if p != seq_dedup[-1]:
                    seq_dedup.append(p)
            if len(seq_dedup) >= 2:
                seqs.append(seq_dedup)

        if return_sequences:
            return seqs
        # compat: lista achatada
        flat = [p for s in seqs for p in s]
        return flat

    def create_graph_from_sequences(self, kp_sequences):
        """
        kp_sequences: List[List[(x,y)]] — uma lista de sequências (um stroke por contorno).
        Cria arestas APENAS entre vizinhos dentro de cada sequência.
        """
        self.graph.clear()
        node_idx = 0
        for seq in kp_sequences:
            if len(seq) < 2:
                continue
            # adiciona nós da sequência
            idxs = []
            for (x, y) in seq:
                self.graph.add_node(node_idx, pos=(x, y))
                idxs.append(node_idx)
                node_idx += 1
            # liga apenas vizinhos da própria sequência
            for a, b in zip(idxs[:-1], idxs[1:]):
                (x1, y1) = self.graph.nodes[a]["pos"]
                (x2, y2) = self.graph.nodes[b]["pos"]
                w = euclidean((x1, y1), (x2, y2))
                self.graph.add_edge(a, b, weight=w)

    def create_graph(self, keypoints):
        """
        Compatibilidade: se receber uma lista simples [(x,y)], trata como UMA sequência.
        Preferir usar create_graph_from_sequences(...) no pipeline.
        """
        if not keypoints:
            self.graph.clear()
            return
        if isinstance(keypoints[0], (list, tuple)) and len(keypoints) > 0 and isinstance(keypoints[0][0], (int, np.integer)):
            # lista plana
            self.create_graph_from_sequences([keypoints])
        else:
            # já é lista de sequências
            self.create_graph_from_sequences(keypoints)

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
