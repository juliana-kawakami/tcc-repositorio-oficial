import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

class GraphBasedWriterIdentification:
    def __init__(self):
        self.graph = nx.Graph()

    def detect_keypoints(self, skeleton_image):
        """
        Detecta pontos-chave (alta curvatura) nos contornos do esqueleto.
        Espera imagem binária (0/255) com traços em branco.
        """
        keypoints = []
        # garante binária 0/255
        _, skel = cv2.threshold(skeleton_image, 0, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(skel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) < 3:
                continue
            for i in range(1, len(contour) - 1):
                p0 = contour[i - 1][0]
                p1 = contour[i][0]
                p2 = contour[i + 1][0]
                curvature = self.compute_curvature(p0, p1, p2)
                if curvature > 1.0:  # limiar simples; ajuste conforme seu dataset
                    keypoints.append((int(p1[0]), int(p1[1])))
        return keypoints

    def compute_curvature(self, p0, p1, p2):
        # diferença de ângulo normalizada para [-π, π] (evita wrap-around)
        dx1, dy1 = p1[0] - p0[0], p1[1] - p0[1]
        dx2, dy2 = p2[0] - p1[0], p2[1] - p1[1]
        a1 = np.arctan2(dy1, dx1)
        a2 = np.arctan2(dy2, dx2)
        d = a2 - a1
        d = np.arctan2(np.sin(d), np.cos(d))
        return abs(d)

    def create_graph(self, keypoints):
        self.graph.clear()
        # nós
        for idx, (x, y) in enumerate(keypoints):
            self.graph.add_node(idx, pos=(x, y))
        # arestas simples entre vizinhos consecutivos
        for i in range(len(keypoints) - 1):
            x1, y1 = keypoints[i]
            x2, y2 = keypoints[i + 1]
            distance = euclidean((x1, y1), (x2, y2))
            self.graph.add_edge(i, i + 1, weight=distance)

    def visualize_graph(self, save_path: str = "grafo.png", show: bool = True):
        pos = nx.get_node_attributes(self.graph, "pos")
        # desenha simples
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_size=50,
            node_color="r",
            font_size=8,
            font_color="blue",
        )
        plt.title("Grafo da Escrita")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
