from src.preprocessing import Preprocessing
from src.graph_creation import GraphBasedWriterIdentification

def main(image_path: str) -> None:
    # 1) Pré-processa (grayscale + binarização)
    binary_image = Preprocessing.preprocess_image(image_path)
    if binary_image is None:
        return

    # 2) Esqueletiza
    skeleton_image = Preprocessing.skeletonize_image(binary_image)

    # 3) Gera grafo a partir dos keypoints
    model = GraphBasedWriterIdentification()
    keypoints = model.detect_keypoints(skeleton_image)
    model.create_graph(keypoints)
    model.visualize_graph()  # salva e/ou mostra a figura

if __name__ == "__main__":
    main("images/CF00315_01.bmp")
