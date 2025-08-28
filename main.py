from src.preprocessing import Preprocessing
from src.segmentation import Segmentation
from src.graph_creation import GraphCreation
from src.classifier import Classifier

def main(image_path):
    # 1. Pré-processamento
    preprocessor = Preprocessing()
    preprocessed_image = preprocessor.preprocess_image(image_path)

    # 2. Corrigir inclinação e esqueletização
    corrected_image = preprocessor.correct_skew(preprocessed_image)
    skeleton_image = preprocessor.skeletonize_image(corrected_image)

    # 3. Segmentação (linhas, palavras, caracteres)
    line_images, lines = Segmentation.segment_lines(skeleton_image)
    all_words = []
    all_chars = []

    for line_img in line_images:
        word_images, word_coords = Segmentation.segment_words(line_img)
        all_words.append(word_coords)

        line_chars = []
        for word_img in word_images:
            char_images, char_coords = Segmentation.segment_characters(word_img)
            line_chars.append(char_coords)
        all_chars.append(line_chars)

    # 4. Construção do grafo
    graph_creator = GraphCreation()
    keypoints = graph_creator.detect_keypoints(skeleton_image)
    graph = graph_creator.create_graph(keypoints)

    # 5. Extração de características e classificação
    classifier = Classifier()
    features = classifier.extract_features(graph)
    writer_id = classifier.predict(features)

    print(f"Identidade do escritor: {writer_id}")

if __name__ == "__main__":
    main("caminho_da_imagem.jpg")
