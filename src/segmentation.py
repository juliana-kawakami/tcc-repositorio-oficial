import cv2
import numpy as np

class Segmentation:
    @staticmethod
    def segment_lines(image, min_line_height=10):
        # projeção horizontal (1D)
        projection = np.sum(image, axis=1).astype(np.float32)

        # suavização 1D (substitui GaussianBlur que não aceita 1D)
        k = 15
        kernel = np.ones(k, dtype=np.float32) / k
        projection = np.convolve(projection, kernel, mode="same")

        threshold = np.max(projection) * 0.05
        is_text = projection > threshold

        line_boundaries, in_line, start = [], False, 0
        for y, val in enumerate(is_text):
            if val and not in_line:
                in_line, start = True, y
            elif not val and in_line:
                in_line = False
                end = y
                height = end - start
                if height >= min_line_height:
                    margin = int(height * 0.3)
                    top = max(0, start - margin)
                    bottom = min(image.shape[0], end + margin)
                    line_boundaries.append((top, bottom))

        if in_line:
            end = len(is_text)
            height = end - start
            if height >= min_line_height:
                margin = int(height * 0.3)
                top = max(0, start - margin)
                bottom = min(image.shape[0], end + margin)
                line_boundaries.append((top, bottom))

        # remove sobreposições muito próximas
        filtered = []
        for (top, bottom) in line_boundaries:
            if not filtered or (top - filtered[-1][1]) > 5:
                filtered.append((top, bottom))

        line_images = [image[top:bottom, :] for top, bottom in filtered]
        return line_images, filtered

    @staticmethod
    def segment_words(line_image):
        _, binary = cv2.threshold(
            line_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        word_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 15 and h > 10:
                word_boxes.append((x, x + w))

        word_boxes.sort(key=lambda b: b[0])
        word_images = [line_image[:, start:end] for (start, end) in word_boxes]
        return word_images, word_boxes

    @staticmethod
    def segment_characters(word_image):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        processed = cv2.dilate(word_image, kernel, iterations=1)

        projection = np.sum(processed, axis=0).astype(np.float32)
        threshold = np.max(projection) * 0.15
        is_text = projection > threshold

        char_boundaries, in_char, start = [], False, 0
        for x, val in enumerate(is_text):
            if val and not in_char:
                in_char, start = True, x
            elif not val and in_char:
                in_char = False
                end = x
                if (end - start) >= 5:
                    char_boundaries.append((start, end))

        if in_char:
            end = len(is_text)
            if (end - start) >= 5:
                char_boundaries.append((start, end))

        height_threshold = int(word_image.shape[0] * 0.4)
        filtered = []
        for (cstart, cend) in char_boundaries:
            char_img = word_image[:, cstart:cend]
            if np.max(np.sum(char_img, axis=1)) > height_threshold:
                filtered.append((cstart, cend))

        char_images = [word_image[:, s:e] for (s, e) in filtered]
        return char_images, filtered
