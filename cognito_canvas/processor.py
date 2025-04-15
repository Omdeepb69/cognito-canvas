# cognito_canvas/processor.py

"""
Backend processing logic for Cognito Canvas.

Handles image preprocessing, handwriting recognition (OCR), mathematical
expression solving, flowchart element detection, code generation from
flowcharts, and content summarization.
"""

import cv2
import numpy as np
import easyocr
import sympy
import logging
import re
from collections import defaultdict

# --- Configuration & Initialization ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize EasyOCR Reader
# Use GPU if available, otherwise CPU. Specify languages.
# This initialization can be time-consuming, so it's done once globally.
try:
    # Consider adding more languages if needed, e.g., ['en', 'es', 'fr']
    # Use gpu=True if a CUDA-enabled GPU is available and PyTorch is installed with CUDA support.
    # Set gpu=False explicitly if you want to force CPU usage or don't have a compatible GPU.
    reader = easyocr.Reader(['en'], gpu=False)
    logging.info("EasyOCR Reader initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize EasyOCR Reader: {e}")
    # Fallback or raise error depending on desired behavior
    reader = None # Indicate failure

# --- Image Processing ---

def process_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Preprocesses an image for better OCR results.

    Args:
        image: Input image as a NumPy array (BGR format).

    Returns:
        Preprocessed image as a NumPy array (grayscale).
        Returns None if the input image is invalid.
    """
    if image is None or image.size == 0:
        logging.error("process_image_for_ocr: Invalid input image.")
        return None

    try:
        # 1. Convert to Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2. Apply Gaussian Blur to reduce noise
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0) # Optional, might blur small text

        # 3. Apply Adaptive Thresholding
        # Better for varying lighting conditions than simple thresholding
        processed_image = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 5 # Block size and C value might need tuning
        )

        # Optional: Dilation/Erosion to connect broken parts or remove noise
        # kernel = np.ones((2,2),np.uint8)
        # processed_image = cv2.dilate(processed_image, kernel, iterations = 1)
        # processed_image = cv2.erode(processed_image, kernel, iterations = 1)

        logging.info("Image preprocessed successfully for OCR.")
        return processed_image
    except cv2.error as e:
        logging.error(f"OpenCV error during image preprocessing: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error during image preprocessing: {e}")
        return None

# --- Handwriting Recognition ---

def recognize_handwriting(image: np.ndarray) -> list:
    """
    Recognizes handwritten text and its bounding boxes from an image using EasyOCR.

    Args:
        image: Input image as a NumPy array (BGR format).

    Returns:
        A list of tuples, where each tuple contains:
        (bounding_box, text, confidence_score).
        Returns an empty list if OCR fails or no text is found, or if reader is not initialized.
    """
    if reader is None:
        logging.error("recognize_handwriting: EasyOCR Reader not initialized.")
        return []
    if image is None or image.size == 0:
        logging.error("recognize_handwriting: Invalid input image.")
        return []

    try:
        # EasyOCR prefers BGR images directly
        results = reader.readtext(image)
        logging.info(f"Handwriting recognition found {len(results)} text blocks.")
        # Format results for consistency (e.g., ensure bounding box is list of points)
        formatted_results = []
        for (bbox, text, prob) in results:
            # bbox is typically [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            formatted_results.append((bbox, text, prob))
        return formatted_results
    except Exception as e:
        logging.error(f"Error during handwriting recognition: {e}")
        return []

# --- Mathematical Expression Solving ---

def solve_mathematical_expression(text: str) -> str:
    """
    Parses and solves a mathematical expression string using SymPy.

    Args:
        text: The mathematical expression as a string.

    Returns:
        The solution as a string, or an error message string if parsing/solving fails.
    """
    if not text or not isinstance(text, str):
        return "Error: Invalid input expression."

    # Basic cleanup: remove common OCR errors or ambiguities if needed
    # Example: Replace 'x' with '*' if it likely means multiplication in context
    # This needs careful handling to avoid breaking variable names like 'x'
    processed_text = text.strip()
    # A simple heuristic: replace 'x' with '*' if surrounded by digits or spaces/digits
    processed_text = re.sub(r'(?<=\d|\s)\s*x\s*(?=\d|\s)', ' * ', processed_text)
    # Replace common visual ambiguities if necessary (e.g., 'I' -> '1', 'O' -> '0') - Use with caution!
    # processed_text = processed_text.replace('I', '1').replace('O', '0')

    logging.info(f"Attempting to solve expression: {processed_text}")

    try:
        # Use sympify to parse the expression
        # Add implicit multiplication parsing if needed (e.g., '2x' -> '2*x')
        expr = sympy.sympify(processed_text, evaluate=True, locals={'pi': sympy.pi, 'e': sympy.E})

        # Evaluate the expression numerically if possible, otherwise simplify
        # Use evalf() for numerical evaluation, simplify() for symbolic simplification
        if expr.is_Relational:
             # Solve equations or inequalities
             solution = sympy.solve(expr)
        elif expr.is_number:
            solution = expr
        else:
            # Try numerical evaluation first
            try:
                # Use N() which is an alias for evalf()
                solution = sympy.N(expr)
            except (TypeError, AttributeError):
                 # Fallback to simplification if numerical evaluation fails
                 solution = sympy.simplify(expr)


        solution_str = str(solution)
        logging.info(f"Expression '{processed_text}' solved. Solution: {solution_str}")
        return solution_str
    except (sympy.SympifyError, TypeError, SyntaxError) as e:
        logging.error(f"Failed to parse or solve expression '{processed_text}': {e}")
        return f"Error: Could not solve '{text}'. Invalid expression or unsupported format."
    except Exception as e:
        logging.error(f"Unexpected error solving expression '{processed_text}': {e}")
        return f"Error: An unexpected error occurred while solving '{text}'."


# --- Flowchart Detection ---

def _classify_shape(approx_poly):
    """Helper function to classify shapes based on approximated polygon vertices."""
    num_vertices = len(approx_poly)
    if num_vertices == 3:
        return "Triangle" # Potentially part of an arrow or other symbol
    elif num_vertices == 4:
        # Could be rectangle or diamond. Check aspect ratio and angles for better classification.
        # For simplicity here, we'll differentiate based on bounding box vs contour area later if needed.
        # A simple check: bounding box aspect ratio
        x, y, w, h = cv2.boundingRect(approx_poly)
        aspect_ratio = float(w)/h if h != 0 else 0
        # Diamonds are often wider than tall or vice-versa significantly, or rotated.
        # Rectangles are typically closer to standard orientations.
        # This is a heuristic and might need refinement.
        if 0.8 < aspect_ratio < 1.2: # More square-like might be diamond (if rotated)
             # Further checks needed for rotation/angles for robust diamond detection
             return "Diamond" # Placeholder, needs better logic
        else:
             return "Rectangle"
    # elif num_vertices > 4: # Could be circle/ellipse if smooth
    #     area = cv2.contourArea(approx_poly)
    #     x,y,w,h = cv2.boundingRect(approx_poly)
    #     radius = w / 2
    #     if abs(1 - (float(w)/h)) <= 0.2 and abs(1 - (area / (np.pi * (radius**2)))) <= 0.2:
    #         return "Circle/Ellipse" # Often used for start/end nodes
    else:
        return "Unknown"

def detect_flowchart_elements(image: np.ndarray) -> list:
    """
    Detects basic flowchart elements (rectangles, diamonds) in an image using OpenCV.
    Associates recognized text with detected shapes.

    Args:
        image: Input image as a NumPy array (BGR format).

    Returns:
        A list of dictionaries, each representing a detected element:
        {'shape': 'Rectangle'|'Diamond'|'Unknown', 'bbox': [x, y, w, h], 'center': (cx, cy), 'text': 'associated_text'}
        Returns an empty list if detection fails or no elements are found.
    """
    if image is None or image.size == 0:
        logging.error("detect_flowchart_elements: Invalid input image.")
        return []

    elements = []
    try:
        # 1. Preprocessing for Shape Detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Use Canny edge detection or adaptive thresholding
        # Canny might be better for distinct shapes
        edges = cv2.Canny(blurred, 50, 150)
        # Or use thresholding (might merge text with shapes)
        # thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                cv2.THRESH_BINARY_INV, 11, 2)

        # Dilate edges to close gaps
        kernel = np.ones((3,3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)


        # 2. Find Contours
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Use RETR_TREE if shapes can be nested, but EXTERNAL is simpler for basic flowcharts

        # 3. Recognize Text (run once on the original image)
        text_results = recognize_handwriting(image) # Use original image for better OCR

        # 4. Filter and Classify Contours
        min_area = 500 # Minimum area to filter out noise - adjust based on expected element size
        detected_shapes = []

        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue

            # Approximate the contour shape
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True) # Adjust epsilon (0.02-0.05)

            shape_type = _classify_shape(approx)

            if shape_type in ["Rectangle", "Diamond"]:
                x, y, w, h = cv2.boundingRect(approx)
                center_x = x + w // 2
                center_y = y + h // 2
                detected_shapes.append({
                    'shape': shape_type,
                    'bbox': [x, y, w, h],
                    'center': (center_x, center_y),
                    'contour': contour, # Keep contour for point-in-polygon test
                    'text': '' # Placeholder for associated text
                })

        # 5. Associate Text with Shapes
        # Find which shape contains the center of each text block
        if text_results:
            for shape in detected_shapes:
                shape_contour = shape['contour']
                shape_center_x, shape_center_y = shape['center']
                min_dist = float('inf')
                associated_text = ""

                for text_bbox, text, _ in text_results:
                    # Calculate center of the text bounding box
                    # EasyOCR bbox format: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                    pts = np.array(text_bbox, dtype=np.int32)
                    text_center_x = int(np.mean(pts[:, 0]))
                    text_center_y = int(np.mean(pts[:, 1]))

                    # Check if text center is inside the shape contour
                    # Use cv2.pointPolygonTest
                    # dist = cv2.pointPolygonTest(shape_contour, (text_center_x, text_center_y), False) >= 0 # True if inside or on edge

                    # Alternative: Check distance between shape center and text center
                    dist = np.sqrt((shape_center_x - text_center_x)**2 + (shape_center_y - text_center_y)**2)

                    # Simple association: find the closest text block within a reasonable distance
                    # A better approach might involve checking overlap percentage or text containment.
                    # For now, associate the closest text block whose center is reasonably near the shape center.
                    # This threshold needs tuning based on typical drawing scale.
                    max_association_distance = max(shape['bbox'][2], shape['bbox'][3]) # Max of width/height

                    if dist < min_dist and dist < max_association_distance:
                         # Check if text center is actually inside the shape's bounding box as a sanity check
                         sx, sy, sw, sh = shape['bbox']
                         if sx <= text_center_x <= sx + sw and sy <= text_center_y <= sy + sh:
                            min_dist = dist
                            associated_text = text

                shape['text'] = associated_text.strip()
                # Remove contour from final output if not needed downstream
                del shape['contour']


        elements = detected_shapes
        logging.info(f"Detected {len(elements)} potential flowchart elements.")

    except cv2.error as e:
        logging.error(f"OpenCV error during flowchart detection: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error during flowchart detection: {e}")
        return []

    return elements


# --- Code Generation ---

def generate_code_from_flowchart(elements: list) -> str:
    """
    Generates basic Python code stubs from detected flowchart elements.
    Assumes a simple top-to-bottom flow based on vertical position.

    Args:
        elements: A list of dictionaries representing flowchart elements,
                  as returned by detect_flowchart_elements.
                  Each dict should have 'shape', 'center', and 'text'.

    Returns:
        A string containing the generated Python code, or a message if no elements provided.
    """
    if not elements:
        return "# No flowchart elements detected to generate code."

    # Sort elements by vertical position (top to bottom)
    elements.sort(key=lambda el: el['center'][1]) # Sort by y-coordinate of center

    code_lines = []
    indent_level = 0

    def add_line(text):
        code_lines.append("    " * indent_level + text)

    add_line("import time # Placeholder import")
    add_line("")

    # Basic mapping: Rectangle -> function call/action, Diamond -> if/else
    # This is highly simplified and doesn't handle loops, complex branches, or actual flow arrows.
    for i, element in enumerate(elements):
        shape = element.get('shape', 'Unknown')
        text = element.get('text', '').strip()
        # Sanitize text to be used as comments or potentially variable/function names
        sanitized_text = re.sub(r'\W|^(?=\d)', '_', text) if text else f"step_{i}"

        if shape == "Rectangle":
            # Assume rectangle represents an action or function call
            if text:
                add_line(f"# Action: {text}")
                # Try to make a plausible function name
                func_name = sanitized_text.lower()
                add_line(f"print('Executing: {text}') # Placeholder for: {func_name}()")
                add_line("time.sleep(0.5) # Simulate action")
            else:
                add_line(f"# Action: Step {i}")
                add_line(f"print('Executing step {i}')")
                add_line("time.sleep(0.5)")
            add_line("") # Add spacing

        elif shape == "Diamond":
            # Assume diamond represents a condition
            condition = text if text else f"condition_{i}"
            add_line(f"if True: # Condition: {condition}")
            indent_level += 1
            add_line("# Code for 'True' branch")
            add_line("print(f'Condition \"{condition}\" is True')")
            add_line("pass # Replace with actual logic")
            indent_level -= 1
            add_line("else:")
            indent_level += 1
            add_line("# Code for 'False' branch")
            add_line("print(f'Condition \"{condition}\" is False')")
            add_line("pass # Replace with actual logic")
            indent_level -= 1
            add_line("") # Add spacing

        # Other shapes could be added here (e.g., Start/End nodes)

    if not code_lines:
         return "# No recognized flowchart shapes found to generate code."

    # Add a basic structure if needed (e.g., wrap in a main function)
    final_code = ["def generated_flowchart_logic():"]
    final_code.extend(["    " + line for line in code_lines])
    final_code.append("\nif __name__ == '__main__':")
    final_code.append("    generated_flowchart_logic()")


    logging.info("Generated Python code stub from flowchart elements.")
    return "\n".join(final_code)


# --- Content Summarization ---

def summarize_canvas_content(image: np.ndarray, text_results: list) -> str:
    """
    Generates a simple summary of the canvas content based on recognized text
    and potentially detected shapes (though shape info isn't used here yet).

    Args:
        image: Input image (NumPy array) - currently unused but available for future enhancements.
        text_results: A list of tuples (bbox, text, confidence) from OCR.

    Returns:
        A string containing a summary of the recognized text.
    """
    if image is None: # Keep image arg for potential future use (e.g., analyzing layout)
        logging.warning("summarize_canvas_content: Input image is None.")
        # Continue with text if available

    if not text_results:
        return "Summary: No text content recognized on the canvas."

    # Simple summarization: Concatenate all recognized text.
    # Could be improved by sorting text spatially, filtering low confidence, using NLP etc.
    full_text = " ".join([res[1] for res in text_results])

    # Basic cleanup of concatenated text
    summary = ' '.join(full_text.split()) # Remove extra whitespace

    # Optional: Add context if shapes were detected (requires passing shape info)
    # num_shapes = len(detected_elements) # If elements were passed
    # summary = f"Detected {num_shapes} shapes.\nNotes: {summary}"

    logging.info("Generated summary from recognized text.")
    return f"Summary:\n{summary}"


# --- Main Processing Function (Example Usage) ---

def process_canvas(image_path: str) -> dict:
    """
    Main function to process a canvas image file.
    Reads an image, performs all processing steps, and returns results.
    This is an example of how the functions might be orchestrated.

    Args:
        image_path: Path to the image file.

    Returns:
        A dictionary containing results from different processing steps:
        {
            'ocr_results': list,
            'math_solutions': list,
            'flowchart_elements': list,
            'generated_code': str,
            'summary': str,
            'error': str or None
        }
    """
    results = {
        'ocr_results': [],
        'math_solutions': [],
        'flowchart_elements': [],
        'generated_code': "",
        'summary': "",
        'error': None
    }

    try:
        # 1. Read Image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image file: {image_path}")

        # 2. Recognize Handwriting/Text
        # Preprocessing for OCR is often handled internally by EasyOCR or can be detrimental.
        # Pass the original color image directly to EasyOCR.
        # If results are poor, try passing the preprocessed image instead:
        # processed_for_ocr = process_image_for_ocr(image)
        # if processed_for_ocr is not None:
        #     results['ocr_results'] = recognize_handwriting(processed_for_ocr) # Pass processed
        # else: # Fallback or handle error
        #     results['ocr_results'] = recognize_handwriting(image) # Pass original
        results['ocr_results'] = recognize_handwriting(image) # Use original image

        # 3. Solve Mathematical Expressions found in text
        for _, text, _ in results['ocr_results']:
            # Basic check if text looks like a math expression (contains numbers and operators)
            # This is a very simple heuristic. More robust detection might be needed.
            if any(c in text for c in '+-*/=') and any(c.isdigit() for c in text):
                 solution = solve_mathematical_expression(text)
                 if solution: # Avoid adding empty solutions or just errors if not desired
                     results['math_solutions'].append({'expression': text, 'solution': solution})

        # 4. Detect Flowchart Elements
        # Use the original image for shape detection as preprocessing might distort shapes
        results['flowchart_elements'] = detect_flowchart_elements(image)

        # 5. Generate Code from Flowchart
        if results['flowchart_elements']:
            results['generated_code'] = generate_code_from_flowchart(results['flowchart_elements'])
        else:
            results['generated_code'] = "# No flowchart elements detected."

        # 6. Summarize Content
        results['summary'] = summarize_canvas_content(image, results['ocr_results'])

    except FileNotFoundError as e:
        logging.error(f"Processing error: {e}")
        results['error'] = str(e)
    except cv2.error as e:
        logging.error(f"OpenCV error during processing: {e}")
        results['error'] = f"Image processing error: {e}"
    except Exception as e:
        logging.error(f"Unexpected error during canvas processing: {e}", exc_info=True)
        results['error'] = f"An unexpected error occurred: {e}"

    return results

# Example usage when running the script directly
if __name__ == '__main__':
    print("Cognito Canvas Processor Module")
    print("Running example processing on a placeholder image path...")

    # Create a dummy white image for testing if no image is provided
    dummy_image_path = "dummy_canvas.png"
    try:
        # Check if dummy file exists, if not create one
        import os
        if not os.path.exists(dummy_image_path):
            dummy_img = np.ones((600, 800, 3), dtype=np.uint8) * 255 # White image
            # Add some sample elements for testing
            # Draw a rectangle
            cv2.rectangle(dummy_img, (100, 100), (300, 200), (0, 0, 0), 2)
            cv2.putText(dummy_img, "Start Process", (110, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            # Draw a diamond (approx)
            pts = np.array([[400, 250], [500, 300], [400, 350], [300, 300]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(dummy_img, [pts], isClosed=True, color=(0,0,0), thickness=2)
            cv2.putText(dummy_img, "x > 10?", (330, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            # Draw some math
            cv2.putText(dummy_img, "2 * (5 + 3) = ?", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.imwrite(dummy_image_path, dummy_img)
            print(f"Created dummy image: {dummy_image_path}")

        # Process the dummy image
        processing_results = process_canvas(dummy_image_path)

        print("\n--- Processing Results ---")
        if processing_results['error']:
            print(f"Error: {processing_results['error']}")
        else:
            print("\nOCR Results:")
            if processing_results['ocr_results']:
                for bbox, text, conf in processing_results['ocr_results']:
                    print(f"- Text: '{text}', Confidence: {conf:.2f}") # Bbox omitted for brevity
            else:
                print("- No text found.")

            print("\nMath Solutions:")
            if processing_results['math_solutions']:
                for item in processing_results['math_solutions']:
                    print(f"- Expression: '{item['expression']}', Solution: {item['solution']}")
            else:
                print("- No mathematical expressions solved.")

            print("\nFlowchart Elements:")
            if processing_results['flowchart_elements']:
                for elem in processing_results['flowchart_elements']:
                    print(f"- Shape: {elem['shape']}, Center: {elem['center']}, Text: '{elem['text']}'")
            else:
                print("- No flowchart elements detected.")

            print("\nGenerated Code:")
            print(processing_results['generated_code'])

            print("\nSummary:")
            print(processing_results['summary'])

    except ImportError:
         print("\nNote: OpenCV is required to create/run the dummy image example.")
    except Exception as e:
        print(f"\nAn error occurred during the example run: {e}")