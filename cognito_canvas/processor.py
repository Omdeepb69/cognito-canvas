# cognito_canvas/processor.py

"""
Enhanced backend processing logic for Cognito Canvas.

Handles image preprocessing, handwriting recognition (OCR), mathematical
expression solving, flowchart element detection, code generation from
flowcharts, content summarization, and AI-assisted enhancements using Gemini 2.5.
"""

import cv2
import numpy as np
import easyocr
import sympy
import logging
import re
import os
import json
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional, Union
import time
import base64
import requests
from PIL import Image
import io

# Import for Gemini integration
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Google Generative AI package not installed. Gemini features will be disabled.")

# --- Configuration & Initialization ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize EasyOCR Reader with expanded language support
SUPPORTED_LANGUAGES = ['en', 'es', 'fr', 'de', 'zh', 'ja', 'ko', 'ar', 'hi']
DEFAULT_LANGUAGES = ['en']  # Default to English

# Initialize Gemini API
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', None)
if GEMINI_API_KEY and GEMINI_AVAILABLE:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Configure Gemini model
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        gemini_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        logging.info("Gemini 2.5 model initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize Gemini model: {e}")
        gemini_model = None
else:
    gemini_model = None

# Dictionary of OCR readers for different language combinations
ocr_readers = {}

def get_ocr_reader(languages: List[str] = None, gpu: bool = False) -> easyocr.Reader:
    """
    Gets or initializes an EasyOCR reader for the specified languages.
    Caches readers to avoid reinitializing.
    
    Args:
        languages: List of language codes to recognize
        gpu: Whether to use GPU acceleration
    
    Returns:
        An initialized EasyOCR reader
    """
    global ocr_readers
    
    if not languages:
        languages = DEFAULT_LANGUAGES
    
    # Sort languages to ensure consistent cache key
    lang_key = "-".join(sorted(languages))
    reader_key = f"{lang_key}_{gpu}"
    
    if reader_key not in ocr_readers:
        try:
            ocr_readers[reader_key] = easyocr.Reader(languages, gpu=gpu)
            logging.info(f"Initialized EasyOCR reader for languages: {languages}")
        except Exception as e:
            logging.error(f"Failed to initialize EasyOCR reader for {languages}: {e}")
            # Fall back to English if available
            if lang_key != "en":
                logging.info("Falling back to English OCR")
                return get_ocr_reader(["en"], gpu)
            # Otherwise return None
            return None
    
    return ocr_readers[reader_key]

# --- Image Processing ---

def process_image_for_ocr(image: np.ndarray, enhance_contrast: bool = True) -> np.ndarray:
    """
    Preprocesses an image for better OCR results with enhanced contrast options.

    Args:
        image: Input image as a NumPy array (BGR format)
        enhance_contrast: Whether to apply adaptive contrast enhancement

    Returns:
        Preprocessed image as a NumPy array (grayscale)
    """
    if image is None or image.size == 0:
        logging.error("process_image_for_ocr: Invalid input image.")
        return None

    try:
        # Convert to Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhanced contrast with CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        # Noise reduction
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Adaptive Thresholding
        processed_image = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 5
        )
        
        # Morphological operations to improve text quality
        kernel = np.ones((1, 1), np.uint8)
        processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_CLOSE, kernel)
        
        logging.info("Image preprocessed successfully for OCR.")
        return processed_image
    except cv2.error as e:
        logging.error(f"OpenCV error during image preprocessing: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error during image preprocessing: {e}")
        return None

# --- Handwriting Recognition ---

def recognize_handwriting(
    image: np.ndarray, 
    languages: List[str] = None,
    use_gemini: bool = False
) -> list:
    """
    Recognizes handwritten text from an image using EasyOCR or Gemini 2.5 vision.

    Args:
        image: Input image as a NumPy array (BGR format)
        languages: List of language codes to use for recognition
        use_gemini: Whether to use Gemini 2.5 for text recognition

    Returns:
        A list of tuples, where each tuple contains:
        (bounding_box, text, confidence_score)
    """
    if image is None or image.size == 0:
        logging.error("recognize_handwriting: Invalid input image.")
        return []
    
    # Try Gemini first if requested and available
    if use_gemini and gemini_model:
        try:
            start_time = time.time()
            logging.info("Attempting handwriting recognition with Gemini 2.5...")
            
            # Convert image to bytes for Gemini
            success, buffer = cv2.imencode(".jpg", image)
            if not success:
                logging.error("Failed to encode image for Gemini")
                # Fall back to EasyOCR
            else:
                # Create base64 encoded image
                image_bytes = buffer.tobytes()
                
                # Create prompt with the image
                gemini_response = gemini_model.generate_content([
                    "Extract all text from this handwritten or printed document. Return ONLY the text content.",
                    {"mime_type": "image/jpeg", "data": base64.b64encode(image_bytes).decode('utf-8')}
                ])
                
                # Process Gemini response
                extracted_text = gemini_response.text.strip()
                if extracted_text:
                    # Since Gemini doesn't provide bounding boxes, we create a single entry
                    # covering the entire image
                    h, w = image.shape[:2]
                    bbox = [[0, 0], [w, 0], [w, h], [0, h]]  # Full image bounding box
                    result = [(bbox, extracted_text, 0.95)]  # Assume high confidence
                    
                    logging.info(f"Gemini recognition completed in {time.time() - start_time:.2f}s")
                    return result
                else:
                    logging.info("Gemini didn't extract any text, falling back to EasyOCR")
        except Exception as e:
            logging.error(f"Error using Gemini for text recognition: {e}")
            # Fall back to EasyOCR
    
    # Use EasyOCR as fallback or primary method
    try:
        reader = get_ocr_reader(languages)
        if not reader:
            logging.error("No OCR reader available")
            return []
        
        # Process the image with EasyOCR
        start_time = time.time()
        results = reader.readtext(image)
        logging.info(f"EasyOCR recognition completed in {time.time() - start_time:.2f}s, found {len(results)} text blocks")
        
        # Format results
        formatted_results = []
        for (bbox, text, prob) in results:
            formatted_results.append((bbox, text, prob))
        
        # If Gemini is available, we can use it to improve the recognition results
        if gemini_model and formatted_results and not use_gemini:
            try:
                # Extract text from results
                extracted_text = " ".join([text for _, text, _ in formatted_results])
                
                # Ask Gemini to improve/correct the text
                gemini_response = gemini_model.generate_content([
                    f"The following text was extracted from an image using OCR and may contain errors. "
                    f"Please correct any obvious OCR errors while preserving the original meaning:\n\n{extracted_text}"
                ])
                
                corrected_text = gemini_response.text.strip()
                
                # For now, just log the corrected text
                logging.info(f"Gemini corrected text: {corrected_text}")
                
                # In a real system, you might want to update the OCR results with the corrected text
                # or return both versions
            except Exception as e:
                logging.error(f"Error using Gemini to improve OCR results: {e}")
        
        return formatted_results
    except Exception as e:
        logging.error(f"Error during handwriting recognition: {e}")
        return []

# --- Mathematical Expression Solving ---

def solve_mathematical_expression(text: str, use_gemini: bool = False) -> str:
    """
    Parses and solves a mathematical expression using SymPy or Gemini 2.5.

    Args:
        text: The mathematical expression as a string
        use_gemini: Whether to use Gemini for complex expressions

    Returns:
        The solution as a string, or an error message
    """
    if not text or not isinstance(text, str):
        return "Error: Invalid input expression."
    
    # Clean and preprocess the text
    processed_text = text.strip()
    # Replace common OCR errors in math expressions
    processed_text = re.sub(r'(?<=\d|\s)\s*x\s*(?=\d|\s)', ' * ', processed_text)
    processed_text = processed_text.replace('×', '*').replace('÷', '/')
    
    # Try Gemini for complex expressions if requested and available
    if use_gemini and gemini_model and ('integral' in processed_text.lower() or 
                                        'derivative' in processed_text.lower() or
                                        'limit' in processed_text.lower() or
                                        'sum' in processed_text.lower()):
        try:
            logging.info(f"Using Gemini to solve complex expression: {processed_text}")
            
            # Create prompt for Gemini
            prompt = f"""
            Solve the following mathematical expression. Show your work step by step:
            
            {processed_text}
            
            Return the solution in a clear format.
            """
            
            gemini_response = gemini_model.generate_content(prompt)
            solution = gemini_response.text.strip()
            
            logging.info(f"Gemini solution for '{processed_text}': {solution}")
            return solution
        except Exception as e:
            logging.error(f"Error using Gemini for math solving: {e}")
            # Fall back to SymPy
    
    # Use SymPy for standard expressions
    logging.info(f"Attempting to solve expression with SymPy: {processed_text}")
    
    try:
        # Handle equation solving vs expression evaluation
        if "=" in processed_text:
            # This is an equation to solve
            sides = processed_text.split("=")
            if len(sides) != 2:
                return f"Error: Invalid equation format in '{text}'"
            
            left_side = sympy.sympify(sides[0].strip())
            right_side = sympy.sympify(sides[1].strip())
            equation = sympy.Eq(left_side, right_side)
            
            # Find all symbols in the equation
            symbols = list(equation.free_symbols)
            if not symbols:
                return f"Error: No variables found in equation '{text}'"
            
            # Solve for the first symbol (usually x)
            solution = sympy.solve(equation, symbols[0])
            solution_str = f"{symbols[0]} = {solution}"
        else:
            # This is an expression to evaluate
            expr = sympy.sympify(processed_text, evaluate=True)
            
            # Handle different types of expressions
            if expr.is_Relational:
                # Handle inequalities
                solution = sympy.solve(expr)
                solution_str = str(solution)
            elif expr.is_number:
                # Simple numeric expression
                solution = expr
                solution_str = str(float(solution))
            else:
                # Try numerical evaluation first
                try:
                    solution = sympy.N(expr)
                    solution_str = str(solution)
                except (TypeError, AttributeError):
                    # Fallback to simplification
                    solution = sympy.simplify(expr)
                    solution_str = str(solution)
        
        logging.info(f"Expression '{processed_text}' solved. Solution: {solution_str}")
        return solution_str
    except (sympy.SympifyError, TypeError, SyntaxError) as e:
        logging.error(f"Failed to parse or solve expression '{processed_text}': {e}")
        
        # Try Gemini as fallback if available
        if gemini_model:
            try:
                logging.info(f"Attempting to solve with Gemini as fallback: {processed_text}")
                prompt = f"Solve this mathematical expression: {processed_text}"
                gemini_response = gemini_model.generate_content(prompt)
                solution = gemini_response.text.strip()
                logging.info(f"Gemini fallback solution for '{processed_text}': {solution}")
                return solution
            except Exception as gemini_error:
                logging.error(f"Gemini fallback also failed: {gemini_error}")
        
        return f"Error: Could not solve '{text}'. Invalid expression or unsupported format."
    except Exception as e:
        logging.error(f"Unexpected error solving expression '{processed_text}': {e}")
        return f"Error: An unexpected error occurred while solving '{text}'."

# --- Flowchart and Shape Detection ---

def detect_shapes(image: np.ndarray) -> List[Dict]:
    """
    Detects basic shapes (rectangles, circles, triangles, diamonds) in an image.
    
    Args:
        image: Input image as a NumPy array (BGR format)
        
    Returns:
        A list of dictionaries, each representing a detected shape
    """
    if image is None or image.size == 0:
        logging.error("detect_shapes: Invalid input image.")
        return []

    shapes = []
    try:
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply blur and edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate to close gaps in edges
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Minimum area to consider to filter out noise
        min_area = 500
        
        # Process each contour
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            
            # Approximate the contour
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            
            # Get bounding box and center
            x, y, w, h = cv2.boundingRect(approx)
            center = (x + w // 2, y + h // 2)
            
            # Determine shape type based on number of vertices
            vertices = len(approx)
            shape_type = "unknown"
            
            # For circle detection
            area = cv2.contourArea(contour)
            radius = w / 2
            # Check if shape is approximately circular
            circle_area_ratio = abs(1 - (area / (np.pi * (radius**2))))
            aspect_ratio = float(w) / h if h > 0 else 0
            
            if vertices == 3:
                shape_type = "triangle"
            elif vertices == 4:
                # Check if square or rectangle or diamond
                aspect_ratio = float(w) / h if h > 0 else 0
                if 0.95 <= aspect_ratio <= 1.05:
                    # Check if it's a diamond (rotated square)
                    # Calculate standard deviation of x and y coordinates
                    coords = approx.reshape(-1, 2)
                    std_x = np.std(coords[:, 0])
                    std_y = np.std(coords[:, 1])
                    if abs(std_x - std_y) < 10:  # Similar variation in x and y -> likely square
                        shape_type = "square"
                    else:
                        shape_type = "diamond"
                else:
                    shape_type = "rectangle"
            elif vertices >= 5 and vertices <= 10 and circle_area_ratio < 0.2:
                # Approximate circles have consistent distance from center to edge
                shape_type = "circle"
            elif vertices > 10:
                # Many vertices suggests a circle or complex shape
                shape_type = "circle" if circle_area_ratio < 0.2 else "complex"
            
            shapes.append({
                "shape": shape_type,
                "bbox": [x, y, w, h],
                "center": center,
                "vertices": vertices,
                "contour": contour.tolist(),  # Convert to list for serialization
                "text": ""  # Placeholder for text to be associated
            })
        
        logging.info(f"Detected {len(shapes)} shapes in the image")
        return shapes
    except Exception as e:
        logging.error(f"Error detecting shapes: {e}")
        return []

def detect_flowchart_elements(image: np.ndarray, use_gemini: bool = False) -> List[Dict]:
    """
    Detects flowchart elements including shapes and connections.
    
    Args:
        image: Input image as a NumPy array (BGR format)
        use_gemini: Whether to use Gemini for more accurate detection
        
    Returns:
        A list of dictionaries representing detected elements
    """
    if image is None or image.size == 0:
        logging.error("detect_flowchart_elements: Invalid input image.")
        return []
    
    # Try using Gemini to analyze the flowchart if requested and available
    if use_gemini and gemini_model:
        try:
            logging.info("Using Gemini for flowchart analysis...")
            
            # Convert image to bytes for Gemini
            success, buffer = cv2.imencode(".jpg", image)
            if not success:
                logging.error("Failed to encode image for Gemini flowchart analysis")
            else:
                # Create base64 encoded image
                image_bytes = buffer.tobytes()
                
                # Create prompt with the image
                prompt = """
                Analyze this flowchart image. Identify all elements including:
                1. Shapes (rectangles, diamonds, circles, etc.)
                2. Text within each shape
                3. Connections between shapes (arrows)
                
                Return the results in JSON format with this structure:
                {
                  "elements": [
                    {
                      "shape": "rectangle|diamond|circle|etc.",
                      "text": "content inside the shape",
                      "position": "top|middle|bottom|etc."
                    },
                    ...
                  ],
                  "connections": [
                    {
                      "from": "text of source shape",
                      "to": "text of destination shape",
                      "label": "text on the arrow (if any)"
                    },
                    ...
                  ]
                }
                """
                
                gemini_response = gemini_model.generate_content([
                    prompt,
                    {"mime_type": "image/jpeg", "data": base64.b64encode(image_bytes).decode('utf-8')}
                ])
                
                # Process Gemini response
                response_text = gemini_response.text
                
                # Extract the JSON part from the response
                json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = response_text
                
                # Try to parse as JSON
                try:
                    flowchart_data = json.loads(json_str)
                    logging.info(f"Gemini successfully analyzed flowchart: {len(flowchart_data.get('elements', []))} elements")
                    
                    # Convert to our format
                    elements = []
                    for idx, element in enumerate(flowchart_data.get('elements', [])):
                        elements.append({
                            'shape': element.get('shape', 'unknown').lower(),
                            'text': element.get('text', ''),
                            'position': element.get('position', ''),
                            # Add placeholder for bbox and center since Gemini doesn't provide pixel coordinates
                            'bbox': [0, 0, 100, 50],  # Placeholder
                            'center': (50, 25),  # Placeholder
                            'index': idx  # Keep track of order
                        })
                    
                    # We can also store the connections information
                    connections = flowchart_data.get('connections', [])
                    
                    # TODO: If we need actual bounding boxes, we could combine this with OpenCV shape detection
                    return elements
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse Gemini flowchart response as JSON: {e}")
                    logging.error(f"Raw response: {response_text}")
        except Exception as e:
            logging.error(f"Error using Gemini for flowchart analysis: {e}")
    
    # Fall back to traditional CV approach
    try:
        # 1. Detect shapes
        shapes = detect_shapes(image)
        
        # 2. Detect text
        text_results = recognize_handwriting(image)
        
        # 3. Associate text with shapes
        for shape in shapes:
            shape_center_x, shape_center_y = shape['center']
            min_dist = float('inf')
            associated_text = ""
            
            for text_bbox, text, _ in text_results:
                # Calculate center of the text bounding box
                pts = np.array(text_bbox, dtype=np.int32)
                text_center_x = int(np.mean(pts[:, 0]))
                text_center_y = int(np.mean(pts[:, 1]))
                
                # Check distance between shape center and text center
                dist = np.sqrt((shape_center_x - text_center_x)**2 + (shape_center_y - text_center_y)**2)
                
                # Associate the closest text block within a reasonable distance
                max_association_distance = max(shape['bbox'][2], shape['bbox'][3])
                
                if dist < min_dist and dist < max_association_distance:
                    # Check if text center is inside the shape's bounding box
                    sx, sy, sw, sh = shape['bbox']
                    if sx <= text_center_x <= sx + sw and sy <= text_center_y <= sy + sh:
                        min_dist = dist
                        associated_text = text
            
            shape['text'] = associated_text.strip()
        
        # 4. Detect arrows/connections (simplified)
        # This is just a placeholder - comprehensive arrow detection would require more complex analysis
        # For now, we'll just use vertical position to infer connections
        sorted_shapes = sorted(shapes, key=lambda s: s['center'][1])  # Sort by y-coordinate
        
        # Tag shapes with their position in the flowchart
        for i, shape in enumerate(sorted_shapes):
            if i == 0:
                shape['position'] = 'start'
            elif i == len(sorted_shapes) - 1:
                shape['position'] = 'end'
            else:
                shape['position'] = 'middle'
        
        # 5. Classify flowchart elements based on shape and position
        for shape in sorted_shapes:
            shape_type = shape['shape']
            if shape_type == 'circle' or shape_type == 'oval':
                if shape['position'] == 'start' or shape['position'] == 'end':
                    shape['flowchart_type'] = 'terminal'
                else:
                    shape['flowchart_type'] = 'connector'
            elif shape_type == 'diamond':
                shape['flowchart_type'] = 'decision'
            elif shape_type == 'rectangle' or shape_type == 'square':
                shape['flowchart_type'] = 'process'
            elif shape_type == 'parallelogram':
                shape['flowchart_type'] = 'input_output'
            else:
                shape['flowchart_type'] = 'unknown'
        
        logging.info(f"Detected {len(sorted_shapes)} flowchart elements")
        return sorted_shapes
    except Exception as e:
        logging.error(f"Error detecting flowchart elements: {e}")
        return []

# --- Code Generation ---

def generate_code_from_flowchart(elements: List[Dict], language: str = 'python', use_gemini: bool = True) -> str:
    """
    Generates code from detected flowchart elements.
    
    Args:
        elements: A list of dictionaries representing flowchart elements
        language: Target programming language ('python', 'javascript', 'java', etc.)
        use_gemini: Whether to use Gemini for code generation
        
    Returns:
        A string containing the generated code
    """
    if not elements:
        return f"// No flowchart elements detected to generate {language} code."
    
    # Use Gemini for more intelligent code generation if available
    if use_gemini and gemini_model:
        try:
            logging.info(f"Using Gemini for {language} code generation from flowchart...")
            
            # Create structured representation of the flowchart
            flowchart_text = []
            for i, element in enumerate(elements):
                shape_type = element.get('flowchart_type', element.get('shape', 'unknown'))
                text = element.get('text', f'Step {i+1}')
                position = element.get('position', 'middle')
                
                flowchart_text.append(f"Element {i+1}: {shape_type} - '{text}' - Position: {position}")
            
            # For connections, we'll use position as a simple proxy
            # In a real system, you'd have actual connection data
            connections = []
            for i in range(len(elements) - 1):
                connections.append(f"Connection: Element {i+1} -> Element {i+2}")
            
            # Create prompt for Gemini
            prompt = f"""
            Generate {language} code from the following flowchart:
            
            Flowchart Elements:
            {'\n'.join(flowchart_text)}
            
            Connections:
            {'\n'.join(connections)}
            
            Requirements:
            1. Generate executable, well-structured {language} code that follows best practices
            2. Include comments explaining the logic and flow
            3. Handle decision points (diamonds) with proper conditional statements
            4. Include appropriate error handling
            5. Structure the code with proper functions or classes if needed
            
            Return only the code with no explanations before or after.
            """
            
            gemini_response = gemini_model.generate_content(prompt)
            generated_code = gemini_response.text.strip()
            
            # Strip code block markers if present
            code_match = re.search(r'```(?:\w+)?\n(.*?)\n```', generated_code, re.DOTALL)
            if code_match:
                generated_code = code_match.group(1)
            
            logging.info(f"Generated {language} code using Gemini")
            return generated_code
        except Exception as e:
            logging.error(f"Error using Gemini for code generation: {e}")
            # Fall back to basic generation
    
    # Basic generation if Gemini not available or failed
    logging.info(f"Using basic method for {language} code generation")
    
    # Sort elements by vertical position (top to bottom)
    elements.sort(key=lambda el: el['center'][1])
    
    # Generate appropriate code based on language
    if language.lower() == 'python':
        code = generate_python_code(elements)
    elif language.lower() == 'javascript':
        code = generate_javascript_code(elements)
    elif language.lower() in ['java', 'c++', 'c#']:
        code = generate_



# --- Code Generation (continued from where it cut off) ---

def generate_python_code(elements: List[Dict]) -> str:
    """Generates Python code from flowchart elements."""
    code_lines = ["# Python code generated from flowchart", ""]
    
    # Track decision blocks and their content
    decision_stack = []
    indentation = 0
    indent = "    "  # 4 spaces
    
    def add_line(line):
        code_lines.append(indent * indentation + line)
    
    # Add main function definition
    add_line("def main():")
    indentation += 1
    
    # Process each flowchart element
    for i, element in enumerate(elements):
        shape_type = element.get('flowchart_type', element.get('shape', 'unknown'))
        text = element.get('text', '').strip()
        
        if not text:
            text = f"step_{i+1}"
        
        # Normalize text for variable names
        var_text = text.lower().replace(' ', '_').replace('-', '_')
        var_text = re.sub(r'[^\w]', '', var_text)
        
        if shape_type == 'terminal':
            if i == 0:  # Start terminal
                add_line(f"# Start: {text}")
                add_line("print('Starting process...')")
            else:  # End terminal
                add_line(f"# End: {text}")
                add_line("print('Process completed.')")
                add_line("return")
                
        elif shape_type == 'process':
            add_line(f"# Process: {text}")
            add_line(f"# TODO: Implement {var_text} logic")
            add_line(f"{var_text}_result = process_{var_text}()")
            
        elif shape_type == 'input_output':
            if 'input' in text.lower():
                add_line(f"# Input: {text}")
                add_line(f"{var_text} = input('Enter {text}: ')")
            else:
                add_line(f"# Output: {text}")
                add_line(f"print({var_text})")
                
        elif shape_type == 'decision':
            add_line(f"# Decision: {text}")
            add_line(f"if check_condition('{text}'):")
            decision_stack.append(indentation)
            indentation += 1
            
        elif shape_type == 'connector':
            add_line(f"# Connector: {text}")
            add_line("# This is a connector point in the flowchart")
            
            # Check if we need to close any decision blocks
            if decision_stack and i < len(elements) - 1:
                next_element = elements[i + 1]
                next_y = next_element['center'][1]
                if next_y > element['center'][1] + 20:  # Simple heuristic
                    indentation = decision_stack.pop()
                    add_line("else:")
                    indentation += 1
    
    # Close any remaining decision blocks
    while decision_stack:
        indentation = decision_stack.pop()
    
    # Decrease indentation back to main level
    indentation = 1
    
    # Add helper functions
    add_line("")
    add_line("def check_condition(condition):")
    add_line("    # TODO: Implement actual condition checking")
    add_line("    print(f'Checking condition: {condition}')")
    add_line("    return True  # Default to True for demonstration")
    
    # Add other necessary helper functions based on processes found
    for element in elements:
        shape_type = element.get('flowchart_type', element.get('shape', 'unknown'))
        if shape_type == 'process':
            text = element.get('text', '').strip()
            if text:
                var_text = text.lower().replace(' ', '_').replace('-', '_')
                var_text = re.sub(r'[^\w]', '', var_text)
                add_line("")
                add_line(f"def process_{var_text}():")
                add_line(f"    # TODO: Implement {text} process")
                add_line("    return True")
    
    # Add execution guard
    code_lines.extend([
        "",
        "# Execute the main function when script is run directly",
        "if __name__ == '__main__':",
        "    main()"
    ])
    
    return "\n".join(code_lines)

def generate_javascript_code(elements: List[Dict]) -> str:
    """Generates JavaScript code from flowchart elements."""
    code_lines = ["// JavaScript code generated from flowchart", ""]
    
    # Track decision blocks and their content
    decision_stack = []
    indentation = 0
    indent = "  "  # 2 spaces (JS convention)
    
    def add_line(line):
        code_lines.append(indent * indentation + line)
    
    # Add main function definition
    add_line("function main() {")
    indentation += 1
    
    # Process each flowchart element
    for i, element in enumerate(elements):
        shape_type = element.get('flowchart_type', element.get('shape', 'unknown'))
        text = element.get('text', '').strip()
        
        if not text:
            text = f"step{i+1}"
        
        # Normalize text for variable names
        var_text = text.toLowerCase().replace(/ /g, '_').replace(/-/g, '_')
        var_text = var_text.replace(/[^\w]/g, '')
        
        if shape_type == 'terminal':
            if i == 0:  # Start terminal
                add_line(`// Start: ${text}`)
                add_line("console.log('Starting process...');")
            else:  # End terminal
                add_line(`// End: ${text}`)
                add_line("console.log('Process completed.');")
                add_line("return;")
                
        elif shape_type == 'process':
            add_line(`// Process: ${text}`)
            add_line(`// TODO: Implement ${var_text} logic`)
            add_line(`const ${var_text}Result = process${var_text.charAt(0).toUpperCase() + var_text.slice(1)}();`)
            
        elif shape_type == 'input_output':
            if text.toLowerCase().includes('input'):
                add_line(`// Input: ${text}`)
                add_line(`const ${var_text} = prompt('Enter ${text}:');`)
            else:
                add_line(`// Output: ${text}`)
                add_line(`console.log(${var_text});`)
                
        elif shape_type == 'decision':
            add_line(`// Decision: ${text}`)
            add_line(`if (checkCondition('${text}')) {`)
            decision_stack.append(indentation)
            indentation += 1
            
        elif shape_type == 'connector':
            add_line(`// Connector: ${text}`)
            add_line("// This is a connector point in the flowchart")
            
            # Check if we need to close any decision blocks
            if decision_stack.length > 0 && i < elements.length - 1:
                next_element = elements[i + 1]
                next_y = next_element['center'][1]
                if next_y > element['center'][1] + 20:  # Simple heuristic
                    indentation = decision_stack.pop()
                    add_line("} else {")
                    indentation += 1
    
    # Close any remaining decision blocks
    while decision_stack.length > 0:
        indentation = decision_stack.pop()
        add_line("}")
    
    # Decrease indentation back to main level
    indentation = 1
    add_line("}")
    
    # Add helper functions
    add_line("")
    add_line("function checkCondition(condition) {")
    add_line("  // TODO: Implement actual condition checking")
    add_line("  console.log(`Checking condition: ${condition}`);")
    add_line("  return true;  // Default to true for demonstration")
    add_line("}")
    
    # Add other necessary helper functions based on processes found
    for element in elements:
        shape_type = element.get('flowchart_type', element.get('shape', 'unknown'))
        if shape_type == 'process':
            text = element.get('text', '').strip()
            if text:
                var_text = text.toLowerCase().replace(/ /g, '_').replace(/-/g, '_')
                var_text = var_text.replace(/[^\w]/g, '')
                func_name = var_text.charAt(0).toUpperCase() + var_text.slice(1)
                add_line("")
                add_line(`function process${func_name}() {`)
                add_line(`  // TODO: Implement ${text} process`)
                add_line("  return true;")
                add_line("}")
    
    # Add execution line
    code_lines.push("", "// Execute the main function", "main();")
    
    return code_lines.join("\n")

def generate_java_code(elements: List[Dict]) -> str:
    """Generates Java code from flowchart elements."""
    code_lines = ["// Java code generated from flowchart", ""]
    
    # Add class definition
    class_name = "FlowchartImplementation"
    code_lines.append(f"public class {class_name} {{")
    
    # Track decision blocks and their content
    decision_stack = []
    indentation = 1
    indent = "    "  # 4 spaces
    
    def add_line(line):
        code_lines.append(indent * indentation + line)
    
    # Add main method definition
    add_line("public static void main(String[] args) {")
    add_line("    new FlowchartImplementation().execute();")
    add_line("}")
    
    # Add execute method
    add_line("")
    add_line("public void execute() {")
    indentation += 1
    
    # Process each flowchart element
    for i, element in enumerate(elements):
        shape_type = element.get('flowchart_type', element.get('shape', 'unknown'))
        text = element.get('text', '').strip()
        
        if not text:
            text = f"step{i+1}"
        
        # Normalize text for variable names
        var_text = text.lower().replace(' ', '_').replace('-', '_')
        var_text = re.sub(r'[^\w]', '', var_text)
        
        if shape_type == 'terminal':
            if i == 0:  # Start terminal
                add_line(f"// Start: {text}")
                add_line("System.out.println(\"Starting process...\");")
            else:  # End terminal
                add_line(f"// End: {text}")
                add_line("System.out.println(\"Process completed.\");")
                add_line("return;")
                
        elif shape_type == 'process':
            add_line(f"// Process: {text}")
            add_line(f"// TODO: Implement {var_text} logic")
            add_line(f"boolean {var_text}Result = process{var_text.capitalize()}();")
            
        elif shape_type == 'input_output':
            if 'input' in text.lower():
                add_line(f"// Input: {text}")
                add_line("Scanner scanner = new Scanner(System.in);")
                add_line(f"System.out.print(\"Enter {text}: \");")
                add_line(f"String {var_text} = scanner.nextLine();")
            else:
                add_line(f"// Output: {text}")
                add_line(f"System.out.println({var_text});")
                
        elif shape_type == 'decision':
            add_line(f"// Decision: {text}")
            add_line(f"if (checkCondition(\"{text}\")) {{")
            decision_stack.append(indentation)
            indentation += 1
            
        elif shape_type == 'connector':
            add_line(f"// Connector: {text}")
            add_line("// This is a connector point in the flowchart")
            
            # Check if we need to close any decision blocks
            if decision_stack and i < len(elements) - 1:
                next_element = elements[i + 1]
                next_y = next_element['center'][1]
                if next_y > element['center'][1] + 20:  # Simple heuristic
                    indentation = decision_stack.pop()
                    add_line("} else {")
                    indentation += 1
    
    # Close any remaining decision blocks
    while decision_stack:
        indentation = decision_stack.pop()
        add_line("}")
    
    # Decrease indentation back to execute level
    indentation = 2
    add_line("}")
    
    # Add helper methods
    indentation = 1
    add_line("")
    add_line("private boolean checkCondition(String condition) {")
    add_line("    // TODO: Implement actual condition checking")
    add_line("    System.out.println(\"Checking condition: \" + condition);")
    add_line("    return true;  // Default to true for demonstration")
    add_line("}")
    
    # Add other necessary helper methods based on processes found
    for element in elements:
        shape_type = element.get('flowchart_type', element.get('shape', 'unknown'))
        if shape_type == 'process':
            text = element.get('text', '').strip()
            if text:
                var_text = text.lower().replace(' ', '_').replace('-', '_')
                var_text = re.sub(r'[^\w]', '', var_text)
                method_name = var_text.capitalize()
                add_line("")
                add_line(f"private boolean process{method_name}() {{")
                add_line(f"    // TODO: Implement {text} process")
                add_line("    return true;")
                add_line("}")
    
    # Close class
    code_lines.append("}")
    
    return "\n".join(code_lines)

# --- Content Summarization ---

def summarize_notes(text: str, use_gemini: bool = True) -> str:
    """
    Summarizes handwritten or typed notes.
    
    Args:
        text: The text to summarize
        use_gemini: Whether to use Gemini for better summarization
        
    Returns:
        A summarized version of the text
    """
    if not text or len(text.strip()) == 0:
        return "Error: No text provided for summarization."
    
    # Use Gemini for intelligent summarization if available
    if use_gemini and gemini_model:
        try:
            logging.info("Using Gemini for note summarization...")
            
            # Create prompt for Gemini
            prompt = f"""
            Summarize the following notes. Extract key points, main ideas, and important details.
            Keep the summary concise while maintaining the core information and meaning:

            {text}
            """
            
            gemini_response = gemini_model.generate_content(prompt)
            summary = gemini_response.text.strip()
            
            logging.info("Generated note summary with Gemini")
            return summary
        except Exception as e:
            logging.error(f"Error using Gemini for summarization: {e}")
            # Fall back to basic summarization
    
    # Basic summarization if Gemini not available
    logging.info("Using basic summarization method")
    
    try:
        # Simple extractive summarization
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) <= 3:
            return text  # Return original text if it's already short
        
        # Simple sentence scoring based on position and common important phrases
        scored_sentences = []
        important_words = ['key', 'important', 'significant', 'main', 'critical', 'essential', 'note', 'remember']
        
        for i, sentence in enumerate(sentences):
            # Score based on position (first and last sentences often have important info)
            position_score = 1.0 if i == 0 or i == len(sentences) - 1 else 0.0
            
            # Score based on presence of important words
            word_score = sum(1.0 for word in important_words if word.lower() in sentence.lower()) / len(important_words)
            
            # Score based on sentence length (prefer medium-length sentences)
            length = len(sentence.split())
            length_score = 1.0 if 5 <= length <= 20 else 0.5
            
            # Combine scores
            total_score = position_score * 0.4 + word_score * 0.4 + length_score * 0.2
            scored_sentences.append((sentence, total_score))
        
        # Sort sentences by score in descending order
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 30% of sentences or at least 3 sentences
        num_sentences = max(3, int(len(sentences) * 0.3))
        top_sentences = [s[0] for s in scored_sentences[:num_sentences]]
        
        # Reorder sentences to maintain original flow
        original_order = []
        for sentence in sentences:
            if sentence in [s for s in top_sentences]:
                original_order.append(sentence)
        
        summary = ' '.join(original_order)
        return summary
    except Exception as e:
        logging.error(f"Error during basic summarization: {e}")
        return f"Error: Failed to summarize text: {e}"

# --- Combined Processing & Main Class ---

class CanvasProcessor:
    """
    Main processor class for Cognito Canvas.
    
    Handles all the processing operations for images captured from the canvas,
    including handwriting recognition, math solving, flowchart analysis, and note summarization.
    """
    
    def __init__(self, use_gemini: bool = True):
        """
        Initializes the canvas processor.
        
        Args:
            use_gemini: Whether to use Google's Gemini model for enhanced processing
        """
        self.use_gemini = use_gemini and gemini_model is not None
        logging.info(f"CanvasProcessor initialized. Gemini enabled: {self.use_gemini}")
    
    def process_math(self, image: np.ndarray) -> str:
        """
        Processes an image of a mathematical expression.
        Performs handwriting recognition followed by expression evaluation.
        
        Args:
            image: Image containing a mathematical expression (numpy array, BGR format)
            
        Returns:
            A string with the recognized expression and its solution
        """
        logging.info("Processing mathematical expression...")
        
        if image is None or image.size == 0:
            return "Error: Invalid input image."
        
        try:
            # 1. Preprocess the image for better OCR
            processed_image = process_image_for_ocr(image)
            
            # 2. Recognize the handwritten math expression
            ocr_results = recognize_handwriting(processed_image, use_gemini=self.use_gemini)
            
            if not ocr_results:
                return "Error: Could not recognize any text in the image. Please ensure the handwriting is clear."
            
            # 3. Combine all recognized text (for math expressions, they should be on a single line)
            math_expression = ' '.join([text for _, text, _ in ocr_results])
            
            logging.info(f"Recognized expression: {math_expression}")
            
            # 4. Solve the mathematical expression
            solution = solve_mathematical_expression(math_expression, use_gemini=self.use_gemini)
            
            return f"Expression: {math_expression}\nSolution: {solution}"
        except Exception as e:
            logging.error(f"Error processing math expression: {e}")
            return f"Error: Failed to process mathematical expression: {e}"
    
    def process_flowchart(self, image: np.ndarray) -> str:
        """
        Processes an image of a flowchart.
        Detects flowchart elements and generates corresponding code.
        
        Args:
            image: Image containing a flowchart (numpy array, BGR format)
            
        Returns:
            Generated code as a string
        """
        logging.info("Processing flowchart...")
        
        if image is None or image.size == 0:
            return "Error: Invalid input image."
        
        try:
            # 1. Detect flowchart elements
            elements = detect_flowchart_elements(image, use_gemini=self.use_gemini)
            
            if not elements:
                return "Error: Could not detect any flowchart elements. Please ensure the drawing is clear."
            
            # 2. Generate Python code from the flowchart
            code = generate_code_from_flowchart(elements, language='python', use_gemini=self.use_gemini)
            
            return f"Detected {len(elements)} flowchart elements.\nGenerated Code:\n\n{code}"
        except Exception as e:
            logging.error(f"Error processing flowchart: {e}")
            return f"Error: Failed to process flowchart: {e}"
    
    def process_notes(self, image: np.ndarray) -> str:
        """
        Processes an image of handwritten notes.
        Performs handwriting recognition and summarizes the content.
        
        Args:
            image: Image containing handwritten notes (numpy array, BGR format)
            
        Returns:
            Original recognized text and its summary
        """
        logging.info("Processing handwritten notes...")
        
        if image is None or image.size == 0:
            return "Error: Invalid input image."
        
        try:
            # 1. Preprocess the image for better OCR
            processed_image = process_image_for_ocr(image)
            
            # 2. Recognize the handwritten text
            ocr_results = recognize_handwriting(processed_image, use_gemini=self.use_gemini)
            
            if not ocr_results:
                return "Error: Could not recognize any text in the image. Please ensure the handwriting is clear."
            
            # 3. Combine all recognized text, preserving paragraph breaks
            lines = []
            prev_y = None
            line_texts = []
            
            # Sort by vertical position (y-coordinate)
            sorted_results = sorted(ocr_results, key=lambda x: x[0][0][1])  # Sort by top y-coordinate
            
            for bbox, text, _ in sorted_results:
                # Calculate average y-coordinate of the bounding box
                y_coords = [pt[1] for pt in bbox]
                avg_y = sum(y_coords) / len(y_coords)
                
                # Check if this is a new line
                if prev_y is None or abs(avg_y - prev_y) > 20:  # Threshold for new line
                    if line_texts:
                        lines.append(' '.join(line_texts))
                    line_texts = [text]
                else:
                    line_texts.append(text)
                
                prev_y = avg_y
            
            # Add the last line
            if line_texts:
                lines.append(' '.join(line_texts))
            
            full_text = '\n'.join(lines)
            
            logging.info(f"Recognized text ({len(full_text)} chars)")
            
            # 4. Summarize the recognized text
            if len(full_text) > 100:  # Only summarize if there's enough text
                summary = summarize_notes(full_text, use_gemini=self.use_gemini)
                return f"Original Text:\n\n{full_text}\n\nSummary:\n\n{summary}"
            else:
                return f"Recognized Text:\n\n{full_text}\n\n(Text too short to summarize)"
        except Exception as e:
            logging.error(f"Error processing notes: {e}")
            return f"Error: Failed to process handwritten notes: {e}"

# --- Main execution (for testing) ---

if __name__ == "__main__":
    # This section allows testing the processor functions directly
    logging.info("Testing the processor module...")
    
    # Check Gemini availability
    if gemini_model:
        logging.info("Gemini model is available")
    else:
        logging.info("Gemini model is not available, using fallback methods")
    
    # Example test function
    def test_with_image(image_path, process_func):
        if not os.path.exists(image_path):
            logging.error(f"Test image not found: {image_path}")
            return None
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Failed to load image: {image_path}")
            return None
        
        # Process image
        result = process_func(image)
        logging.info(f"Result: {result[:100]}...")  # Log first 100 chars
        return result
    
    # Define test images (user would need to create these)
    test_dir = "test_images"
    if os.path.exists(test_dir):
        processor = CanvasProcessor()
        
        # Test math processing
        math_image = os.path.join(test_dir, "math_example.jpg")
        if os.path.exists(math_image):
            test_with_image(math_image, processor.process_math)
        
        # Test flowchart processing
        flowchart_image = os.path.join(test_dir, "flowchart_example.jpg")
        if os.path.exists(flowchart_image):
            test_with_image(flowchart_image, processor.process_flowchart)
        
        # Test note processing
        notes_image = os.path.join(test_dir, "notes_example.jpg")
        if os.path.exists(notes_image):
            test_with_image(notes_image, processor.process_notes)
    else:
        logging.info(f"Test directory not found: {test_dir}")
        logging.info("Skipping image tests")
