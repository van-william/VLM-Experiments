"""
Interface for the Google Gemini multimodal model.
Uses the actual Google GenAI library API.
"""

from PIL import Image
import logging

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logging.warning("Google GenAI library not found. Install with: pip install google-genai")


class GeminiInterface:
    """Interface for the Google Gemini multimodal model."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize the Gemini client.
        
        Args:
            api_key (str): Google API key for Gemini
            model_name (str): Gemini model to use
        """
        if not GENAI_AVAILABLE:
            raise ImportError(
                "Google GenAI library not installed. Install with: pip install google-genai"
            )
        
        self.api_key = api_key
        self.model_name = model_name
        
        try:
            # Configure the client with API key
            self.client = genai.Client(api_key=api_key)
            print(f"Gemini client initialized with model: {model_name}")
        except Exception as e:
            logging.error(f"Failed to initialize Gemini client: {e}")
            raise
    
    def answer_question(self, image: Image.Image, question: str) -> str:
        """
        Answer a question about an image using Gemini.
        
        Args:
            image (PIL.Image): Input image
            question (str): Question to ask about the image
            
        Returns:
            str: Model's answer to the question
        """
        try:
            # Convert PIL image to bytes
            import io
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Generate response using the correct API
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(
                        data=img_byte_arr,
                        mime_type='image/jpeg',
                    ),
                    question
                ]
            )
            
            return response.text if response.text else "No response generated from Gemini."
                
        except Exception as e:
            logging.error(f"Error during Gemini inference: {e}")
            return f"Error processing image: {str(e)}"
    
    def __repr__(self):
        return f"GeminiInterface(model_name='{self.model_name}')" 