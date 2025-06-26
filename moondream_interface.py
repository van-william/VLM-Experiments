"""
Interface for the Moondream multimodal model.
Uses the current Moondream Cloud API.
"""

from PIL import Image
import logging

try:
    import moondream as md
    MOONDREAM_AVAILABLE = True
except ImportError:
    MOONDREAM_AVAILABLE = False
    logging.warning("Moondream library not found. Install with: pip install moondream")


class MoondreamInterface:
    """Interface for the Moondream multimodal model."""
    
    def __init__(self, api_key: str):
        """
        Initialize the Moondream model.
        
        Args:
            api_key (str): API key from console.moondream.ai
        """
        if not MOONDREAM_AVAILABLE:
            raise ImportError(
                "Moondream library not installed. Install with: pip install moondream"
            )
        
        try:
            self.model = md.vl(api_key=api_key)
            print("Moondream Cloud client initialized successfully!")
        except Exception as e:
            logging.error(f"Failed to initialize Moondream: {e}")
            raise
    
    def answer_question(self, image: Image.Image, question: str) -> str:
        """
        Answer a question about an image using Moondream.
        
        Args:
            image (PIL.Image): Input image
            question (str): Question to ask about the image
            
        Returns:
            str: Model's answer to the question
        """
        try:
            # Encode image for better performance (recommended by Moondream)
            encoded_image = self.model.encode_image(image)
            
            # Use the query method from the moondream API
            response = self.model.query(encoded_image, question)
            return response.get('answer', 'No answer generated.')
        except Exception as e:
            logging.error(f"Error during Moondream inference: {e}")
            return f"Error processing image: {str(e)}"
    
    def caption(self, image: Image.Image, length: str = "normal") -> str:
        """
        Generate a caption for an image.
        
        Args:
            image (PIL.Image): Input image
            length (str): Caption length ("short", "normal")
            
        Returns:
            str: Generated caption
        """
        try:
            # Encode image for better performance
            encoded_image = self.model.encode_image(image)
            
            response = self.model.caption(encoded_image, length=length)
            return response.get('caption', 'No caption generated.')
        except Exception as e:
            logging.error(f"Error during Moondream captioning: {e}")
            return f"Error generating caption: {str(e)}"
    
    def __repr__(self):
        return f"MoondreamInterface(using Moondream Cloud API)" 