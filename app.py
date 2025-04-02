"""
Style Finder Gradio interface.
"""

import gradio as gr
import pandas as pd
import os
from tempfile import NamedTemporaryFile

# Import local modules
from models.image_processor import ImageProcessor
from models.llm_service import LlamaVisionService
from utils.helpers import get_all_items_for_image, format_alternatives_response, process_response
import config

def process_image(self, image):
    """
    Process a user-uploaded image and generate a fashion response.
    
    Args:
        image: PIL image uploaded through Gradio
            
    Returns:
        str: Formatted response with fashion analysis
    """
    # Save the image temporarily if it's not already a file path
    if not isinstance(image, str):
        temp_file = NamedTemporaryFile(delete=False, suffix=".jpg")
        image_path = temp_file.name
        image.save(image_path)
    else:
        image_path = image
    
    # Step 1: Encode the image
    user_encoding = self.image_processor.encode_image(image_path, is_url=False)
    if user_encoding['vector'] is None:
        return "Error: Unable to process the image. Please try another image."
    
    # Step 2: Find the closest match
    closest_row, similarity_score = self.image_processor.find_closest_match(user_encoding['vector'], self.data)
    if closest_row is None:
        return "Error: Unable to find a match. Please try another image."
    
    print(f"Closest match: {closest_row['Item Name']} with similarity score {similarity_score:.2f}")
    
    # Step 3: Get all related items
    all_items = get_all_items_for_image(closest_row['Image URL'], self.data)
    if all_items.empty:
        return "Error: No items found for the matched image."
    
    # Step 4: Generate fashion response
    bot_response = self.llm_service.generate_fashion_response(
        user_image_base64=user_encoding['base64'],
        matched_row=closest_row,
        all_items=all_items,
        similarity_score=similarity_score,
        threshold=config.SIMILARITY_THRESHOLD
    )
    
    # Clean up temporary file
    if not isinstance(image, str):
        try:
            os.unlink(image_path)
        except:
            pass
    
    return process_response(bot_response)

"""
And here's the updated create_gradio_interface function:
"""

def create_gradio_interface(app):
    """
    Create and configure the Gradio interface.
    
    Args:
        app (StyleFinderApp): Instance of the StyleFinderApp
        
    Returns:
        gr.Blocks: Configured Gradio interface
    """
    with gr.Blocks(theme=gr.themes.Soft(), title="Fashion Style Analyzer") as demo:
        # Introduction
        gr.Markdown(
            """
            # Fashion Style Analyzer
            
            Upload an image to analyze fashion elements and get detailed information about the items.
            This application combines computer vision, vector similarity, and large language models 
            to provide detailed fashion analysis.
            """
        )
        
        # Example images section - moved higher up
        gr.Markdown("### Example Images")
        with gr.Row():
            # Display the images directly
            gr.Image(value="examples/test-1.png", label="Example 1", show_label=True, scale=1)
            gr.Image(value="examples/test-2.png", label="Example 2", show_label=True, scale=1)
            gr.Image(value="examples/test-3.png", label="Example 3", show_label=True, scale=1)
        
        # Example image buttons
        with gr.Row():
            example1_btn = gr.Button("Use Example 1")
            example2_btn = gr.Button("Use Example 2")
            example3_btn = gr.Button("Use Example 3")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Image input component
                image_input = gr.Image(
                    type="pil", 
                    label="Upload Fashion Image"
                )
                
                # Submit button
                submit_btn = gr.Button("Analyze Style", variant="primary")
                
                # Status indicator
                status = gr.Markdown("Ready to analyze.")
            
            with gr.Column(scale=2):
                # Output markdown component for displaying analysis results
                output = gr.Markdown(
                    label="Style Analysis Results",
                    height=700
                )
        
        # Event handlers
        # 1. Submit button click with processing indicator
        submit_btn.click(
            fn=lambda: "Analyzing image... This may take a few moments.",
            inputs=None,
            outputs=status
        ).then(
            fn=app.process_image,
            inputs=[image_input],
            outputs=output
        ).then(
            fn=lambda: "Analysis complete!",
            inputs=None,
            outputs=status
        )
        
        # 2. Example image buttons
        example1_btn.click(
            fn=lambda: "examples/test-1.png", 
            inputs=None, 
            outputs=image_input
        ).then(
            fn=lambda: "Example 1 loaded. Click 'Analyze Style' to process.",
            inputs=None,
            outputs=status
        )
        
        example2_btn.click(
            fn=lambda: "examples/test-2.png", 
            inputs=None, 
            outputs=image_input
        ).then(
            fn=lambda: "Example 2 loaded. Click 'Analyze Style' to process.",
            inputs=None,
            outputs=status
        )
        
        example3_btn.click(
            fn=lambda: "examples/test-3.png", 
            inputs=None, 
            outputs=image_input
        ).then(
            fn=lambda: "Example 3 loaded. Click 'Analyze Style' to process.",
            inputs=None,
            outputs=status
        )
        
        # Information about the application
        gr.Markdown(
            """
            ### About This Application
            
            This system analyzes fashion images using:
            
            - **Image Encoding**: Converting fashion images into numerical vectors
            - **Similarity Matching**: Finding visually similar items in a database
            - **Advanced AI**: Generating detailed descriptions of fashion elements
            
            The analyzer identifies garments, fabrics, colors, and styling details from images.
            The database includes information on outfits with brand and pricing details.
            """
        )
    
    return demo