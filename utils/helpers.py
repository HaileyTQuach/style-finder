"""
Utility functions for the Style Finder application.
"""

import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_all_items_for_image(image_url, dataset):
    """
    Get all items related to a specific image from the dataset.
    
    Args:
        image_url (str): The URL of the matched image
        dataset (DataFrame): Dataset containing outfit information
        
    Returns:
        DataFrame: All items related to the image
    """
    related_items = dataset[dataset['Image URL'] == image_url]
    logger.info(f"Found {len(related_items)} items related to image URL: {image_url}")
    return related_items

def format_alternatives_response(user_response, alternatives, similarity_score, threshold=0.8):
    """
    Append alternatives to the user response in a formatted way.
    
    Args:
        user_response (str): Original response from the model
        alternatives (dict): Dictionary of alternatives for each item
        similarity_score (float): Similarity score of the match
        threshold (float): Threshold for determining match quality
        
    Returns:
        str: Enhanced response with alternatives
    """
    # Check if user_response is problematic
    if not user_response or any(phrase in user_response for phrase in [
            "I'm not able to provide", 
            "I cannot", 
            "I apologize, but", 
            "I don't feel comfortable"]):
        # Create a basic response if the model refused
        user_response = "## Fashion Analysis Results\n\nHere are the items detected in your image:"
    
    if similarity_score >= threshold:
        enhanced_response = user_response + "\n\n## Similar Items Found\n\nHere are some similar items we found:\n"
    else:
        enhanced_response = user_response + "\n\n## Similar Items Found\n\nHere are some visually similar items:\n"
    
    # Count items added to ensure we're not exceeding reasonable limits
    items_added = 0
    max_items = 10
    
    for item, alts in alternatives.items():
        enhanced_response += f"\n### {item}:\n"
        if alts:
            for alt in alts[:3]:  # Limit to 3 alternatives per item
                if items_added < max_items:
                    enhanced_response += f"- {alt['title']} for {alt['price']} from {alt['source']} ([Buy it here]({alt['link']}))\n"
                    items_added += 1
        else:
            enhanced_response += "- No alternatives found.\n"
    
    return enhanced_response

def process_response(response: str) -> str:
    """
    Process and escape problematic characters in the response.
    
    Args:
        response (str): The original response text
        
    Returns:
        str: Processed response with escaped characters and proper formatting
    """
    if not response:
        logger.warning("Empty response received")
        return "# Fashion Analysis\n\nNo detailed analysis was generated. Please refer to the item details below."
    
    # Check for rejection messages
    rejection_phrases = [
        "I'm not able to provide",
        "I cannot provide",
        "I apologize, but I cannot",
        "I don't feel comfortable",
        "violated our content policy"
    ]
    
    # If the model rejected but we still have item details, extract and format them
    if any(phrase in response for phrase in rejection_phrases):
        logger.warning("Model rejected the request, extracting item details")
        
        # Try to extract the item details section
        items_section = None
        
        if "ITEM DETAILS:" in response:
            # Extract everything after ITEM DETAILS:
            items_section = "## Item Details\n\n" + response.split("ITEM DETAILS:")[1].strip()
        elif "SIMILAR ITEMS:" in response:
            # Extract everything after SIMILAR ITEMS:
            items_section = "## Similar Items\n\n" + response.split("SIMILAR ITEMS:")[1].strip()
        
        if items_section:
            # Format item details with proper Markdown
            formatted_items = re.sub(r'^\* ', '- ', items_section, flags=re.MULTILINE)
            return "# Fashion Analysis\n\nHere are the items detected in your image:\n\n" + formatted_items
        else:
            # Return whatever we got with minimal processing
            return response.replace("$", "\\$")
    
    # Escape $ signs for Markdown
    processed = response.replace("$", "\\$")
    
    # Ensure important sections are properly formatted
    if "ITEM DETAILS:" in processed:
        processed = processed.replace("ITEM DETAILS:", "## Item Details")
    
    if "SIMILAR ITEMS:" in processed:
        processed = processed.replace("SIMILAR ITEMS:", "## Similar Items")
    
    # Add proper formatting to ensure aesthetic display
    if not processed.startswith("#"):
        processed = "# Fashion Analysis\n\n" + processed
    
    # Ensure all bullet points use consistent Markdown
    processed = re.sub(r'^\* ', '- ', processed, flags=re.MULTILINE)
    
    return processed