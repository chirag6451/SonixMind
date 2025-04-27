import streamlit as st
import os
import sys

# Import branding
from app_branding import setup_page_config, apply_custom_css, display_logo
from app_workflow import show_workflow

def main():
    """Main entry point for the SonixMind application"""
    
    # Setup page configuration
    setup_page_config()
    
    # Apply custom CSS
    apply_custom_css()
    
    # Display logo if exists
    if os.path.exists("logo.png"):
        display_logo()
    
    # Always use workflow interface
    show_workflow()

if __name__ == "__main__":
    # Create assets directory if it doesn't exist
    os.makedirs("assets", exist_ok=True)
    
    # Generate logo if it doesn't exist
    if not os.path.exists(os.path.join("assets", "logo.png")):
        try:
            from assets.create_logo import create_logo
            create_logo(os.path.join("assets", "logo.png"))
        except Exception as e:
            st.warning(f"Failed to create logo: {str(e)}")
    
    # Run the application
    main() 