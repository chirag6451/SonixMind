import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_logo(save_path='logo.png', width=600, height=200):
    """
    Generate a simple logo for SonixMind application
    
    Args:
        save_path: Path to save the generated logo
        width: Width of the logo
        height: Height of the logo
    """
    # Create a blank image with a gradient background
    image = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    
    # Create a gradient background
    for y in range(height):
        # Create a blue gradient from dark to light
        r = int(25 + (y / height) * 20)
        g = int(70 + (y / height) * 40)
        b = int(150 + (y / height) * 60)
        draw.line([(0, y), (width, y)], fill=(r, g, b, 255))
    
    # Create a sound wave pattern
    wave_color = (255, 255, 255, 200)
    wave_height = height * 0.25
    wave_center = height * 0.6
    
    # Draw multiple sound waves
    for x in range(0, width, 3):
        # Main waveform
        amplitude = 30 * np.sin(x * 0.05) + 15 * np.sin(x * 0.1)
        y1 = wave_center - amplitude
        y2 = wave_center + amplitude
        draw.line([(x, y1), (x, y2)], fill=wave_color, width=2)
    
    # Try to load a font, use default if not available
    try:
        font_size = int(height * 0.3)
        font = ImageFont.truetype("Arial Bold.ttf", font_size)
    except IOError:
        try:
            # Try system font on macOS
            font = ImageFont.truetype("/Library/Fonts/Arial Bold.ttf", font_size)
        except IOError:
            try:
                # Try system font on Windows
                font = ImageFont.truetype("arialbd.ttf", font_size)
            except IOError:
                # Use default font
                font = ImageFont.load_default()
    
    # Add app name
    app_name = "SonixMind"
    
    # Get text dimensions - compatible with newer PIL versions
    if hasattr(draw, 'textsize'):
        # For older versions of PIL
        text_width, text_height = draw.textsize(app_name, font=font)
    else:
        # For newer versions of PIL
        left, top, right, bottom = font.getbbox(app_name)
        text_width = right - left
        text_height = bottom - top
    
    # Position text in the center
    text_position = ((width - text_width) // 2, height // 5)
    
    # Draw text with shadow for depth
    shadow_offset = 2
    draw.text((text_position[0] + shadow_offset, text_position[1] + shadow_offset), 
              app_name, font=font, fill=(30, 30, 30, 180))
    draw.text(text_position, app_name, font=font, fill=(255, 255, 255, 255))
    
    # Add a tagline
    try:
        tagline_font_size = int(height * 0.12)
        tagline_font = ImageFont.truetype("Arial.ttf", tagline_font_size)
    except IOError:
        tagline_font = ImageFont.load_default()
    
    tagline = "Professional Audio Processing"
    
    # Get tagline dimensions - compatible with newer PIL versions
    if hasattr(draw, 'textsize'):
        # For older versions of PIL
        tagline_width, tagline_height = draw.textsize(tagline, font=tagline_font)
    else:
        # For newer versions of PIL
        left, top, right, bottom = tagline_font.getbbox(tagline)
        tagline_width = right - left
        tagline_height = bottom - top
    
    tagline_position = ((width - tagline_width) // 2, text_position[1] + text_height + 15)
    
    draw.text(tagline_position, tagline, font=tagline_font, fill=(255, 255, 255, 230))
    
    # Save the image
    image.save(save_path)
    print(f"Logo created successfully and saved to {save_path}")
    return save_path

if __name__ == "__main__":
    # Create the directory if it doesn't exist
    os.makedirs('assets', exist_ok=True)
    
    # Create and save the logo
    logo_path = os.path.join('assets', 'logo.png')
    create_logo(logo_path) 