from PIL import Image, ImageDraw, ImageFont
import os

def create_fraudguard_logo():
    """Create FraudGuard AI logo"""
    # Create image
    width, height = 400, 400
    img = Image.new('RGB', (width, height), color='#0e1117')
    draw = ImageDraw.Draw(img)
    
    # Draw shield
    shield_points = [
        (200, 50),   # top
        (320, 120),  # right top
        (320, 250),  # right bottom
        (200, 350),  # bottom point
        (80, 250),   # left bottom
        (80, 120)    # left top
    ]
    draw.polygon(shield_points, fill='#ff4b4b', outline='#ffffff', width=3)
    
    # Draw checkmark
    check_points = [
        (140, 200),
        (180, 240),
        (260, 140)
    ]
    draw.line(check_points, fill='#ffffff', width=15, joint='curve')
    
    # Save
    logo_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'fraudguard_logo.png')
    img.save(logo_path)
    print(f"Logo saved to {logo_path}")
    return logo_path

if __name__ == "__main__":
    create_fraudguard_logo()
