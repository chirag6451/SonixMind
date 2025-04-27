import streamlit as st
import markdown
import base64

def display_logo():
    """Display the SonixMind logo"""
    st.image("logo.png", width=300)

def get_author_info():
    """Return author and company information"""
    return {
        "author": "Chirag Kansara",
        "role": "CTO",
        "linkedin": "https://www.linkedin.com/in/indapoint/",
        "email": "chirag@indapoint.com",
        "company": "IndaPoint Technologies Pvt. Ltd.",
        "website": "https://www.indapoint.com/",
        "address": "Mumbai, Maharashtra, India",
        "company_email": "info@indapoint.com",
        "copyright_year": "2023-2025"
    }

def get_app_info():
    """Return application information"""
    return {
        "name": "SonixMind",
        "version": "1.0.0",
        "description": "A powerful audio and video processing application that helps you extract, enhance, and master audio with professional-quality results."
    }

def get_ai_models_info():
    """Return information about AI models used in the application"""
    return [
        {
            "name": "VoiceFixer",
            "purpose": "Speech restoration and enhancement",
            "capabilities": "Restores degraded speech regardless of severity, handling noise, reverberation, and low resolution",
            "paper": "https://arxiv.org/abs/2109.13731",
            "architecture": "Neural vocoder-based restoration",
            "author": "Haohe Liu et al."
        },
        {
            "name": "Matchering",
            "purpose": "Audio mastering to match professional reference tracks",
            "capabilities": "Analyzes target and reference audio to match spectral balance, loudness, and dynamics",
            "github": "https://github.com/sergree/matchering",
            "technology": "Advanced DSP algorithms and psychoacoustic models"
        },
        {
            "name": "Streamlit ML Components",
            "purpose": "Interactive visualization and processing",
            "capabilities": "Real-time audio processing with visual feedback",
            "technology": "Streamlit's machine learning-optimized components"
        },
        {
            "name": "Custom Neural Processing",
            "purpose": "Audio enhancement and noise reduction",
            "capabilities": "Custom-tuned algorithms for audio clarity enhancement",
            "implementation": "Based on spectral processing techniques with neural optimizations"
        }
    ]

def show_about_section():
    """Display an About section with app, author, and AI model information"""
    app_info = get_app_info()
    author_info = get_author_info()
    
    st.title(f"About {app_info['name']}")
    st.markdown(f"""
    ## {app_info['name']} v{app_info['version']}
    {app_info['description']}
    
    ## Author
    
    **{author_info['author']}** - {author_info['role']}  
    [LinkedIn]({author_info['linkedin']}) | Email: {author_info['email']}
    
    ## Company
    
    **{author_info['company']}**  
    Website: [{author_info['website'].replace('https://', '')}]({author_info['website']})  
    Address: {author_info['address']}  
    Email: {author_info['company_email']}
    
    © {author_info['copyright_year']} {author_info['company']}. All rights reserved.
    """)
    
    st.markdown("---")
    
    st.subheader("AI Models Used")
    
    for model in get_ai_models_info():
        with st.expander(f"{model['name']} - {model['purpose']}"):
            st.markdown(f"""
            **Capabilities**: {model['capabilities']}
            
            **Technology**: {model.get('technology', model.get('architecture', 'Not specified'))}
            
            {f"**Paper**: [{model['paper']}]({model['paper']})" if 'paper' in model else ''}
            {f"**GitHub**: [{model['github']}]({model['github']})" if 'github' in model else ''}
            {f"**Author**: {model['author']}" if 'author' in model else ''}
            {f"**Implementation**: {model['implementation']}" if 'implementation' in model else ''}
            """)

def load_help_content():
    """Load help content from HELP.md file"""
    try:
        with open("HELP.md", "r") as f:
            return f.read()
    except Exception as e:
        return f"Error loading help content: {str(e)}"

def get_markdown_html(md_content):
    """Convert markdown to HTML for better styling"""
    # Using markdown library to convert md to html
    html = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
    return html

def show_help_section():
    """Display help content with navigation"""
    st.title("SonixMind Help")
    
    md_content = load_help_content()
    
    # Define help sections for navigation
    help_sections = {
        "Getting Started": "#getting-started",
        "Audio Extraction": "#audio-extraction",
        "Audio Processing": "#audio-processing",
        "Audio Mastering": "#audio-mastering",
        "YouTube Reference": "#youtube-reference",
        "Visualization Features": "#visualization-features",
        "Advanced Options": "#advanced-options",
        "Troubleshooting": "#troubleshooting"
    }
    
    # Initialize help section state if not exists
    if 'help_section' not in st.session_state:
        st.session_state.help_section = list(help_sections.keys())[0]
    
    # Create a sidebar for help navigation
    selected_section = st.sidebar.selectbox(
        "Navigate to section",
        list(help_sections.keys()),
        index=list(help_sections.keys()).index(st.session_state.help_section)
    )
    
    # Update session state if selection changes
    if selected_section != st.session_state.help_section:
        st.session_state.help_section = selected_section
    
    # Create navigation buttons for help sections
    cols = st.columns(4)
    for i, section in enumerate(help_sections.keys()):
        col_index = i % 4
        with cols[col_index]:
            if st.button(section, key=f"help_nav_{section}"):
                st.session_state.help_section = section
                st.experimental_rerun()
    
    st.markdown("---")
    
    # Display full help content using HTML components for better formatting
    html_content = get_markdown_html(md_content)
    
    # Extract the specific section requested
    section_id = help_sections[st.session_state.help_section].replace('#', '')
    
    # Use a custom component to display the HTML with proper styling
    st.components.v1.html(
        f"""
        <style>
            .help-content h1 {{ color: #1E88E5; font-size: 2em; margin-top: 1em; }}
            .help-content h2 {{ color: #0D47A1; font-size: 1.5em; margin-top: 1em; border-bottom: 1px solid #ddd; padding-bottom: 0.3em; }}
            .help-content h3 {{ color: #1565C0; font-size: 1.2em; margin-top: 0.8em; }}
            .help-content ul {{ padding-left: 2em; }}
            .help-content ol {{ padding-left: 2em; }}
            .help-content code {{ background-color: #f8f8f8; padding: 0.2em 0.4em; border-radius: 3px; font-family: monospace; }}
            .help-content pre {{ background-color: #f8f8f8; padding: 1em; overflow-x: auto; }}
            .help-content a {{ color: #1E88E5; text-decoration: none; }}
            .help-content a:hover {{ text-decoration: underline; }}
            .help-content table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
            .help-content th, .help-content td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .help-content th {{ background-color: #f8f8f8; }}
        </style>
        <div class="help-content">
            {html_content}
        </div>
        <script>
            // Scroll to the selected section
            document.addEventListener('DOMContentLoaded', function() {{
                const section = document.getElementById('{section_id}');
                if (section) {{
                    section.scrollIntoView({{behavior: 'smooth'}});
                }}
            }});
        </script>
        """,
        height=600,
        scrolling=True
    )
    
    st.markdown("---")
    
    # Download option for help content
    st.download_button(
        label="Download Help as Markdown",
        data=md_content,
        file_name="SonixMind_Help.md",
        mime="text/markdown"
    )

def add_footer():
    """Add a footer to the app"""
    author_info = get_author_info()
    app_info = get_app_info()
    
    footer_html = f"""
    <div style="text-align: center; margin-top: 30px; padding: 10px; border-top: 1px solid #e0e0e0;">
        <p style="color: #666; font-size: 14px;">
            {app_info['name']} v{app_info['version']} | 
            © {author_info['copyright_year']} {author_info['company']} | (C) Chirag Kansara/Ahmedabadi. All rights reserved.
        </p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

def apply_custom_css():
    """Apply custom CSS styling to the app"""
    st.markdown("""
    <style>
        /* Main app styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Headers styling */
        h1 {
            color: #0D47A1;
            font-weight: 700;
        }
        h2 {
            color: #1565C0;
            font-weight: 600;
        }
        h3 {
            color: #1976D2;
            font-weight: 500;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #1E88E5;
            color: white;
            border-radius: 4px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        .stButton>button:hover {
            background-color: #0D47A1;
        }
    </style>
    """, unsafe_allow_html=True)

def setup_page_config():
    """Set up the Streamlit page configuration"""
    app_info = get_app_info()
    
    # Set page title, icon, and other configurations
    st.set_page_config(
        page_title=app_info['name'],
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    apply_custom_css() 