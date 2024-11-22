import streamlit as st
import logging
from datetime import datetime
import sys
from typing import Optional, Any, Dict
from pathlib import Path
from crew import CADModelGenerator
from streamlit_stl import stl_from_text, stl_from_file
import os
from io import BytesIO
from dotenv import load_dotenv

class ModelingApp:
    """
    A Streamlit application for 3D model generation and visualization.
    
    This class provides functionality for:
    - Uploading and viewing STL files
    - Generating 3D models using AI
    - Configuring visualization parameters
    - Managing application state
    
    Attributes:
        logger (logging.Logger): Logger instance for the application
        color (str): Current selected model color
        material (str): Current selected material type
        auto_rotate (bool): Auto-rotation setting for 3D viewer
        height (int): Height setting for 3D viewer
    """
    
    def __init__(self) -> None:
        """
        Initialize the ModelingApp with logging, page configuration, and session state.
        """
        self.logger = logging.getLogger(__name__)
        load_dotenv()
        self.setup_page_config()
        self.setup_logging()
        self.initialize_session_state()
        
    def setup_page_config(self) -> None:
        """
        Configure the Streamlit page layout and settings.
        
        Attempts to load custom CSS styles from assets/style.css if available.
        """
        self.logger.debug("Setting up page configuration")
        st.set_page_config(
            page_title="NPD 3D Model Generator",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        try:
            css_path = Path('assets/style.css')
            if css_path.exists():
                with open(css_path) as f:
                    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
                self.logger.info("Successfully loaded custom styles")
            else:
                self.logger.warning("Custom style file not found at: %s", css_path)
        except Exception as e:
            self.logger.error("Failed to load custom styles: %s", str(e), exc_info=True)
    
    def setup_logging(self) -> None:
        """
        Configure logging settings with file and console handlers.
        
        Creates a daily log file and sets up console logging with INFO level.
        """
        log_filename = f'logs/3d_modeling_app_{datetime.now().strftime("%Y%m%d")}.log'
        
        try:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_filename),
                    logging.StreamHandler(sys.stdout)
                ]
            )
            self.logger.info("Logging initialized successfully to: %s", log_filename)
        except Exception as e:
            print(f"Failed to initialize logging: {str(e)}")  # Fallback console output

    def initialize_session_state(self) -> None:
        """
        Initialize Streamlit session state variables with default values.
        
        Sets up state for:
        - Generation mode
        - LLM model selection
        - SDK selection
        - Generated code storage
        - STL file path storage
        - Success status tracking
        """
        self.logger.debug("Initializing session state")
        default_states: Dict[str, Any] = {
            'generation_mode': 'upload',
            'llm_model': 'gemini',
            'sdk': 'cadquery',
            'generated_code': None,
            'generated_stl_path': None,
            'success_status': False
        }
        
        for key, default_value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
                self.logger.debug("Initialized %s with default value: %s", key, default_value)

    def setup_sidebar(self) -> None:
        """
        Configure and display sidebar elements for application controls.
        
        Sets up:
        - Mode selection (Upload/Generate)
        - Generation settings (LLM model, SDK)
        - Visualization settings (color, material, rotation, height)
        """
        self.logger.debug("Setting up sidebar elements")
        try:
            with st.sidebar:
                st.sidebar.divider()
                st.sidebar.subheader("🛠️ 3D Modeling Controls")
                
                selected_mode = st.radio(
                    "Select Mode", 
                    ["Upload STL", "Generate Model"],
                    key="mode_selection"
                )
                self.logger.info("Mode selected: %s", selected_mode)
                
                st.session_state.generation_mode = selected_mode

                if selected_mode == "Generate Model":
                    st.subheader("Generation Settings")
                    st.selectbox("LLM Model", 
                               ["gemini", "llama"],
                               key='llm_model')
                    st.selectbox("Python SDK", 
                               ["cadquery", "build123d"],
                               key='sdk')
                    self.logger.debug("Generation settings updated - LLM: %s, SDK: %s", 
                                    st.session_state.llm_model, 
                                    st.session_state.sdk)
                
                st.sidebar.divider()
                st.subheader("📊 Visualization Settings")
                self.color = st.color_picker("Model Color", "#0099FF")
                self.material = st.selectbox("Material", ["material", "wireframe"])
                self.auto_rotate = st.toggle("Auto Rotation")
                self.height = st.slider("Height", 50, 1000, 300)
                st.sidebar.divider()
                
                self.logger.debug("Visualization settings updated - Color: %s, Material: %s, Height: %d", 
                                self.color, self.material, self.height)

        except Exception as e:
            self.logger.error("Error in sidebar setup: %s", str(e), exc_info=True)
            st.error("An error occurred while setting up the sidebar")

    def main_content(self) -> None:
        """
        Configure and display the main content area of the application.
        
        Handles:
        - Logo display
        - Title rendering
        - Mode-specific content (Upload/Generate)
        """
        self.logger.debug("Rendering main content")
        try:
            st.logo("assets/yash_logo.png", size='large')
            st.title("🚀 NPD 3D Model Generator")

            if st.session_state.generation_mode == "Upload STL":
                self.upload_mode()
            else:
                self.generation_mode()

            st.empty()

        except Exception as e:
            self.logger.error("Error in main content: %s", str(e), exc_info=True)
            st.error("An error occurred in the main application area")

    def upload_mode(self) -> None:
        """
        Handle STL file upload functionality.
        
        Provides:
        - File upload interface
        - STL file validation
        - 3D model rendering
        """
        self.logger.debug("Entering upload mode")
        try:
            file_input = st.file_uploader(
                "Upload STL File",
                type=["stl"],
                help="Select a .stl file to view"
            )
            
            if file_input:
                self.logger.info("Processing uploaded file: %s", file_input.name)
                self.render_stl(file_input)
                st.success(f"Successfully loaded: {file_input.name}")

        except Exception as e:
            self.logger.error("Error in upload mode: %s", str(e), exc_info=True)
            st.error("Error processing the uploaded file")

    def generation_mode(self) -> None:
        """
        Handle AI-based model generation functionality.
        
        Provides:
        - Prompt input interface
        - Model generation controls
        - Code and STL file visualization
        - Download options
        """
        self.logger.debug("Entering generation mode")
        try:
            prompt = st.text_area(
                "Enter your model description",
                height=150,
                placeholder="Describe the 3D model you want to generate...\nExample: A coffee mug with a handle and rounded edges"
            )
            
            generate_button = st.button("Generate Model", type="primary")
            
            if generate_button:
                if prompt:
                    st.session_state.generated_code = None
                    self.logger.info("Starting model generation with prompt: %s", prompt)
                    with st.spinner("Generating 3D model..."):
                        self.generate_model(prompt)
                else:
                    self.logger.warning("Generation attempted without prompt")
                    st.warning("Please enter a description before generating")
                    
            if st.session_state.generated_code:
                st.subheader("💻 Generated Code")
                st.code(st.session_state.generated_code, language='python', line_numbers=True)
                
                st.download_button(
                    label='Download Code',
                    data=st.session_state.generated_code,
                    file_name="generated_model_code.py",
                    mime="text/x-python"
                )
                
            if st.session_state.generated_stl_path and st.session_state.success_status:
                st.subheader("📦 STL File")
                self.render_stl_file(st.session_state.generated_stl_path)
                
                with open(st.session_state.generated_stl_path, "rb") as stl_file:
                    st.download_button(
                        label='Download STL File',
                        data=stl_file.read(),
                        file_name="generated_model.stl",
                        mime="model/stl"
                    )

        except Exception as e:
            self.logger.error("Error in generation mode: %s", str(e), exc_info=True)
            st.error("Error during model generation")

    def render_stl(self, file_input: BytesIO) -> None:
        """
        Render an uploaded STL file in the 3D viewer.

        Args:
            file_input (BytesIO): Uploaded STL file as bytes
        """
        self.logger.debug("Attempting to render uploaded STL")
        try:
            if file_input:
                stl_from_text(
                    text=file_input.getvalue(),
                    color=self.color,
                    material=self.material,
                    auto_rotate=self.auto_rotate,
                    height=self.height,
                    key='stl_viewer'
                )
                self.logger.info("Successfully rendered uploaded STL file")

        except Exception as e:
            self.logger.error("Error rendering uploaded STL: %s", str(e), exc_info=True)
            st.error("Error rendering the 3D model")

    def render_stl_file(self, file_path: str) -> None:
        """
        Render an STL file from disk in the 3D viewer.

        Args:
            file_path (str): Path to the STL file on disk
        """
        self.logger.debug("Attempting to render STL file from path: %s", file_path)
        try:
            if file_path:
                stl_from_file(
                    file_path=file_path,
                    color=self.color,
                    material=self.material,
                    auto_rotate=self.auto_rotate,
                    height=self.height,
                    key='stl_viewer'
                )
                self.logger.info("Successfully rendered STL file from path")

        except Exception as e:
            self.logger.error("Error rendering STL file: %s", str(e), exc_info=True)
            st.error("Error rendering the 3D model")

    def generate_model(self, query: str) -> None:
        """
        Generate a 3D model from a text description using AI.

        Args:
            query (str): Text description of the desired 3D model

        The function handles:
        - API key validation
        - Model generation
        - Error handling
        - Success/failure status updates
        """
        self.logger.info("Starting model generation with query: %s", query)
        try:
            api_key = self._get_api_key()
            if not api_key:
                self.logger.error("API key not found for %s", st.session_state.llm_model)
                st.error("API Key not found for selected model")
                return
            
            if not query.strip():
                self.logger.warning("Empty query provided")
                st.error("Please enter design requirements")
                return
            
            generator = CADModelGenerator(st.session_state.llm_model, api_key)
            self.logger.debug("Initialized CADModelGenerator")
            
            stl_path, generated_code, error = generator.generate(query)
            
            if error:
                lines = error.splitlines()
                last_line = lines[-1]
                self.logger.error("Model generation failed: %s", last_line)
                st.error(f"Model generation failed: {last_line}")
                st.session_state.success_status = False
                return
                
            st.session_state.generated_code = generated_code
            st.session_state.generated_stl_path = stl_path
            st.session_state.success_status = True
            
            self.logger.info("Model generated successfully")
            st.success("Model Generated Successfully!")
            
        except Exception as e:
            self.logger.error("Error in model generation: %s", str(e), exc_info=True)
            st.error(f"Unexpected error during model generation: {str(e)}")
            st.session_state.success_status = False

    def _get_api_key(self) -> Optional[str]:
        """
        Retrieve the appropriate API key based on selected LLM model.

        Returns:
            Optional[str]: API key if found, None otherwise
        """
        try:
            return os.environ["GEMINI_API_KEY"] if st.session_state.llm_model == "gemini" else os.environ["GROQ_API_KEY"]
        except KeyError as e:
            self.logger.error("API key not found in environment: %s", str(e))
            return None

    def run(self) -> None:
        """
        Main application entry point.
        
        Initializes and runs the complete application flow:
        - Sets up sidebar
        - Renders main content
        - Handles exceptions
        """
        self.logger.info("Starting ModelingApp")
        try:
            self.setup_sidebar()
            self.main_content()
            
        except Exception as e:
            self.logger.error("Critical application error: %s", str(e), exc_info=True)
            st.error("An unexpected error occurred")
            
        self.logger.info("ModelingApp session completed")

if __name__ == "__main__":
    app = ModelingApp()
    app.run()