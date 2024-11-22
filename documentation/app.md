# NPD 3D Model Generator App

This Streamlit application provides a user-friendly interface for generating and visualizing 3D models. It leverages the `crew` module (specifically the `CADModelGenerator` class) to handle the AI-powered code generation and model creation.

## Key Features:

* **Dual Modes:**
    * **Upload & Visualize:** Upload existing STL files and view them in an interactive 3D viewer.
    * **Generate from Text:** Describe a model in natural language and have the AI generate it.
* **Interactive 3D Viewer:**  Visualize models with customizable settings (color, material, rotation, height).
* **Code Display and Download:** View and download the generated Python CAD code.
* **STL File Download:** Download the generated 3D model in STL format.
* **Multiple LLM and CAD Library Support:**  Inherits the flexibility of the `crew` module, supporting different LLMs and CAD libraries.
* **User-Friendly Interface:** Streamlit-based UI for easy interaction and navigation.
* **Robust Error Handling and Logging:**  Provides informative error messages and detailed logs for troubleshooting.

## Running the App:

1.  **Install dependencies:**  Make sure you have installed the required packages listed in `requirements.txt`.
2.  **Set environment variables:** Create a `.env` file in the project's root directory and set your API keys for the LLMs.
3.  **Run the app:** `streamlit run app.py`

## Code Overview:

* **`ModelingApp` Class:**
    * **`__init__()`:** Initializes the app, sets up logging, and configures the Streamlit page.
    * **`setup_sidebar()`:** Creates the sidebar with controls for mode selection, generation settings, and visualization parameters.
    * **`main_content()`:**  Handles the main content area, displaying the appropriate UI elements based on the selected mode (upload or generate).
    * **`upload_mode()`:**  Manages STL file uploads and rendering.
    * **`generation_mode()`:**  Handles text input, model generation, code display, and download options.
    * **`render_stl()` and `render_stl_file()`:**  Renders STL data (from upload or file) in the 3D viewer.
    * **`generate_model()`:**  Calls the `CADModelGenerator` to generate the model and handles the results.
    * **`_get_api_key()`:** Retrieves the appropriate API key based on the selected LLM.
    * **`run()`:**  The main entry point for the application.


## UI Elements:

* **Sidebar:**
    * Mode Selection (Upload STL / Generate Model)
    * LLM Model Selection (e.g., Gemini, Llama)
    * CAD Library Selection (e.g., CadQuery, Build123D)
    * Visualization settings (color picker, material selection, auto-rotation toggle, height slider)

* **Main Content:**
    * **Upload Mode:** File uploader for STL files, 3D viewer.
    * **Generation Mode:** Text input area for model description, "Generate Model" button, code display area, download buttons for code and STL.

## Error Handling:

The app includes error handling to manage issues like invalid file uploads, incorrect API keys, and failures during model generation. Error messages are displayed to the user, and detailed logs are written to files for debugging.

## Logging:

Logs are written to files in the `logs/` directory. These logs contain detailed information about the application's execution, including user interactions, API calls, code generation steps, and error messages. This information is crucial for troubleshooting and monitoring the app's performance.


## Dependencies:

This application relies on several libraries, including `streamlit`, `crewai`, `streamlit-stl`, and others specified in `requirements.txt`. Ensure these are installed before running the app.