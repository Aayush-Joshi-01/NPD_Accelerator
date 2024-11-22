# NPD 3D Model Generator

This project provides a streamlined solution for generating 3D models using AI-powered code generation. It leverages the power of large language models (LLMs) to translate textual descriptions into executable Python code that creates 3D models using CAD libraries like CadQuery and Build123D.  The application offers two primary modes: uploading and visualizing existing STL files, and generating new models from textual descriptions.

## Key Features:

* **AI-Driven Model Generation:** Describe your desired model in natural language, and let the AI generate the corresponding CAD code.
* **Multiple LLM Backends:** Supports different LLMs (e.g., Gemini, Llama) to provide flexibility and choice.
* **CAD Library Support:**  Generates code compatible with popular Python CAD libraries like CadQuery and Build123D.
* **STL File Visualization:**  Upload and view STL files directly within the application using an integrated 3D viewer.
* **Interactive Controls:**  Customize visualization parameters like color, material, rotation, and height.
* **Code Download:** Download the generated Python code for further modification or integration into other projects.
* **STL File Download:** Download the generated 3D model in STL format.
* **Robust Error Handling:** Includes mechanisms for resolving code generation errors and refining the output.
* **Logging and Debugging:** Comprehensive logging facilitates troubleshooting and monitoring application behavior.


## Project Structure:

* **`app.py`:** The main Streamlit application file. Handles UI elements, user interactions, and orchestrates model generation.
* **`crew.py`:** Contains the core logic for CAD model generation using the CrewAI framework. Defines agents, tasks, and the generation process.
* **`configs/configuration.yaml`:** YAML file storing configuration parameters for agents, tasks, and other settings.
* **`code/`:**  Directory where generated Python code and STL files are stored.
* **`logs/`:** Directory for log files.
* **`assets/`:**  Directory for static assets (e.g., logo, styles).

## Setup and Installation:

1. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys:**

    Create a `.env` file in the project's root directory and add your API keys for the LLMs you want to use:

   ```
   GEMINI_API_KEY="your_gemini_api_key"
   GROQ_API_KEY="your_groq_api_key"
   ```

4. **Run the Streamlit app:**

   ```bash
   streamlit run app.py
   ```

## Using CrewAI as a Standalone Package:

The `crew.py` file can be used independently of the Streamlit application, allowing you to integrate the CAD model generation pipeline into other UIs or workflows.  Here's how:

1. **Install CrewAI (if not already installed):**

   ```bash
   pip install crewai
   ```

2. **Import necessary modules:**

   ```python
   from crew import CADModelGenerator  # Assuming you've renamed crew.py to crew.py
   import os
   ```

3. **Initialize the generator:**

   ```python
   api_key = os.environ.get("GEMINI_API_KEY") # Or GROQ_API_KEY, depending on your chosen LLM.
   model_name = "gemini" # or "llama"
   generator = CADModelGenerator(model_name, api_key)
   ```

4. **Generate a model:**

   ```python
   user_query = "A coffee mug with a handle"
   stl_path, generated_code, error = generator.generate(user_query)
   if error:
       print(f"Error generating model: {error}")
   else:
       print(f"STL file generated at: {stl_path}")
       print(f"Generated code:\n{generated_code}")
   # Integrate stl_path and generated_code into your UI.
   ```


5. **UI Integration:**

   * You'll need to handle user input (the model description) in your chosen UI framework.
   * Call `generator.generate()` with the user's input.
   * Display the generated code in a code editor or similar component.
   * Display or provide a download link for the generated STL file (`stl_path`).
   * Implement error handling to display any errors returned by `generator.generate()`.

By separating the CrewAI pipeline (`crew.py`) from the Streamlit app (`app.py`), you gain the flexibility to integrate this powerful 3D model generation functionality into any application of your choice. You'll need to manage UI elements, data flow, and error handling within your specific UI framework, but the core model generation logic is encapsulated and reusable.
