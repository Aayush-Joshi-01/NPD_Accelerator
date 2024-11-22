# CrewAI CAD Model Generation

This module provides a robust and flexible pipeline for generating 3D CAD models from textual descriptions using the CrewAI framework. It leverages large language models (LLMs) to translate user queries into executable Python code compatible with popular CAD libraries like CadQuery and Build123D.

## Key Functionality:

* **AI-Powered Code Generation:** Translates natural language descriptions of 3D models into Python CAD code.
* **Multi-LLM Support:** Compatible with various LLMs (e.g., Gemini, Llama), configurable through initialization.
* **CAD Library Integration:** Generates code for CadQuery and Build123D, with the potential for extending to other libraries.
* **Error Handling and Resolution:**  Includes an iterative error resolution process to refine generated code and address potential issues.
* **Code Execution and STL Export:** Executes the generated code and exports the resulting 3D model in STL format.
* **Configuration Flexibility:** Uses a YAML configuration file to manage agent roles, goals, and other parameters.
* **Logging:** Detailed logging for monitoring and debugging the generation process.


## Core Components:

* **`CADModelGenerator` Class:**
    * **`__init__(model_name, api_key)`:** Initializes the generator with the chosen LLM and API key.
    * **`generate(query, language='cadquery')`:**  The main method for generating a CAD model. Takes a user query and the target modeling language (default: 'cadquery') as input. Returns the path to the generated STL file, the generated code, and any errors encountered.
    * Internal methods handle agent creation, task definition, code execution, error resolution, and file management.

* **Configuration (`configs/configuration.yaml`):**
    * Defines the roles, goals, and behavior of the CrewAI agents.
    * Allows customization of the generation process without modifying the core code.

* **Agents:**
    * **Planner:** Decomposes the user query into a structured modeling strategy.
    * **Code Generator:** Generates the Python CAD code based on the planner's strategy.
    * **Error Resolver:**  Analyzes and fixes errors in the generated code.

* **Tasks:**
    * Define specific goals for each agent within the CrewAI framework.


## Usage Example:

```python
import os
from crew import CADModelGenerator

# Load API key from environment variables (recommended)
api_key = os.environ.get("GEMINI_API_KEY")  # Replace with your actual key
model_name = "gemini" # or "llama"


generator = CADModelGenerator(model_name, api_key)


user_query = "A cylindrical container with a lid"
stl_path, generated_code, error = generator.generate(user_query)

if error:
    print(f"Error generating model: {error}")
else:
    print(f"STL file generated at: {stl_path}")
    print(f"Generated code:\n{generated_code}")

```

## Configuration:

The `configs/configuration.yaml` file allows you to customize various aspects of the generation process, including agent roles, goals, and other parameters.  Refer to the comments within the YAML file for details on available options.


## Extending Functionality:

* **Supporting New CAD Libraries:**  You can extend this module to support other Python CAD libraries by adding new language configurations and adapting the code generation templates.
* **Customizing Agents and Tasks:** The YAML configuration provides flexibility for tailoring agent behavior and task definitions to specific requirements.


## Logging:

Log messages are written to `cad_generator.log` and provide valuable insights into the generation process, including agent interactions, code generation steps, and error resolution attempts. This information can be useful for debugging and optimizing the pipeline.


## Dependencies:

This module depends on the `crewai` library, as well as other libraries specified in the project's `requirements.txt` file. Ensure these are installed before using the `CADModelGenerator`.