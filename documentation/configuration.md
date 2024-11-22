# CrewAI CAD Model Generation Configuration

This YAML file configures the agents, tasks, and processes used by the `CADModelGenerator` class in the `crew` module.  It defines the roles, goals, backstories, and other parameters that govern the AI-driven code generation process.

## Structure:

```yaml
agents:
  # Agent configurations (planner, code_generator, error_resolver)
tasks:
  # Task configurations (planner, code_generator, error_resolver)
```

## Agents:

Each agent section defines the characteristics and behavior of a specific agent in the CrewAI framework.

* **`planner`:**
    * **`role_template`:**  A template for the planner's role, which is formatted with the chosen modeling language.
    * **`goal_template`:** Describes the planner's overall objective, also formatted with the language.
    * **`backstory_template`:** Provides background information and context for the planner's actions.
    * **`config`:**  Additional settings for the agent, such as verbosity and rate limiting.

* **`code_generator`:**
    * Similar structure to `planner`, with templates and configuration specific to code generation.

* **`error_resolver`:**
    * Configured to analyze and resolve errors in the generated code.

## Tasks:

Each task section outlines a specific goal for an agent.

* **`planner`:**
    * **`description_template`:** A template for describing the planning task, which is formatted with the user's query.
    * **`expected_output`:** Defines the expected format of the planner's output.

* **`code_generator`:**
    * **`description_template`:** A template for the code generation task, including language-specific instructions.
    * **`expected_output`:** Specifies that the generated code should be enclosed in a Python code block.

* **`error_resolver`:**
    * **`description_template`:**  Provides context for the error resolver, including the query, error message, previous errors, and current code.
    * **`expected_output`:**  Specifies that the corrected code should be returned in a code block.


## Customization:

* **Agent Roles and Goals:** You can modify the templates for agent roles, goals, and backstories to fine-tune their behavior and the style of the generated code.
* **Task Descriptions:**  Adjust task descriptions to provide more specific instructions or constraints.
* **Agent Configuration:** The `config` section within each agent allows you to control parameters like verbosity, rate limiting, and emotional intelligence.
* **Adding New Languages:**  To support additional CAD libraries, you would add new entries under the `agents` and `tasks` sections, adapting the templates and configuration to the new language.


## Example:

```yaml
agents:
  planner:
    role_template: "Strategic {language} Modeling Architect"
    # ... other planner configurations
  code_generator:
    role_template: "Advanced {language} Code Generation Specialist"
    # ... other code generator configurations
  # ... error_resolver configuration

tasks:
  planner:
    description_template: |
      Comprehensive Query Analysis and Modeling Strategy Development
      # ... other planner task configurations
  code_generator:
    description_template: |
      Code Generation Task for {language} 3D Modeling
      # ... other code generator task configurations
  # ... error_resolver task configuration
```
