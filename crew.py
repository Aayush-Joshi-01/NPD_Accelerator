import re
import traceback
import logging
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional, Any
from crewai_tools import CodeDocsSearchTool
import crewai
from crewai import Agent, Task, Crew, Process
import os
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cad_generator.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class CADModelGenerator:
    """
    A class to generate CAD models using various LLM backends and modeling languages.
    
    This class orchestrates the generation of CAD models by coordinating between different
    agents (planners, code generators, error resolvers) using the CrewAI framework.
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None) -> None:
        """
        Initialize the CAD Model Generator with specified LLM backend.
        
        Args:
            model_name: Name of the LLM model to use ('gemini' or 'llama')
            api_key: API key for the specified model. If None, will try to load from environment
        
        Raises:
            ValueError: If model_name is not supported
            EnvironmentError: If required API keys are not found in environment
        """
        logger.info(f"Initializing CADModelGenerator with model: {model_name}")
        self.api_key = api_key
        self.model_name = model_name
        
        try:
            if model_name == "gemini":
                self.llm = crewai.LLM(
                    model="gemini/gemini-1.5-pro-002",
                    api_key=api_key if api_key is not None else os.environ["GEMINI_API_KEY"]
                )
            elif model_name == "llama":
                self.llm = crewai.LLM(
                    model="groq/llama3-8b-8192",
                    temperature=0.5,
                    api_key=api_key if api_key is not None else os.environ["GROQ_API_KEY"]
                )
            else:
                raise ValueError(f"Unsupported model: {model_name}")
        except KeyError as e:
            logger.error(f"Missing required API key for model {model_name}")
            raise EnvironmentError(f"Missing required API key: {str(e)}")
        
        self.filename = "code/generation.py"
        self.log_filename = "logs/errors.log"
        self.config_path = "configs/configuration.yaml"
        
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info("Successfully loaded configuration file")
        except Exception as e:
            logger.error(f"Failed to load configuration file: {str(e)}")
            raise
        
        self.languages = self._get_languages()
        self.max_iteration = 15
        flag = False
        if flag:
            self.coder_tool = [self._create_doc_search_tool(self.languages['cadquery'])]
        else:
            self.coder_tool = []

    def _get_languages(self) -> Dict[str, SimpleNamespace]:
        """
        Define supported modeling languages and their configurations.
        
        Returns:
            Dictionary mapping language names to their configurations as SimpleNamespace objects
        """
        logger.debug("Initializing supported modeling languages")
        return {
            "cadquery": SimpleNamespace(**{
                "name": "cadquery",
                "package_name": "cadquery",
                "import_command": "import cadquery as cq",
                "export_command": "result_shape.exportStl",
                "documentation_url": "https://cadquery.readthedocs.io/en/latest/index.html"
            }),
            "build123d": SimpleNamespace(**{
                "name": "build123d",
                "package_name": "build123d",
                "import_command": "from build123d import *",
                "export_command": "export_stl",
                "documentation_url": "https://build123d.readthedocs.io/en/latest/index.html"
            })
        }

    def _save_code_to_file(self, code: str, filename: str) -> None:
        """
        Save generated code to a Python file.
        
        Args:
            code: Python code to save
            filename: Target filename
            
        Raises:
            IOError: If unable to write to file
        """
        try:
            with open(filename, "w") as file:
                file.write(code)
            logger.debug(f"Successfully saved code to {filename}")
        except IOError as e:
            logger.error(f"Failed to save code to {filename}: {str(e)}")
            raise

    def _save_output_to_log(self, output: Optional[str], log_filename: str) -> None:
        """
        Save execution output to log file.
        
        Args:
            output: Output string to log
            log_filename: Target log filename
        """
        if output is not None:
            try:
                with open(log_filename, "a") as log_file:
                    log_file.write(output)
                logger.debug(f"Successfully wrote output to {log_filename}")
            except IOError as e:
                logger.error(f"Failed to write to log file {log_filename}: {str(e)}")

    def _run_executer(self, filename: str) -> Optional[str]:
        """
        Execute Python file and capture any errors.
        
        Args:
            filename: Python file to execute
            
        Returns:
            Error message if execution fails, None otherwise
        """
        logger.info(f"Executing Python file: {filename}")
        try:
            exec(open(filename).read())
            logger.info("Successfully executed code")
            return None
        except Exception as e:
            error_message = traceback.format_exc()
            logger.error(f"Error executing code: {error_message}")
            return error_message

    def _format_template(self, template: str, **kwargs) -> str:
        """
        Format template string with provided kwargs.
        
        Args:
            template: Template string to format
            **kwargs: Keyword arguments for formatting
            
        Returns:
            Formatted string
            
        Raises:
            KeyError: If required formatting keys are missing
        """
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing required template parameter: {str(e)}")
            raise

    def _create_agents(self, 
                      modelling_language: SimpleNamespace,
                      llm: Any,
                      coder_tool: List[Any] = []) -> SimpleNamespace:
        """
        Create agents from configuration.
        
        Args:
            modelling_language: Language configuration
            llm: Language model instance
            coder_tool: List of tools for code generation
            
        Returns:
            SimpleNamespace containing created agents
        """
        logger.info(f"Creating agents for {modelling_language.name}")
        agents = {}
        
        try:
            # Create planner agent
            planner_config = self.config['agents']['planner']
            agents['agent_planner'] = Agent(
                role=self._format_template(planner_config['role_template'], 
                                         language=modelling_language.name.upper()),
                goal=self._format_template(planner_config['goal_template'], 
                                         language=modelling_language.name),
                backstory=self._format_template(planner_config['backstory_template'], 
                                              language=modelling_language.name),
                llm=llm,
                **planner_config['config']
            )
            logger.debug("Successfully created planner agent")

            # Create code generator agent
            code_gen_config = self.config['agents']['code_generator']
            agents['agent_code_generator'] = Agent(
                role=self._format_template(code_gen_config['role_template'], 
                                         language=modelling_language.name.upper()),
                goal=self._format_template(code_gen_config['goal_template'], 
                                         language=modelling_language.name),
                backstory=self._format_template(code_gen_config['backstory_template'],
                                              language=modelling_language.name,
                                              package_name=modelling_language.package_name),
                llm=llm,
                tools=coder_tool,
                **code_gen_config['config']
            )
            logger.debug("Successfully created code generator agent")
            
            return SimpleNamespace(**agents)
        except Exception as e:
            logger.error(f"Failed to create agents: {str(e)}")
            raise

    def _get_tasks(self,
                  user_query: str,
                  agents: SimpleNamespace,
                  modelling_language: SimpleNamespace) -> SimpleNamespace:
        """
        Create tasks from configuration.
        
        Args:
            user_query: User's design requirements
            agents: Available agents
            modelling_language: Language configuration
            
        Returns:
            SimpleNamespace containing created tasks
        """
        logger.info("Creating tasks for model generation")
        tasks = {}
        
        try:
            # Create planner task
            planner_config = self.config['tasks']['planner']
            tasks['planner_task'] = Task(
                description=self._format_template(planner_config['description_template'],
                                               user_query=user_query),
                agent=agents.agent_planner,
                expected_output=planner_config['expected_output']
            )
            logger.debug("Successfully created planner task")

            # Create code generator task
            code_gen_config = self.config['tasks']['code_generator']
            tasks['initial_code_generation_task'] = Task(
                description=self._format_template(code_gen_config['description_template'],
                                               language=modelling_language.name.upper(),
                                               import_command=modelling_language.import_command,
                                               export_command=modelling_language.export_command),
                agent=agents.agent_code_generator,
                context=[tasks['planner_task']],
                expected_output=code_gen_config['expected_output']
            )
            logger.debug("Successfully created code generator task")
            
            return SimpleNamespace(**tasks)
        except Exception as e:
            logger.error(f"Failed to create tasks: {str(e)}")
            raise

    def _get_error_resolvers(self,
                           query: str,
                           llm: Any,
                           modelling_language: SimpleNamespace,
                           error: str,
                           prev_error: List[str],
                           code: str,
                           coder_tool: List[Any] = []) -> Tuple[SimpleNamespace, SimpleNamespace]:
        """
        Create error resolver agent and task from configuration.
        
        Args:
            query: Original user query
            llm: Language model instance
            modelling_language: Language configuration
            error: Current error message
            prev_error: List of previous errors
            code: Current code
            coder_tool: List of tools for code generation
            
        Returns:
            Tuple of (resolver agents, resolver tasks)
        """
        logger.info("Creating error resolver agent and task")
        try:
            # Create error resolver agent
            error_resolver_config = self.config['agents']['error_resolver']
            error_resolver_agent = Agent(
                role=self._format_template(error_resolver_config['role_template'],
                                         language=modelling_language.name),
                goal=self._format_template(error_resolver_config['goal_template'],
                                         language=modelling_language.name),
                backstory=self._format_template(error_resolver_config['backstory_template'],
                                              language=modelling_language.name,
                                              documentation_url=modelling_language.documentation_url),
                llm=llm,
                tools=coder_tool,
                **error_resolver_config['config']
            )
            logger.debug("Successfully created error resolver agent")

            # Create error resolver task
            error_task_config = self.config['tasks']['error_resolver']
            error_resolver_task = Task(
                description=self._format_template(error_task_config['description_template'],
                                               language=modelling_language.name,
                                               query=query,
                                               error=error,
                                               previous_errors=', '.join(prev_error) if prev_error else 'None',
                                               code=code,
                                               import_command=modelling_language.import_command,
                                               export_command=modelling_language.export_command),
                agent=error_resolver_agent,
                expected_output=error_task_config['expected_output']
            )
            logger.debug("Successfully created error resolver task")

            return (SimpleNamespace(**{"error_resolver_agent": error_resolver_agent}),
                    SimpleNamespace(**{"error_resolver_task": error_resolver_task}))
        except Exception as e:
            logger.error(f"Failed to create error resolvers: {str(e)}")
            raise

    def _process_result(self, result: Any) -> Tuple[Optional[str], str]:
        """
        Process the result from code generation or error resolution.
        
        Args:
            result: Result from crew execution
            
        Returns:
            Tuple of (error message or None, extracted code)
        """
        logger.info("Processing generation result")
        try:
            match = re.search(r"```(?:python)?\n(.*?)```", str(result.raw), re.DOTALL | re.IGNORECASE)
            if not match:
                logger.error("Failed to extract code from result")
                raise ValueError("No code block found in result")
            
            code = match.group(1).strip()
            self._save_code_to_file(code, self.filename)
            error = self._run_executer(self.filename)
            self._save_output_to_log(error, self.log_filename)
            return error, code
        except Exception as e:
            logger.error(f"Error processing result: {str(e)}")
            raise

    def _create_doc_search_tool(self, modelling_language: SimpleNamespace) -> CodeDocsSearchTool:
        """
        Create documentation search tool for the specified modeling language.
        
        Args:
            modelling_language: Language configuration
            
        Returns:
            Configured CodeDocsSearchTool instance
        """
        logger.info(f"Creating documentation search tool for {modelling_language.name}")
        try:
            return CodeDocsSearchTool(
                docs_url=modelling_language.documentation_url,
                config=dict(
                    llm=dict(
                        provider="google",
                        config=dict(
                            api_key=os.environ["GEMINI_API_KEY"],
                            model="gemini/gemini-1.5-flash-002",
                            temperature=0.5,
                            top_p=1,
                        ),
                    ),
                    embedder=dict(
                        provider="google",
                        config=dict(
                            model="models/embedding-001",
                            task_type="retrieval_document",
                            title="Embeddings"
                        ),
                    ),
                )
            )
        except Exception as e:
            logger.error(f"Failed to create doc search tool: {str(e)}")
            raise
        
    def generate(self, query: str, language: str = 'cadquery') -> Tuple[str, str]:
        """
        Main method to generate a CAD model
        
        :param query: Design requirements for the model
        :param language: Modeling language (default: cadquery)
        :return: Tuple of (STL file path, generated code)
        """
        # Select language configuration
        lang_config = self.languages[language]
        
        # Create agents and tasks
        agents = self._create_agents(lang_config, self.llm)
        tasks = self._get_tasks(query, agents, lang_config)
        
        # Create crew for generation
        crew_generation = Crew(
            agents=list(vars(agents).values()),
            tasks=list(vars(tasks).values()),
            process=Process.sequential,
            verbose=True
        )
        
        # Generate initial result
        try:
            result = crew_generation.kickoff()
        except Exception as e:
            print(f"Error in processing query: {str(e)}")
            raise

        # Extract and process code
        error, code = self._process_result(result)
        prev_errors = []
        iterator = 0
        
        while error is not None and iterator < self.max_iteration:
            try:
                prev_errors.append(error)
                input_param = SimpleNamespace(**{
                    "current_error": error,
                    "prev_errors": prev_errors,
                    "code": code
                })
                resolver_agents, resolver_tasks = self._get_error_resolvers(query, self.llm, self._get_languages()[language], input_param.current_error, input_param.prev_errors, input_param.code)
                crew_error_resolvers = Crew(
                        agents=list(vars(resolver_agents).values()),
                        tasks=list(vars(resolver_tasks).values()),
                        process=Process.sequential,
                        verbose=True
                    )
                result = crew_error_resolvers.kickoff()
            except Exception as e:
                print(f"Error in processing query: {str(e)}")
            error, code = self._process_result(result)
            iterator += 1
        # Assume STL file is always named 'generation.stl'
        return 'code/generation.stl', code, error