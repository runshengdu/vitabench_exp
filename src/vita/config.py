import os
import yaml
from pathlib import Path

_models_yaml_path = Path(__file__).parent / "models.yaml"
if os.environ.get("VITA_MODEL_CONFIG_PATH", None):
    _models_yaml_path = os.environ.get("VITA_MODEL_CONFIG_PATH")

if not os.path.exists(str(_models_yaml_path)):
    raise FileNotFoundError(
        f"Model configuration file ({_models_yaml_path}) dose not exists, you should create it first.")


def _deep_merge_dict(base_dict: dict, override_dict: dict) -> dict:
    result = base_dict.copy()

    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value

    return result


def _expand_env_vars(value):
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


try:
    with open(_models_yaml_path, 'r') as f:
        models_config_yaml = yaml.load(f, Loader=yaml.FullLoader)

    models_config_yaml = _expand_env_vars(models_config_yaml)

    default_model_config = models_config_yaml.get('default', {})

    models = {"default": default_model_config}
    for model in models_config_yaml.get('models', []):
        model_name = model['name']
        merged_config = _deep_merge_dict(default_model_config, model)
        models[model_name] = merged_config

    print(f"Available models: {list(models.keys())}")

except FileNotFoundError:
    print(f"Warning: models.yaml not found at {_models_yaml_path}")
    models = {}
except Exception as e:
    print(f"Error loading models.yaml: {e}")
    models = {}

# SIMULATION
DEFAULT_MAX_STEPS = 300
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_ERRORS = 10
DEFAULT_SEED = 300
DEFAULT_MAX_CONCURRENCY = 15
DEFAULT_NUM_TRIALS = 1
DEFAULT_SAVE_TO = None
DEFAULT_LOG_LEVEL = "DEBUG"
DEFAULT_LANGUAGE = "english"
DEFAULT_EVALUATION_TYPE = "trajectory"
DEFAULT_ENABLE_THINK_AGENT = True
DEFAULT_ENABLE_THINK_USER = False
DEFAULT_ENABLE_THINK_EVALUATOR = True

# LLM
DEFAULT_AGENT_IMPLEMENTATION = "llm_agent"
DEFAULT_USER_IMPLEMENTATION = "user_simulator"
DEFAULT_LLM_AGENT = "deepseek-reasoner"
DEFAULT_LLM_USER = "deepseek-chat"
DEFAULT_LLM_EVALUATOR = "deepseek-reasoner"
