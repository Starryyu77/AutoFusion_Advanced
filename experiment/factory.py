"""
Factory Pattern for Component Creation
--------------------------------------
Centralized factory for creating experiment components.
"""

from typing import Dict, Any, Optional
import importlib


def _import_class(class_path: str):
    """动态导入类"""
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def create_controller(name: str, config: Dict[str, Any]) -> "BaseController":
    """
    Controller工厂

    Args:
        name: 控制器名称
        config: 配置字典

    Returns:
        BaseController实例
    """
    controllers = {
        'ppo': 'controllers.ppo.PPOController',
        'grpo': 'controllers.grpo.GRPOController',
        'gdpo': 'controllers.gdpo.GDPOController',
        'evolution': 'controllers.evolution.EvolutionController',
        'cmaes': 'controllers.cmaes.CMAESController',
        'random': 'controllers.random.RandomController',
    }

    if name not in controllers:
        raise ValueError(f"Unknown controller: {name}. Available: {list(controllers.keys())}")

    controller_class = _import_class(controllers[name])
    return controller_class(config)


def create_generator(name: str, llm_client: Any, config: Dict[str, Any]) -> "BaseGenerator":
    """
    Generator工厂

    Args:
        name: 生成器名称
        llm_client: LLM客户端
        config: 配置字典

    Returns:
        BaseGenerator实例
    """
    generators = {
        'cot': 'generators.cot.ChainOfThoughtGenerator',
        'fewshot': 'generators.fewshot.FewShotGenerator',
        'critic': 'generators.critic.CriticGenerator',
        'shape': 'generators.shape.ShapeConstraintGenerator',
        'roleplay': 'generators.roleplay.RolePlayGenerator',
    }

    if name not in generators:
        raise ValueError(f"Unknown generator: {name}. Available: {list(generators.keys())}")

    generator_class = _import_class(generators[name])
    return generator_class(llm_client, config)


def create_evaluator(name: str, config: Dict[str, Any]) -> "BaseEvaluator":
    """
    Evaluator工厂

    Args:
        name: 评估器名称
        config: 配置字典

    Returns:
        BaseEvaluator实例
    """
    evaluators = {
        'sandbox': 'evaluators.surgical_sandbox.SurgicalSandboxEvaluator',
    }

    if name not in evaluators:
        raise ValueError(f"Unknown evaluator: {name}. Available: {list(evaluators.keys())}")

    evaluator_class = _import_class(evaluators[name])
    return evaluator_class(config)


def create_reward(config: Dict[str, Any]) -> "MultiObjectiveReward":
    """
    创建奖励函数

    Args:
        config: 配置字典

    Returns:
        MultiObjectiveReward实例
    """
    from base import MultiObjectiveReward
    return MultiObjectiveReward(config)


def create_experiment_components(
    controller_name: str,
    generator_name: str,
    evaluator_name: str,
    config: Dict[str, Any],
    llm_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    创建完整的实验组件集合

    Args:
        controller_name: 控制器名称
        generator_name: 生成器名称
        evaluator_name: 评估器名称
        config: 配置字典
        llm_client: LLM客户端 (可选)

    Returns:
        包含所有组件的字典
    """
    components = {
        'controller': create_controller(controller_name, config.get('controller', {})),
        'generator': create_generator(generator_name, llm_client, config.get('generator', {})),
        'evaluator': create_evaluator(evaluator_name, config.get('evaluator', {})),
        'reward': create_reward(config.get('reward', {})),
    }

    return components


class ExperimentFactory:
    """实验工厂类 - 用于批量创建实验"""

    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config

    def create_phase0_scaffold(self) -> Dict[str, Any]:
        """创建Phase 0 scaffold组件 (PPO + CoT)"""
        return create_experiment_components(
            controller_name='ppo',
            generator_name='cot',
            evaluator_name='sandbox',
            config=self.base_config,
            llm_client=None,
        )

    def create_prompt_comparison(self, prompt_name: str) -> Dict[str, Any]:
        """创建Prompt对比实验组件"""
        return create_experiment_components(
            controller_name='ppo',
            generator_name=prompt_name,
            evaluator_name='sandbox',
            config=self.base_config,
            llm_client=None,
        )

    def create_controller_comparison(self, controller_name: str, prompt_name: str = 'cot') -> Dict[str, Any]:
        """创建Controller对比实验组件"""
        return create_experiment_components(
            controller_name=controller_name,
            generator_name=prompt_name,
            evaluator_name='sandbox',
            config=self.base_config,
            llm_client=None,
        )
